import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

_PROJ_ROOT = Path(__file__).resolve().parent.parent
_WIN_CA = _PROJ_ROOT / ".venv" / "win_roots.pem"
if _WIN_CA.exists() and "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = str(_WIN_CA)
os.environ.setdefault("TORCH_HOME", str(_PROJ_ROOT / ".cache" / "torch"))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from feot_dataset import (
    compute_crop_size,
    crop_norm_to_frame_box,
    extract_square_crop,
    frame_box_to_crop_norm,
    load_sequence_meta,
    make_gaussian_heatmap,
    pil_to_norm_tensor,
)
from losses import box_iou_and_giou, gaussian_focal_loss
from models import (
    ResNet50LSTMTracker,
    ResNet50TrackerCNN,
    TemporalLSTMHead,
    compute_cnn_summary_stats,
)
from train_cnn import compute_center_err, compute_iou_xywh


def apply_synthetic_occluder(crop_img: Image.Image, target_box_norm: np.ndarray, rng: random.Random) -> Image.Image:
    """
    Paint a realistic occluder rectangle (copied from another part of the search crop or
    a dark/textured block) over the target area so the frozen ResNet-50 produces authentic
    occluded visual features and heatmaps.
    """
    arr = np.array(crop_img, copy=True)
    H, W, _ = arr.shape
    cx, cy, nw, nh = target_box_norm
    bw = max(24, int(nw * W * rng.uniform(1.1, 1.8)))
    bh = max(24, int(nh * H * rng.uniform(1.1, 1.8)))
    x1 = max(0, int(cx * W - bw / 2))
    y1 = max(0, int(cy * H - bh / 2))
    x2 = min(W, x1 + bw)
    y2 = min(H, y1 + bh)
    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:
        return crop_img

    mode = rng.random()
    if mode < 0.65 and W - pw > 4 and H - ph > 4:
        # Copy a background patch from elsewhere in the same frame (hard occluder!)
        sx = rng.randint(0, W - pw)
        sy = rng.randint(0, H - ph)
        arr[y1:y2, x1:x2] = arr[sy:sy + ph, sx:sx + pw]
    else:
        # Solid/noisy dark occluder (pole/wall/shadow)
        color = np.array([rng.randint(20, 140) for _ in range(3)], dtype=np.uint8)
        noise = np.random.randint(-12, 13, size=(ph, pw, 3))
        arr[y1:y2, x1:x2] = np.clip(color + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


@torch.no_grad()
def extract_sequence_features(
    cnn: ResNet50TrackerCNN,
    data_root: Path,
    seq_names: List[str],
    device: torch.device,
    cache_path: Path,
    synth_occlusion: bool = False,
    seed: int = 42,
    batch_size: int = 64,
) -> List[Dict[str, torch.Tensor]]:
    """
    Run frozen Stage-1 ResNet-50 over sequences to extract per-frame visual/spatial features
    plus realistic anchor kinematics. Cached to disk for fast, repeatable LSTM training.
    """
    if cache_path.exists():
        print(f"Loading cached CNN features from {cache_path}", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    cnn.eval()
    ds_dir = data_root / "FEOT_dataset"
    rng = random.Random(seed)
    cached_seqs = []
    t0 = time.time()

    for idx, name in enumerate(seq_names, 1):
        meta = load_sequence_meta(ds_dir / name)
        frames = meta["frames"]
        boxes = meta["boxes"]       # (N, 4) xywh
        occs = meta["occs"]         # (N,)
        ts = meta["timestamps"]     # (N,)
        N = len(frames)
        if N < 3:
            continue

        # Encode frame-0 template once
        img0 = Image.open(frames[0]).convert("RGB")
        b0 = boxes[0]
        tcx, tcy = b0[0] + 0.5 * b0[2], b0[1] + 0.5 * b0[3]
        t_side = compute_crop_size(b0[2], b0[3], scale_factor=2.0, min_size=64.0)
        tmpl_crop, _ = extract_square_crop(img0, tcx, tcy, t_side, 128)
        tmpl_t = pil_to_norm_tensor(tmpl_crop).unsqueeze(0).to(device)
        tmpl_enc = cnn.encode_template(tmpl_t)

        # Pick synthetic occlusion intervals if enabled (4-12 contiguous frames, ~28% coverage)
        synth_mask = np.zeros(N, dtype=bool)
        if synth_occlusion and N > 12:
            cur = rng.randint(4, min(14, N - 6))
            while cur < N - 5:
                max_span = min(12, N - cur - 1)
                if max_span < 4:
                    break
                span = rng.randint(4, max_span)
                synth_mask[cur:cur + span] = True
                cur += span + rng.randint(10, 28)

        # Parallelize PNG decoding + cropping across CPU threads for fast extraction
        centers = np.stack([boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]], axis=-1)
        # Pre-generate deterministic per-frame jitter & seeds so threading is reproducible
        frame_jobs = []
        for t in range(1, N):
            jx = rng.uniform(-0.06, 0.06) if synth_occlusion else 0.0
            jy = rng.uniform(-0.06, 0.06) if synth_occlusion else 0.0
            fseed = rng.randint(0, 2**31 - 1)
            frame_jobs.append((t, jx, jy, fseed))

        def _prep_frame(job):
            t, jx, jy, fseed = job
            prev_box = boxes[t - 1]
            acx = prev_box[0] + 0.5 * prev_box[2]
            acy = prev_box[1] + 0.5 * prev_box[3]
            s_side = compute_crop_size(prev_box[2], prev_box[3], scale_factor=4.0, min_size=256.0)
            acx += jx * s_side
            acy += jy * s_side

            with Image.open(frames[t]) as im:
                img_t = im.convert("RGB")
            srch_crop, crop_meta = extract_square_crop(img_t, acx, acy, s_side, 256)
            gt_norm = frame_box_to_crop_norm(boxes[t], crop_meta)
            gt_norm = np.clip(gt_norm, 0.01, 0.99).astype(np.float32)

            if synth_mask[t]:
                srch_crop = apply_synthetic_occluder(srch_crop, gt_norm, random.Random(fseed))

            dt_cur = max(5000.0, ts[t] - ts[t - 1])
            dt_norm = float(np.clip(dt_cur / 40000.0, 0.25, 3.0))
            if t >= 2:
                dt_prev = max(5000.0, ts[t - 1] - ts[t - 2])
                vx_per_40ms = (centers[t - 1, 0] - centers[t - 2, 0]) * (40000.0 / dt_prev)
                vy_per_40ms = (centers[t - 1, 1] - centers[t - 2, 1]) * (40000.0 / dt_prev)
            else:
                vx_per_40ms, vy_per_40ms = 0.0, 0.0

            side_used = crop_meta[2]
            vel_crop = np.array([vx_per_40ms / side_used, vy_per_40ms / side_used], dtype=np.float32)
            size_crop = np.array([prev_box[2] / side_used, prev_box[3] / side_used], dtype=np.float32)
            return (
                pil_to_norm_tensor(srch_crop),
                gt_norm,
                vel_crop,
                size_crop,
                [dt_norm],
                int(occs[t]),
                1 if synth_mask[t] else 0,
            )

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as ex:
            prepped = list(ex.map(_prep_frame, frame_jobs))

        srch_tensors = [r[0] for r in prepped]
        gt_norm_list = [r[1] for r in prepped]
        vel_list = [r[2] for r in prepped]
        size_list = [r[3] for r in prepped]
        dt_list = [r[4] for r in prepped]
        occ_list = [r[5] for r in prepped]
        synth_list = [r[6] for r in prepped]

        # Batched GPU ResNet-50 forward pass
        feats_all, boxes_all, hms_all, stats_all = [], [], [], []
        M = len(srch_tensors)
        for st in range(0, M, batch_size):
            batch_x = torch.stack(srch_tensors[st:st + batch_size]).to(device)
            b_sz = batch_x.shape[0]
            k_exp = tmpl_enc[0].expand(b_sz, -1, -1, -1)
            g_exp = tmpl_enc[1].expand(b_sz, -1, -1, -1)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = cnn.forward_with_template((k_exp, g_exp), batch_x)
                stats = compute_cnn_summary_stats(out["heatmap_logits"])
            feats_all.append(out["frame_feat"].float().cpu())
            boxes_all.append(out["box"].float().cpu())
            hms_all.append(out["heatmap_logits"].float().cpu())
            stats_all.append(stats.float().cpu())

        cached_seqs.append({
            "name": name,
            "frame_feat": torch.cat(feats_all, dim=0),             # (M, 512)
            "cnn_box": torch.cat(boxes_all, dim=0),                # (M, 4)
            "cnn_hm_logits": torch.cat(hms_all, dim=0),            # (M, 1, 16, 16)
            "cnn_stats": torch.cat(stats_all, dim=0),              # (M, 3)
            "prev_vel_crop": torch.tensor(np.array(vel_list), dtype=torch.float32),
            "prev_size_crop": torch.tensor(np.array(size_list), dtype=torch.float32),
            "dt_norm": torch.tensor(np.array(dt_list), dtype=torch.float32),
            "gt_box": torch.tensor(np.array(gt_norm_list), dtype=torch.float32),
            "occ": torch.tensor(occ_list, dtype=torch.int32),
            "synth_occ": torch.tensor(synth_list, dtype=torch.int32),
        })
        if idx % 20 == 0 or idx == len(seq_names):
            print(f"  Extracted {idx}/{len(seq_names)} seqs ({time.time()-t0:.1f}s)", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached_seqs, cache_path)
    print(f"Saved cached CNN features to {cache_path}", flush=True)
    return cached_seqs


class FEOTFeatureClipDataset(Dataset):
    """Samples temporal clips of length T from pre-extracted ResNet-50 sequence features."""

    def __init__(
        self,
        sequences: List[Dict[str, torch.Tensor]],
        clip_len: int = 24,
        clips_per_epoch: int = 2400,
        augment: bool = True,
    ):
        self.sequences = [s for s in sequences if s["frame_feat"].shape[0] >= 8]
        self.clip_len = clip_len
        self.clips_per_epoch = clips_per_epoch
        self.augment = augment

        # Pre-index occlusion-rich start positions for oversampling occluded clips
        self.occ_starts: List[Tuple[int, int]] = []
        for si, s in enumerate(self.sequences):
            M = s["frame_feat"].shape[0]
            occ_mask = (s["occ"] >= 50) | (s["synth_occ"] == 1)
            occ_idxs = torch.where(occ_mask)[0].tolist()
            for oi in occ_idxs:
                st = max(0, min(M - clip_len, oi - clip_len // 2))
                self.occ_starts.append((si, st))

    def __len__(self) -> int:
        if self.augment:
            return self.clips_per_epoch
        # Deterministic sliding windows for validation
        total = 0
        for s in self.sequences:
            M = s["frame_feat"].shape[0]
            total += max(1, (M + self.clip_len - 1) // self.clip_len)
        return total

    def _get_window(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        if self.augment:
            if self.occ_starts and random.random() < 0.65:
                si, st = random.choice(self.occ_starts)
                s = self.sequences[si]
                M = s["frame_feat"].shape[0]
                st = max(0, min(max(0, M - self.clip_len), st + random.randint(-4, 4)))
                return s, st
            s = self.sequences[idx % len(self.sequences)]
            M = s["frame_feat"].shape[0]
            st = random.randint(0, max(0, M - self.clip_len))
            return s, st
        else:
            cur = idx
            for s in self.sequences:
                M = s["frame_feat"].shape[0]
                n_win = max(1, (M + self.clip_len - 1) // self.clip_len)
                if cur < n_win:
                    st = min(max(0, M - self.clip_len), cur * self.clip_len)
                    return s, st
                cur -= n_win
            return self.sequences[-1], 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s, st = self._get_window(idx)
        M = s["frame_feat"].shape[0]
        T = self.clip_len
        end = min(M, st + T)

        def slc(key: str) -> torch.Tensor:
            t = s[key][st:end].clone()
            if t.shape[0] < T:
                pad_n = T - t.shape[0]
                pad = t[-1:].repeat(pad_n, *([1] * (t.ndim - 1)))
                t = torch.cat([t, pad], dim=0)
            return t

        frame_feat = slc("frame_feat")
        cnn_box = slc("cnn_box")
        cnn_hm = slc("cnn_hm_logits")
        cnn_stats = slc("cnn_stats")
        prev_vel = slc("prev_vel_crop")
        prev_size = slc("prev_size_crop")
        dt_norm = slc("dt_norm")
        gt_box = slc("gt_box")
        occ = slc("occ")
        synth_occ = slc("synth_occ")

        # Additional feature-level occlusion & noise augmentation during training
        if self.augment:
            prev_vel = prev_vel + torch.randn_like(prev_vel) * 0.004
            prev_size = (prev_size * (1.0 + torch.randn_like(prev_size) * 0.03)).clamp(0.02, 0.99)
            # 25% chance to drop a random contiguous 3-8 frame span of CNN features (teaches pure motion coasting)
            if random.random() < 0.25:
                d_len = random.randint(3, 8)
                d_st = random.randint(3, T - d_len)
                frame_feat[d_st:d_st + d_len] *= 0.1
                cnn_hm[d_st:d_st + d_len] = -3.0
                cnn_stats[d_st:d_st + d_len, 0] = 0.05  # low max_prob
                cnn_stats[d_st:d_st + d_len, 2] = 0.95  # high entropy
                cnn_box[d_st:d_st + d_len, :2] = torch.rand(d_len, 2) * 0.8 + 0.1
                synth_occ[d_st:d_st + d_len] = 1

        # Visibility label: 1.0 when target is clearly visible, 0.0 when heavily/synthetically occluded
        vis_target = torch.where(
            (synth_occ == 1) | (occ >= 70),
            torch.zeros(T, dtype=torch.float32),
            torch.where(occ >= 40, torch.full((T,), 0.4, dtype=torch.float32), torch.ones(T, dtype=torch.float32)),
        )

        # Target Gaussian heatmaps (T, 1, 16, 16)
        hms = torch.stack([make_gaussian_heatmap(float(gt_box[i, 0]), float(gt_box[i, 1]), 16, 1.2) for i in range(T)], dim=0)

        return {
            "frame_feat": frame_feat,
            "cnn_box": cnn_box,
            "cnn_hm_logits": cnn_hm,
            "cnn_stats": cnn_stats,
            "prev_vel_crop": prev_vel,
            "prev_size_crop": prev_size,
            "dt_norm": dt_norm,
            "gt_box": gt_box,
            "gt_hm": hms,
            "vis_target": vis_target,
            "occ": occ,
        }


def compute_lstm_loss(out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    gt_box = batch["gt_box"].float()
    gt_hm = batch["gt_hm"].float()
    vis_target = batch["vis_target"].float()

    box_pred = out["box"].float()
    box_mot = out["box_mot"].float()
    hm_logits = out["heatmap_logits"].float()
    vis_logit = out["vis_logit"].float()

    iou, giou = box_iou_and_giou(box_pred, gt_box)
    loss_giou = (1.0 - giou).mean()
    loss_l1 = F.l1_loss(box_pred, gt_box)

    # Auxiliary motion-branch supervision (keeps kinematic extrapolation sharp at all times)
    _, giou_mot = box_iou_and_giou(box_mot, gt_box)
    loss_mot = (1.0 - giou_mot).mean() + 2.5 * F.l1_loss(box_mot, gt_box)

    # Visibility gate supervision
    loss_vis = F.binary_cross_entropy_with_logits(vis_logit, vis_target)

    # Gentle heatmap refinement loss
    B, T = hm_logits.shape[:2]
    loss_hm = gaussian_focal_loss(hm_logits.reshape(B * T, 1, 16, 16), gt_hm.reshape(B * T, 1, 16, 16))

    total = 2.0 * loss_giou + 5.0 * loss_l1 + 0.8 * loss_mot + 0.5 * loss_vis + 0.2 * loss_hm

    # Track IoU separately on occluded frames (vis_target == 0)
    occ_mask = vis_target < 0.1
    occ_iou = float(iou[occ_mask].mean().item()) if occ_mask.any() else float(iou.mean().item())

    return total, {
        "loss": float(total.item()),
        "giou": float(loss_giou.item()),
        "l1": float(loss_l1.item()),
        "vis": float(loss_vis.item()),
        "iou": float(iou.mean().item()),
        "occ_iou": occ_iou,
    }


@torch.no_grad()
def evaluate_lstm_sequential(
    tracker: ResNet50LSTMTracker,
    data_root: Path,
    split_file: str,
    device: torch.device,
) -> Dict:
    """
    True end-to-end online sequential evaluation on raw RGB PNGs:
    Initializes ONLY from frame 0 GT box, passes raw RGB crops through ResNet-50 -> LSTM
    frame-by-frame with persistent LSTM hidden state and smoothed kinematic velocity.
    """
    tracker.eval()
    ds_dir = data_root / "FEOT_dataset"
    seq_names = [
        l.strip()
        for l in (data_root / split_file).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    all_ious, all_errs, all_occs = [], [], []
    per_seq = []

    for name in seq_names:
        meta = load_sequence_meta(ds_dir / name)
        gt_boxes = meta["boxes"]
        occs = meta["occs"]
        frames = meta["frames"]
        ts = meta["timestamps"]

        init_box = gt_boxes[0].copy()
        img0 = Image.open(frames[0]).convert("RGB")
        W, H = img0.size
        tcx, tcy = init_box[0] + 0.5 * init_box[2], init_box[1] + 0.5 * init_box[3]
        t_side = compute_crop_size(init_box[2], init_box[3], scale_factor=2.0, min_size=64.0)
        tmpl_crop, _ = extract_square_crop(img0, tcx, tcy, t_side, 128)
        tmpl_t = pil_to_norm_tensor(tmpl_crop).unsqueeze(0).to(device)
        tmpl_enc = tracker.encode_template(tmpl_t)

        cur_box = init_box.copy()
        prev_cx = cur_box[0] + 0.5 * cur_box[2]
        prev_cy = cur_box[1] + 0.5 * cur_box[3]
        vx_40ms, vy_40ms = 0.0, 0.0
        hidden = None

        seq_ious = [1.0]
        seq_errs = [0.0]

        for fi in range(1, len(frames)):
            img = Image.open(frames[fi]).convert("RGB")
            acx = cur_box[0] + 0.5 * cur_box[2]
            acy = cur_box[1] + 0.5 * cur_box[3]
            s_side = compute_crop_size(cur_box[2], cur_box[3], scale_factor=4.0, min_size=256.0)
            srch_crop, crop_meta = extract_square_crop(img, acx, acy, s_side, 256)
            srch_t = pil_to_norm_tensor(srch_crop).unsqueeze(0).to(device)

            side_used = crop_meta[2]
            dt_cur = max(5000.0, ts[fi] - ts[fi - 1])
            dt_norm = float(np.clip(dt_cur / 40000.0, 0.25, 3.0))

            vel_t = torch.tensor([[vx_40ms / side_used, vy_40ms / side_used]], dtype=torch.float32, device=device)
            size_t = torch.tensor([[cur_box[2] / side_used, cur_box[3] / side_used]], dtype=torch.float32, device=device)
            dt_t = torch.tensor([[dt_norm]], dtype=torch.float32, device=device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = tracker.step(tmpl_enc, srch_t, vel_t, size_t, dt_t, hidden=hidden)

            hidden = out["hidden"]
            pred_norm = out["box"][0].float().cpu().numpy()
            vis_p = float(out["vis_prob"][0].item())
            pred_box = crop_norm_to_frame_box(pred_norm, crop_meta, img_w=W, img_h=H)

            # Smooth box size slightly for stability
            if vis_p >= 0.45:
                pred_box[2] = 0.75 * cur_box[2] + 0.25 * pred_box[2]
                pred_box[3] = 0.75 * cur_box[3] + 0.25 * pred_box[3]
            else:
                # Preserve target size during heavy occlusion
                pred_box[2] = cur_box[2]
                pred_box[3] = cur_box[3]

            new_cx = pred_box[0] + 0.5 * pred_box[2]
            new_cy = pred_box[1] + 0.5 * pred_box[3]

            # Update EMA velocity when visible; gently damp velocity when occluded
            step_vx = (new_cx - prev_cx) * (40000.0 / dt_cur)
            step_vy = (new_cy - prev_cy) * (40000.0 / dt_cur)
            # Clamp extreme outlier jumps
            step_vx = float(np.clip(step_vx, -80.0, 80.0))
            step_vy = float(np.clip(step_vy, -80.0, 80.0))
            if vis_p >= 0.55:
                vx_40ms = 0.55 * vx_40ms + 0.45 * step_vx
                vy_40ms = 0.55 * vy_40ms + 0.45 * step_vy
            else:
                vx_40ms *= 0.94
                vy_40ms *= 0.94

            cur_box = pred_box
            prev_cx, prev_cy = new_cx, new_cy

            seq_ious.append(compute_iou_xywh(cur_box, gt_boxes[fi]))
            seq_errs.append(compute_center_err(cur_box, gt_boxes[fi]))

        all_ious.extend(seq_ious)
        all_errs.extend(seq_errs)
        all_occs.extend(occs.tolist())
        per_seq.append({
            "seq": name,
            "n_frames": len(frames),
            "mean_iou": float(np.mean(seq_ious)),
            "iou_50": float(np.mean(np.array(seq_ious) >= 0.5)),
            "prec_20px": float(np.mean(np.array(seq_errs) <= 20.0)),
        })

    ious = np.array(all_ious)
    errs = np.array(all_errs)
    occs = np.array(all_occs)

    def bucket_stats(mask: np.ndarray) -> Dict[str, float]:
        if mask.sum() == 0:
            return {"count": 0, "mean_iou": 0.0, "iou_50": 0.0, "prec_20px": 0.0}
        return {
            "count": int(mask.sum()),
            "mean_iou": float(ious[mask].mean()),
            "iou_50": float((ious[mask] >= 0.5).mean()),
            "prec_20px": float((errs[mask] <= 20.0).mean()),
        }

    return {
        "split": split_file,
        "overall": bucket_stats(np.ones_like(occs, dtype=bool)),
        "by_occlusion": {
            "occ_0": bucket_stats(occs == 0),
            "occ_10_40": bucket_stats((occs >= 10) & (occs <= 40)),
            "occ_50_90": bucket_stats((occs >= 50) & (occs <= 90)),
            "occ_100": bucket_stats(occs == 100),
        },
        "per_sequence": per_seq,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Attach & Train Temporal LSTM on ResNet-50 CNN")
    parser.add_argument("--data_root", type=str, default=str(_PROJ_ROOT / "data" / "FEOT_100"))
    parser.add_argument("--out_dir", type=str, default=str(_PROJ_ROOT))
    parser.add_argument("--cnn_ckpt", type=str, default=str(_PROJ_ROOT / "checkpoints" / "resnet50_cnn_best.pt"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--clip_len", type=int, default=24)
    parser.add_argument("--clips_per_epoch", type=int, default=2560)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    res_dir = out_dir / "outputs"
    feat_dir = out_dir / ".cache" / "stage2_feats"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Load trained Stage-1 ResNet-50 CNN
    cnn = ResNet50TrackerCNN(pretrained=False, feat_dim=256, freeze_early=True).to(device)
    cnn_ckpt = torch.load(args.cnn_ckpt, map_location=device, weights_only=False)
    cnn.load_state_dict(cnn_ckpt["model_state_dict"])
    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad = False
    print(f"Loaded Stage-1 ResNet-50 CNN from {args.cnn_ckpt} (epoch {cnn_ckpt['epoch']}, val IoU {cnn_ckpt['val_metrics']['iou']:.4f})", flush=True)

    train_names = [l.strip() for l in (data_root / "train_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_names = [l.strip() for l in (data_root / "val_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    # Step 1: Extract ResNet-50 features (clean + real-image synthetic occlusion pass for train, clean for val)
    print("\n[1/3] Extracting ResNet-50 CNN features for temporal clips...", flush=True)
    tr_clean = extract_sequence_features(cnn, data_root, train_names, device, feat_dir / "train_clean.pt", synth_occlusion=False, seed=args.seed)
    tr_synth = extract_sequence_features(cnn, data_root, train_names, device, feat_dir / "train_synth_occ.pt", synth_occlusion=True, seed=args.seed + 1)
    val_clean = extract_sequence_features(cnn, data_root, val_names, device, feat_dir / "val_clean.pt", synth_occlusion=False, seed=args.seed)
    val_synth = extract_sequence_features(cnn, data_root, val_names, device, feat_dir / "val_synth_occ.pt", synth_occlusion=True, seed=args.seed + 2)

    train_ds = FEOTFeatureClipDataset(tr_clean + tr_synth, clip_len=args.clip_len, clips_per_epoch=args.clips_per_epoch, augment=True)
    val_ds = FEOTFeatureClipDataset(val_clean + val_synth, clip_len=args.clip_len, clips_per_epoch=0, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Step 2: Train TemporalLSTMHead
    print(f"\n[2/3] Training TemporalLSTMHead ({args.epochs} epochs, {len(train_ds)} clips/epoch, {len(val_ds)} val clips)...", flush=True)
    lstm_head = TemporalLSTMHead(hidden_size=256, num_layers=2, dropout=0.3).to(device)
    n_lstm = sum(p.numel() for p in lstm_head.parameters())
    print(f"TemporalLSTMHead params: {n_lstm/1e6:.2f}M (all trainable, ResNet-50 CNN frozen)", flush=True)

    optimizer = torch.optim.AdamW(lstm_head.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_score = -1.0
    best_ckpt_path = ckpt_dir / "resnet50_lstm_best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lstm_head.train()
        tr_sums = {"loss": 0.0, "giou": 0.0, "l1": 0.0, "vis": 0.0, "iou": 0.0, "occ_iou": 0.0}
        n_tr = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = lstm_head(
                batch["frame_feat"],
                batch["cnn_box"],
                batch["cnn_hm_logits"],
                batch["cnn_stats"],
                batch["prev_vel_crop"],
                batch["prev_size_crop"],
                batch["dt_norm"],
            )
            loss, met = compute_lstm_loss(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_head.parameters(), max_norm=3.0)
            optimizer.step()
            for k in tr_sums:
                tr_sums[k] += met[k]
            n_tr += 1

        scheduler.step()
        tr_avg = {k: v / max(1, n_tr) for k, v in tr_sums.items()}

        lstm_head.eval()
        va_sums = {"loss": 0.0, "giou": 0.0, "l1": 0.0, "vis": 0.0, "iou": 0.0, "occ_iou": 0.0}
        n_va = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = lstm_head(
                    batch["frame_feat"],
                    batch["cnn_box"],
                    batch["cnn_hm_logits"],
                    batch["cnn_stats"],
                    batch["prev_vel_crop"],
                    batch["prev_size_crop"],
                    batch["dt_norm"],
                )
                _, met = compute_lstm_loss(out, batch)
                for k in va_sums:
                    va_sums[k] += met[k]
                n_va += 1
        va_avg = {k: v / max(1, n_va) for k, v in va_sums.items()}

        # Selection score balances overall Val IoU + occluded Val IoU
        val_score = 0.5 * va_avg["iou"] + 0.5 * va_avg["occ_iou"]
        tag = ""
        if val_score > best_val_score:
            best_val_score = val_score
            tracker = ResNet50LSTMTracker(cnn, lstm_head, freeze_cnn=True)
            torch.save(
                {
                    "epoch": epoch,
                    "cnn_state_dict": cnn.state_dict(),
                    "lstm_state_dict": lstm_head.state_dict(),
                    "full_state_dict": tracker.state_dict(),
                    "val_metrics": va_avg,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            tag = " *BEST*"

        history.append({"epoch": epoch, "train": tr_avg, "val": va_avg})
        if epoch == 1 or epoch % 5 == 0 or tag:
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} ({time.time()-t0:4.1f}s) | "
                f"Tr loss={tr_avg['loss']:.3f} IoU={tr_avg['iou']:.3f} OccIoU={tr_avg['occ_iou']:.3f} | "
                f"Va loss={va_avg['loss']:.3f} IoU={va_avg['iou']:.3f} OccIoU={va_avg['occ_iou']:.3f}{tag}",
                flush=True,
            )

    # Step 3: Load best unified ResNet50LSTMTracker and run full raw-RGB online sequential evaluation
    print("\n[3/3] Running end-to-end raw-RGB online sequential tracking evaluation (ResNet-50 + LSTM)...", flush=True)
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    lstm_head.load_state_dict(best_ckpt["lstm_state_dict"])
    tracker = ResNet50LSTMTracker(cnn, lstm_head, freeze_cnn=True).to(device)
    tracker.eval()

    val_track = evaluate_lstm_sequential(tracker, data_root, "val_list.txt", device)
    test_track = evaluate_lstm_sequential(tracker, data_root, "test_list.txt", device)

    summary = {
        "best_epoch": best_ckpt["epoch"],
        "best_val_clip_metrics": best_ckpt["val_metrics"],
        "history": history,
        "val_tracking": val_track,
        "test_tracking": test_track,
    }
    (res_dir / "stage2_lstm_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Side-by-side comparison vs Stage 1
    s1_path = res_dir / "stage1_cnn_eval.json"
    s1 = json.loads(s1_path.read_text(encoding="utf-8")) if s1_path.exists() else None

    for label, s2_res, s1_key in [("VAL (10 seqs)", val_track, "val_tracking"), ("TEST (20 seqs)", test_track, "test_tracking")]:
        ov = s2_res["overall"]
        s1_ov = s1[s1_key]["overall"] if s1 else None
        delta_ov = f" ({ov['mean_iou'] - s1_ov['mean_iou']:+.4f} vs CNN)" if s1_ov else ""
        print(f"\n=== Stage-2 ResNet-50 + LSTM Sequential Tracking — {label} ===", flush=True)
        print(f"  Overall   : Mean IoU={ov['mean_iou']:.4f}{delta_ov} | IoU@0.5={ov['iou_50']:.4f} | Prec@20px={ov['prec_20px']:.4f} (n={ov['count']})", flush=True)
        for bk, st in s2_res["by_occlusion"].items():
            s1_bk = s1[s1_key]["by_occlusion"][bk] if s1 else None
            d_str = f" ({st['mean_iou'] - s1_bk['mean_iou']:+.4f})" if s1_bk else ""
            print(f"  {bk:10s}: Mean IoU={st['mean_iou']:.4f}{d_str} | IoU@0.5={st['iou_50']:.4f} | Prec@20px={st['prec_20px']:.4f} (n={st['count']})", flush=True)

    # Save curves
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".cache" / "matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(ep, [h["train"]["loss"] for h in history], label="Train Loss")
        ax1.plot(ep, [h["val"]["loss"] for h in history], label="Val Loss")
        ax1.set_title("Stage-2 Temporal LSTM Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(ep, [h["train"]["iou"] for h in history], label="Train IoU")
        ax2.plot(ep, [h["val"]["iou"] for h in history], label="Val IoU")
        ax2.plot(ep, [h["val"]["occ_iou"] for h in history], "--", label="Val Occluded IoU")
        ax2.set_title("Stage-2 Temporal LSTM IoU")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(res_dir / "stage2_lstm_curves.png", dpi=150)
        plt.close(fig)
        print(f"Saved curves to {res_dir / 'stage2_lstm_curves.png'}", flush=True)
    except Exception as e:
        print(f"Skipping curve plot: {e}", flush=True)


if __name__ == "__main__":
    main()
