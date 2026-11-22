import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJ_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_PROJ_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(_PROJ_ROOT / ".cache" / "torch"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
import torch

from rex_xai.explanation.rex import calculate_responsibility, predict_target
from rex_xai.input.config import CausalArgs
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import Prediction
from rex_xai.utils._utils import ResponsibilityStyle

from feot_dataset import (
    compute_crop_size,
    crop_norm_to_frame_box,
    extract_square_crop,
    frame_box_to_crop_norm,
    load_sequence_meta,
    pil_to_norm_tensor,
)
from losses import box_iou_and_giou
from models import ResNet50LSTMTracker, ResNet50TrackerCNN, TemporalLSTMHead
from train_cnn import compute_iou_xywh


def extract_event_crop_binary(
    evt_img: Image.Image,
    crop_meta: Tuple[float, float, float],
    out_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop FEOT event PNG (white [255,255,255] background, red/blue active events) at exact
    `crop_meta = (x1, y1, side)`.
    Returns:
      - evt_rgb_crop: (out_size, out_size, 3) uint8 resized RGB crop for visualization
      - evt_bin: (out_size, out_size) float32 binary mask in {0, 1} using max-preserving resize
                 so sparse 1-pixel events are never lost.
    """
    W, H = evt_img.size
    ix1, iy1, side = int(round(crop_meta[0])), int(round(crop_meta[1])), int(round(crop_meta[2]))
    ix2, iy2 = ix1 + side, iy1 + side

    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    sx1, sy1 = max(0, ix1), max(0, iy1)
    sx2, sy2 = min(W, ix2), min(H, iy2)
    if sx2 > sx1 and sy2 > sy1:
        patch = evt_img.crop((sx1, sy1, sx2, sy2))
        canvas.paste(patch, (sx1 - ix1, sy1 - iy1))

    arr = np.asarray(canvas)
    # Active events are red [255,0,0] or blue [0,0,255] on pure white [255,255,255]
    raw_bin = np.any(arr < 250, axis=-1).astype(np.float32)

    # Max-preserving resize to (out_size, out_size) via PyTorch adaptive_max_pool2d / nearest upsample
    t = torch.from_numpy(raw_bin).unsqueeze(0).unsqueeze(0)
    if side >= out_size:
        evt_bin = torch.nn.functional.adaptive_max_pool2d(t, (out_size, out_size))[0, 0].numpy()
    else:
        evt_bin = torch.nn.functional.interpolate(t, size=(out_size, out_size), mode="nearest")[0, 0].numpy()

    evt_rgb_crop = np.asarray(canvas.resize((out_size, out_size), Image.NEAREST))
    return evt_rgb_crop, (evt_bin > 0.5).astype(np.float32)


def compute_roc_auc(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    """Fast exact ROC-AUC using rank statistic (Mann-Whitney U) with tie handling."""
    y = binary_labels.ravel() > 0.5
    s = scores.ravel().astype(np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(s)
    pos_rank_sum = ranks[y].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_alignment_metrics(
    rex_map: np.ndarray,
    evt_bin: np.ndarray,
    gt_box_norm: np.ndarray,
    pred_box_norm: np.ndarray,
) -> Dict[str, float]:
    """
    Compute spatial alignment metrics between continuous ReX responsibility map (H, W)
    and pixel-aligned FEOT binary event mask (H, W).
    """
    H, W = rex_map.shape
    rm = np.maximum(rex_map.astype(np.float64), 0.0)
    r_max = rm.max()
    if r_max > 1e-12:
        rm_norm = rm / r_max
    else:
        rm_norm = np.zeros_like(rm)

    n_evt = int(evt_bin.sum())

    # 1. NSS (Normalized Scanpath Saliency) at exact active event pixels
    r_std = rm_norm.std()
    if r_std > 1e-12 and n_evt > 0:
        r_z = (rm_norm - rm_norm.mean()) / r_std
        nss = float(r_z[evt_bin > 0.5].mean())
    else:
        nss = 0.0

    # 2. Threshold-free ROC-AUC (ReX score predicting active event pixels)
    roc_auc = compute_roc_auc(rm_norm, evt_bin) if (r_std > 1e-12 and n_evt > 0) else float("nan")

    # 3. Continuous spatial correlation (CC) & histogram intersection (SIM) vs Gaussian-smoothed event density
    evt_smooth = gaussian_filter(evt_bin.astype(np.float64), sigma=5.0)
    if r_std > 1e-12 and evt_smooth.std() > 1e-12:
        cc = float(np.corrcoef(rm_norm.ravel(), evt_smooth.ravel())[0, 1])
    else:
        cc = 0.0

    r_sum = rm_norm.sum()
    e_sum = evt_smooth.sum()
    if r_sum > 1e-12 and e_sum > 1e-12:
        p_r = rm_norm / r_sum
        p_e = evt_smooth / e_sum
        sim = float(np.minimum(p_r, p_e).sum())
    else:
        sim = 0.0

    # 4. Top-20% ReX mask vs dilated event region IoU & Dice (patch-scale spatial overlap)
    if r_max > 1e-12 and n_evt > 0:
        thresh = np.percentile(rm_norm, 80.0)
        rex_top = rm_norm >= max(thresh, 1e-6)
        evt_dil = binary_dilation(evt_bin > 0.5, iterations=4)
        inter = float((rex_top & evt_dil).sum())
        union = float((rex_top | evt_dil).sum())
        mask_iou = inter / max(1.0, union)
        mask_dice = 2.0 * inter / max(1.0, float(rex_top.sum() + evt_dil.sum()))
        # Recall of raw event pixels covered by top-20% ReX area
        evt_recall_top20 = float((rex_top & (evt_bin > 0.5)).sum()) / max(1.0, float(n_evt))
    else:
        mask_iou, mask_dice, evt_recall_top20 = 0.0, 0.0, 0.0

    # 5. Energy-inside-GT-box (Target Focus) for BOTH ReX and Event camera
    def box_mask(b_norm: np.ndarray, expand: float = 1.2) -> np.ndarray:
        cx, cy, bw, bh = b_norm
        bw *= expand
        bh *= expand
        x1 = max(0, int(round((cx - 0.5 * bw) * W)))
        y1 = max(0, int(round((cy - 0.5 * bh) * H)))
        x2 = min(W, int(round((cx + 0.5 * bw) * W)))
        y2 = min(H, int(round((cy + 0.5 * bh) * H)))
        m = np.zeros((H, W), dtype=bool)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
        return m

    gt_m = box_mask(gt_box_norm, expand=1.25)
    pred_m = box_mask(pred_box_norm, expand=1.25)
    rex_in_gt = float(rm_norm[gt_m].sum() / r_sum) if r_sum > 1e-12 else 0.0
    rex_in_pred = float(rm_norm[pred_m].sum() / r_sum) if r_sum > 1e-12 else 0.0
    evt_in_gt = float(evt_bin[gt_m].sum() / n_evt) if n_evt > 0 else 0.0
    gt_area_frac = float(gt_m.mean())

    # Event pixels inside GT box specifically: does ReX rank target-event pixels above rest of image?
    evt_gt_only = (evt_bin > 0.5) & gt_m
    if r_std > 1e-12 and evt_gt_only.sum() > 0:
        nss_target_events = float(((rm_norm - rm_norm.mean()) / r_std)[evt_gt_only].mean())
        auc_target_events = compute_roc_auc(rm_norm, evt_gt_only.astype(np.float32))
    else:
        nss_target_events = float("nan")
        auc_target_events = float("nan")

    return {
        "n_event_pixels": n_evt,
        "nss": nss,
        "roc_auc": roc_auc,
        "cc": cc,
        "sim": sim,
        "top20_iou": mask_iou,
        "top20_dice": mask_dice,
        "evt_recall_top20": evt_recall_top20,
        "rex_in_gt_frac": rex_in_gt,
        "rex_in_pred_frac": rex_in_pred,
        "evt_in_gt_frac": evt_in_gt,
        "gt_area_frac": gt_area_frac,
        "nss_target_events": nss_target_events,
        "auc_target_events": auc_target_events,
    }


def make_rex_prediction_fn(
    tracker: ResNet50LSTMTracker,
    tmpl_enc: Tuple[torch.Tensor, torch.Tensor],
    vel_t: torch.Tensor,
    size_t: torch.Tensor,
    dt_t: torch.Tensor,
    cached_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
    orig_box: torch.Tensor,
    orig_vis_prob: float,
    orig_hm_peak: float,
    device: torch.device,
    iou_thresh: float = 0.60,
):
    """
    Builds a batched GPU prediction_function compatible with `rex_xai`.
    Class 1 = mutant preserves both spatial localization (IoU >= iou_thresh) AND visual
    confirmation from ResNet-50 + LSTM (preventing empty masked frames from trivially passing
    via motion-only fallback).
    """
    orig_box_dev = orig_box.to(device)

    def prediction_function(mutants: torch.Tensor, target=None, raw: bool = False) -> List[Prediction]:
        if mutants.ndim == 3:
            mutants = mutants.unsqueeze(0)
        mutants = mutants.to(device, non_blocking=True)
        B = mutants.shape[0]

        k_exp = tmpl_enc[0].expand(B, -1, -1, -1)
        g_exp = tmpl_enc[1].expand(B, -1, -1, -1)
        vel_exp = vel_t.expand(B, -1)
        size_exp = size_t.expand(B, -1)
        dt_exp = dt_t.expand(B, -1)
        if cached_hidden is not None:
            h_exp = (
                cached_hidden[0].expand(-1, B, -1).contiguous(),
                cached_hidden[1].expand(-1, B, -1).contiguous(),
            )
        else:
            h_exp = None

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = tracker.step((k_exp, g_exp), mutants, vel_exp, size_exp, dt_exp, hidden=h_exp)

        pred_boxes = out["box"].float()
        vis_probs = out["vis_prob"].float()
        hm_peaks = torch.sigmoid(out["heatmap_logits"].float().view(B, -1)).max(dim=-1).values

        ious, _ = box_iou_and_giou(pred_boxes, orig_box_dev.expand(B, -1))

        # Visual confirmation ratio relative to unperturbed frame
        if orig_vis_prob >= 0.30:
            vis_ok = vis_probs >= (0.55 * orig_vis_prob)
            vis_ratio = (vis_probs / max(orig_vis_prob, 1e-4)).clamp(0.0, 1.0)
        else:
            vis_ok = hm_peaks >= (0.60 * orig_hm_peak)
            vis_ratio = (hm_peaks / max(orig_hm_peak, 1e-4)).clamp(0.0, 1.0)

        passed = (ious >= iou_thresh) & vis_ok
        score_1 = (0.5 * ious.clamp(0.0, 1.0) + 0.5 * vis_ratio).clamp(0.0, 1.0)

        if raw:
            return torch.stack([1.0 - score_1, score_1], dim=-1)

        preds = []
        for i in range(B):
            cls_id = 1 if bool(passed[i].item()) else 0
            s1 = float(score_1[i].item())
            conf = s1 if cls_id == 1 else float(1.0 - s1)
            p = Prediction(pred=cls_id, conf=conf)
            if target is not None:
                p.target = target.classification
                p.target_confidence = s1 if target.classification == 1 else (1.0 - s1)
            preds.append(p)
        return preds

    return prediction_function


def run_rex_on_frame(
    tracker: ResNet50LSTMTracker,
    srch_crop_pil: Image.Image,
    srch_tensor: torch.Tensor,
    tmpl_enc: Tuple[torch.Tensor, torch.Tensor],
    vel_t: torch.Tensor,
    size_t: torch.Tensor,
    dt_t: torch.Tensor,
    cached_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
    orig_box: torch.Tensor,
    orig_vis_prob: float,
    orig_hm_peak: float,
    device: torch.device,
    iters: int = 16,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """Run rex_xai causal responsibility mapping on a single 256x256 RGB search crop."""
    args = CausalArgs()
    args.iters = iters
    args.batch_size = batch_size
    args.tree_depth = 8
    args.min_box_size = 12
    args.mask_value = 0.0  # ImageNet mean in normalized tensor space
    args.responsibility_style = ResponsibilityStyle.Additive
    args.progress_bar = False
    args.verbosity = 0
    args.seed = seed

    data = Data(srch_crop_pil, (1, 3, 256, 256), device=device, mode="RGB")
    data.data = srch_tensor.to(device)  # (1, 3, 256, 256)
    data.transposed = True
    data.set_mask_value(0.0)

    pred_fn = make_rex_prediction_fn(
        tracker=tracker,
        tmpl_enc=tmpl_enc,
        vel_t=vel_t,
        size_t=size_t,
        dt_t=dt_t,
        cached_hidden=cached_hidden,
        orig_box=orig_box,
        orig_vis_prob=orig_vis_prob,
        orig_hm_peak=orig_hm_peak,
        device=device,
    )

    data.target = predict_target(data, args, pred_fn)
    maps, stats = calculate_responsibility(data, args, pred_fn)
    rmap = maps.get(data.target.classification)
    if rmap is None:
        rmap = np.zeros((256, 256), dtype=np.float32)
    else:
        rmap = np.asarray(rmap, dtype=np.float32)
    return rmap, stats


def save_comparison_panel(
    out_path: Path,
    seq_name: str,
    frame_idx: int,
    occ: int,
    track_iou: float,
    vis_prob: float,
    srch_pil: Image.Image,
    rex_map: np.ndarray,
    evt_rgb: np.ndarray,
    evt_bin: np.ndarray,
    gt_box_norm: np.ndarray,
    pred_box_norm: np.ndarray,
    metrics: Dict[str, float],
):
    H, W = rex_map.shape
    rm = np.maximum(rex_map, 0.0)
    rm = rm / rm.max() if rm.max() > 1e-12 else rm

    def draw_box(ax, b_norm, color, label, ls="-"):
        cx, cy, bw, bh = b_norm
        rect = patches.Rectangle(
            ((cx - 0.5 * bw) * W, (cy - 0.5 * bh) * H),
            bw * W,
            bh * H,
            linewidth=2.0,
            edgecolor=color,
            facecolor="none",
            linestyle=ls,
            label=label,
        )
        ax.add_patch(rect)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    fig.suptitle(
        f"{seq_name} | frame {frame_idx} | occ={occ}% | Track IoU={track_iou:.2f} | LSTM vis={vis_prob:.2f} | "
        f"NSS={metrics['nss']:.2f} | AUC={metrics['roc_auc']:.2f} | CC={metrics['cc']:.2f} | "
        f"ReX-in-GT={100*metrics['rex_in_gt_frac']:.0f}% vs Evt-in-GT={100*metrics['evt_in_gt_frac']:.0f}%",
        fontsize=10.5,
        fontweight="bold",
    )

    # Panel 1: RGB search crop + GT (green) & Pred (red) boxes
    axes[0].imshow(srch_pil)
    draw_box(axes[0], gt_box_norm, "#00ff44", "GT", ls="--")
    draw_box(axes[0], pred_box_norm, "#ff3333", "Pred", ls="-")
    axes[0].set_title("1. RGB Crop (GT=green, Pred=red)")
    axes[0].axis("off")

    # Panel 2: ReX Causal Responsibility Map overlaid on RGB
    axes[1].imshow(srch_pil)
    axes[1].imshow(rm, cmap="inferno", alpha=0.60, vmin=0.0, vmax=1.0)
    draw_box(axes[1], gt_box_norm, "#00ff44", "GT", ls="--")
    axes[1].set_title("2. ReX Responsibility Map")
    axes[1].axis("off")

    # Panel 3: Aligned FEOT Event Picture
    axes[2].imshow(evt_rgb)
    draw_box(axes[2], gt_box_norm, "#00aa33", "GT", ls="--")
    axes[2].set_title(f"3. FEOT Event Picture ({metrics['n_event_pixels']} px)")
    axes[2].axis("off")

    # Panel 4: ReX map + Event pixels overlaid directly
    axes[3].imshow(rm, cmap="magma", vmin=0.0, vmax=1.0)
    # Overlay active event pixels in bright cyan
    evt_overlay = np.zeros((H, W, 4), dtype=np.float32)
    evt_overlay[evt_bin > 0.5] = [0.0, 1.0, 1.0, 0.85]
    axes[3].imshow(evt_overlay)
    draw_box(axes[3], gt_box_norm, "#00ff44", "GT", ls="--")
    axes[3].set_title("4. ReX (magma) + Events (cyan)")
    axes[3].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def collect_test_frame_snapshots(
    tracker: ResNet50LSTMTracker,
    data_root: Path,
    device: torch.device,
    max_per_bucket: int = 12,
) -> List[Dict]:
    """
    Roll out the Stage-2 ResNet-50 + LSTM tracker online across all 20 test sequences,
    collecting balanced frame snapshots across all 4 occlusion regimes (occ=0, 10-40, 50-90, 100)
    along with each frame's exact cached LSTM hidden state (h_{t-1}, c_{t-1}).
    """
    tracker.eval()
    ds_dir = data_root / "FEOT_dataset"
    seq_names = [
        l.strip()
        for l in (data_root / "test_list.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    buckets: Dict[str, List[Dict]] = {"occ_0": [], "occ_10_40": [], "occ_50_90": [], "occ_100": []}

    for name in seq_names:
        meta = load_sequence_meta(ds_dir / name)
        gt_boxes = meta["boxes"]
        occs = meta["occs"]
        frames = meta["frames"]
        events = meta["events"]
        ts = meta["timestamps"]

        init_box = gt_boxes[0].copy()
        img0 = Image.open(frames[0]).convert("RGB")
        W, H = img0.size
        tcx, tcy = init_box[0] + 0.5 * init_box[2], init_box[1] + 0.5 * init_box[3]
        t_side = compute_crop_size(init_box[2], init_box[3], scale_factor=2.0, min_size=64.0)
        tmpl_crop, _ = extract_square_crop(img0, tcx, tcy, t_side, 128)
        tmpl_t = pil_to_norm_tensor(tmpl_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            tmpl_enc = tracker.encode_template(tmpl_t)
        tmpl_enc_cpu = (tmpl_enc[0].cpu(), tmpl_enc[1].cpu())

        cur_box = init_box.copy()
        prev_cx, prev_cy = cur_box[0] + 0.5 * cur_box[2], cur_box[1] + 0.5 * cur_box[3]
        vx_40ms, vy_40ms = 0.0, 0.0
        hidden = None

        # Candidate frames spaced out so we sample diverse moments
        last_taken = {"occ_0": -99, "occ_10_40": -99, "occ_50_90": -99, "occ_100": -99}

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

            prev_hidden_cpu = (hidden[0].detach().cpu(), hidden[1].detach().cpu()) if hidden is not None else None

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = tracker.step(tmpl_enc, srch_t, vel_t, size_t, dt_t, hidden=hidden)

            hidden = out["hidden"]
            pred_norm = out["box"][0].float().cpu().numpy()
            vis_p = float(out["vis_prob"][0].item())
            hm_peak = float(torch.sigmoid(out["heatmap_logits"][0].float()).max().item())
            pred_box = crop_norm_to_frame_box(pred_norm, crop_meta, img_w=W, img_h=H)

            if vis_p >= 0.45:
                pred_box[2] = 0.75 * cur_box[2] + 0.25 * pred_box[2]
                pred_box[3] = 0.75 * cur_box[3] + 0.25 * pred_box[3]
            else:
                pred_box[2] = cur_box[2]
                pred_box[3] = cur_box[3]

            track_iou = compute_iou_xywh(pred_box, gt_boxes[fi])
            occ = int(occs[fi])
            bname = (
                "occ_0" if occ == 0
                else "occ_10_40" if occ <= 40
                else "occ_50_90" if occ <= 90
                else "occ_100"
            )

            # Sample frames where target is inside the search crop (gt center in [0.1, 0.9]) and spaced >= 15 frames
            gt_norm = frame_box_to_crop_norm(gt_boxes[fi], crop_meta)
            in_crop = (0.10 <= gt_norm[0] <= 0.90) and (0.10 <= gt_norm[1] <= 0.90)
            min_iou_req = 0.45 if bname == "occ_0" else (0.25 if bname == "occ_10_40" else 0.0)

            if fi >= 5 and in_crop and track_iou >= min_iou_req and (fi - last_taken[bname]) >= 18:
                evt_img = Image.open(events[fi]).convert("RGB")
                evt_rgb, evt_bin = extract_event_crop_binary(evt_img, crop_meta, 256)
                if int(evt_bin.sum()) >= 30:
                    buckets[bname].append({
                        "seq": name,
                        "frame_idx": fi,
                        "occ": occ,
                        "bucket": bname,
                        "track_iou": track_iou,
                        "vis_prob": vis_p,
                        "hm_peak": hm_peak,
                        "srch_pil": srch_crop,
                        "srch_t": srch_t.cpu(),
                        "tmpl_enc": tmpl_enc_cpu,
                        "vel_t": vel_t.cpu(),
                        "size_t": size_t.cpu(),
                        "dt_t": dt_t.cpu(),
                        "prev_hidden": prev_hidden_cpu,
                        "orig_box": out["box"][0].float().cpu(),
                        "gt_box_norm": gt_norm,
                        "pred_box_norm": pred_norm,
                        "crop_meta": crop_meta,
                        "evt_rgb": evt_rgb,
                        "evt_bin": evt_bin,
                    })
                    last_taken[bname] = fi

            new_cx = pred_box[0] + 0.5 * pred_box[2]
            new_cy = pred_box[1] + 0.5 * pred_box[3]
            step_vx = float(np.clip((new_cx - prev_cx) * (40000.0 / dt_cur), -80.0, 80.0))
            step_vy = float(np.clip((new_cy - prev_cy) * (40000.0 / dt_cur), -80.0, 80.0))
            if vis_p >= 0.55:
                vx_40ms = 0.55 * vx_40ms + 0.45 * step_vx
                vy_40ms = 0.55 * vy_40ms + 0.45 * step_vy
            else:
                vx_40ms *= 0.94
                vy_40ms *= 0.94
            cur_box = pred_box
            prev_cx, prev_cy = new_cx, new_cy

    # Spread selection across distinct sequences in each bucket
    selected = []
    for bname in ["occ_0", "occ_10_40", "occ_50_90", "occ_100"]:
        cands = buckets[bname]
        # Round-robin across sequences so no single sequence dominates
        by_seq: Dict[str, List[Dict]] = {}
        for c in cands:
            by_seq.setdefault(c["seq"], []).append(c)
        chosen = []
        seq_keys = sorted(by_seq.keys())
        round_idx = 0
        while len(chosen) < max_per_bucket and seq_keys:
            added = False
            for sk in list(seq_keys):
                if round_idx < len(by_seq[sk]) and len(chosen) < max_per_bucket:
                    chosen.append(by_seq[sk][round_idx])
                    added = True
            if not added:
                break
            round_idx += 1
        print(f"Bucket {bname:10s}: {len(cands):3d} candidates -> selected {len(chosen)} frames across {len(set(x['seq'] for x in chosen))} seqs", flush=True)
        selected.extend(chosen)

    return selected


def main():
    parser = argparse.ArgumentParser(description="Stage 3: ReX Causal Responsibility Maps vs. FEOT Event Pictures")
    parser.add_argument("--data_root", type=str, default=str(_PROJ_ROOT / "data" / "FEOT_100"))
    parser.add_argument("--out_dir", type=str, default=str(_PROJ_ROOT / "outputs" / "rex_eval"))
    parser.add_argument("--ckpt", type=str, default=str(_PROJ_ROOT / "checkpoints" / "resnet50_lstm_best.pt"))
    parser.add_argument("--max_per_bucket", type=int, default=12)
    parser.add_argument("--rex_iters", type=int, default=16)
    parser.add_argument("--rex_batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    # Load unified ResNet-50 + LSTM model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cnn = ResNet50TrackerCNN(pretrained=False, feat_dim=256, freeze_early=True)
    lstm_head = TemporalLSTMHead(hidden_size=256, num_layers=2, dropout=0.3)
    tracker = ResNet50LSTMTracker(cnn, lstm_head, freeze_cnn=True).to(device)
    tracker.load_state_dict(ckpt["full_state_dict"])
    tracker.eval()
    print(f"Loaded ResNet50LSTMTracker from {args.ckpt} (epoch {ckpt['epoch']})", flush=True)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Rolling out ResNet-50 + LSTM on test sequences & caching hidden states...", flush=True)
    snapshots = collect_test_frame_snapshots(tracker, data_root, device, max_per_bucket=args.max_per_bucket)
    print(f"Total selected frames for ReX explanation: {len(snapshots)}", flush=True)

    print(f"\n[2/3] Running ReX (iters={args.rex_iters}, batch={args.rex_batch}) & comparing against event frames...", flush=True)
    results = []
    t0 = time.time()
    panels_saved_per_bucket = {"occ_0": 0, "occ_10_40": 0, "occ_50_90": 0, "occ_100": 0}

    for idx, snap in enumerate(snapshots, 1):
        tf0 = time.time()
        tmpl_enc = (snap["tmpl_enc"][0].to(device), snap["tmpl_enc"][1].to(device))
        prev_h = (
            (snap["prev_hidden"][0].to(device), snap["prev_hidden"][1].to(device))
            if snap["prev_hidden"] is not None
            else None
        )

        rex_map, rex_stats = run_rex_on_frame(
            tracker=tracker,
            srch_crop_pil=snap["srch_pil"],
            srch_tensor=snap["srch_t"],
            tmpl_enc=tmpl_enc,
            vel_t=snap["vel_t"].to(device),
            size_t=snap["size_t"].to(device),
            dt_t=snap["dt_t"].to(device),
            cached_hidden=prev_h,
            orig_box=snap["orig_box"],
            orig_vis_prob=snap["vis_prob"],
            orig_hm_peak=snap["hm_peak"],
            device=device,
            iters=args.rex_iters,
            batch_size=args.rex_batch,
            seed=args.seed + idx,
        )

        metrics = compute_alignment_metrics(
            rex_map=rex_map,
            evt_bin=snap["evt_bin"],
            gt_box_norm=snap["gt_box_norm"],
            pred_box_norm=snap["pred_box_norm"],
        )

        rec = {
            "seq": snap["seq"],
            "frame_idx": snap["frame_idx"],
            "occ": snap["occ"],
            "bucket": snap["bucket"],
            "track_iou": round(snap["track_iou"], 4),
            "vis_prob": round(snap["vis_prob"], 4),
            "rex_passing": int(rex_stats["total_passing"]),
            "rex_failing": int(rex_stats["total_failing"]),
            "sec": round(time.time() - tf0, 2),
            "metrics": metrics,
        }
        results.append(rec)

        # Save up to 4 visual 4-panel figures per occlusion bucket
        bk = snap["bucket"]
        if panels_saved_per_bucket[bk] < 4:
            panel_name = f"{bk}_{snap['seq']}_f{snap['frame_idx']:04d}.png"
            save_comparison_panel(
                out_path=panels_dir / panel_name,
                seq_name=snap["seq"],
                frame_idx=snap["frame_idx"],
                occ=snap["occ"],
                track_iou=snap["track_iou"],
                vis_prob=snap["vis_prob"],
                srch_pil=snap["srch_pil"],
                rex_map=rex_map,
                evt_rgb=snap["evt_rgb"],
                evt_bin=snap["evt_bin"],
                gt_box_norm=snap["gt_box_norm"],
                pred_box_norm=snap["pred_box_norm"],
                metrics=metrics,
            )
            panels_saved_per_bucket[bk] += 1

        print(
            f"  [{idx:02d}/{len(snapshots)}] {snap['seq']} f{snap['frame_idx']:04d} ({bk:9s}, occ={snap['occ']:3d}) | "
            f"{rec['sec']:4.1f}s | pass={rec['rex_passing']:3d} | "
            f"NSS={metrics['nss']:+.2f} AUC={metrics['roc_auc']:.3f} CC={metrics['cc']:+.3f} "
            f"Top20IoU={metrics['top20_iou']:.3f} | ReX_GT={100*metrics['rex_in_gt_frac']:4.1f}% Evt_GT={100*metrics['evt_in_gt_frac']:4.1f}%",
            flush=True,
        )

    # Aggregate results overall and by occlusion bucket
    print(f"\n[3/3] Aggregating ReX <-> Event alignment metrics (elapsed {time.time()-t0:.1f}s)...", flush=True)
    metric_keys = [
        "nss", "roc_auc", "cc", "sim", "top20_iou", "top20_dice",
        "evt_recall_top20", "rex_in_gt_frac", "rex_in_pred_frac",
        "evt_in_gt_frac", "gt_area_frac", "nss_target_events", "auc_target_events",
    ]

    def agg(sub: List[Dict]) -> Dict[str, float]:
        out_d = {"count": len(sub)}
        if not sub:
            return out_d
        for k in metric_keys:
            vals = [r["metrics"][k] for r in sub if not np.isnan(r["metrics"][k])]
            out_d[k] = float(np.mean(vals)) if vals else float("nan")
        out_d["mean_track_iou"] = float(np.mean([r["track_iou"] for r in sub]))
        out_d["mean_vis_prob"] = float(np.mean([r["vis_prob"] for r in sub]))
        return out_d

    summary = {
        "args": vars(args),
        "overall": agg(results),
        "by_occlusion": {
            bk: agg([r for r in results if r["bucket"] == bk])
            for bk in ["occ_0", "occ_10_40", "occ_50_90", "occ_100"]
        },
        "frames": results,
    }
    (out_dir / "rex_event_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== ReX Responsibility Map vs. FEOT Event Picture Alignment ===", flush=True)
    for label, d in [("OVERALL", summary["overall"])] + [(bk, summary["by_occlusion"][bk]) for bk in ["occ_0", "occ_10_40", "occ_50_90", "occ_100"]]:
        if d["count"] == 0:
            continue
        print(
            f"  {label:10s} (n={d['count']:2d}) | NSS={d['nss']:+.3f} | AUC={d['roc_auc']:.3f} | "
            f"CC={d['cc']:+.3f} | SIM={d['sim']:.3f} | Top20-IoU={d['top20_iou']:.3f} | "
            f"TargetEvt-AUC={d['auc_target_events']:.3f} | ReX-in-GT={100*d['rex_in_gt_frac']:.1f}% vs Evt-in-GT={100*d['evt_in_gt_frac']:.1f}%",
            flush=True,
        )

    # Save summary chart across occlusion levels
    bks = [bk for bk in ["occ_0", "occ_10_40", "occ_50_90", "occ_100"] if summary["by_occlusion"][bk]["count"] > 0]
    if bks:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        x = np.arange(len(bks))
        labels = [f"{b}\n(n={summary['by_occlusion'][b]['count']})" for b in bks]

        ax1.plot(x, [summary["by_occlusion"][b]["roc_auc"] for b in bks], "o-", lw=2, label="ROC-AUC (All Events)")
        ax1.plot(x, [summary["by_occlusion"][b]["auc_target_events"] for b in bks], "s-", lw=2, label="ROC-AUC (Target Events)")
        ax1.axhline(0.5, color="gray", ls="--", lw=1, label="Random Baseline (0.5)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0.35, 1.0)
        ax1.set_ylabel("ROC-AUC")
        ax1.set_title("ReX Causal Map vs. Event Activity (AUC)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        w_bar = 0.26
        ax2.bar(x - w_bar, [100 * summary["by_occlusion"][b]["rex_in_gt_frac"] for b in bks], width=w_bar, label="ReX Mass in GT Box (%)")
        ax2.bar(x,         [100 * summary["by_occlusion"][b]["evt_in_gt_frac"] for b in bks], width=w_bar, label="Event Pixels in GT Box (%)")
        ax2.bar(x + w_bar, [100 * summary["by_occlusion"][b]["gt_area_frac"]   for b in bks], width=w_bar, label="GT Box Area (% of crop)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Percentage (%)")
        ax2.set_title("Spatial Concentration Inside Target GT Box")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(out_dir / "rex_alignment_summary.png", dpi=150)
        plt.close(fig)
        print(f"Saved summary chart to {out_dir / 'rex_alignment_summary.png'}", flush=True)


if __name__ == "__main__":
    main()
