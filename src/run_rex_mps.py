import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJ_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_PROJ_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(_PROJ_ROOT / ".cache" / "torch"))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import rankdata, wilcoxon
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from rex_xai.explanation.explanation import Explanation
from rex_xai.explanation.rex import calculate_responsibility, predict_target
from rex_xai.input.config import CausalArgs
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import Prediction
from rex_xai.utils._utils import ResponsibilityStyle, Strategy

from feot_dataset import IMAGENET_MEAN, IMAGENET_STD, load_sequence_meta
from train_classifier import build_resnet50_classifier, crop_gt_box


def compute_mann_whitney_auc(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    """
    Exact ROC-AUC (Mann-Whitney U with tie handling):
    Does a positive pixel (MPS = 1) tend to have higher `scores` (event count) than a negative pixel (MPS = 0)?
    0.5 = chance, > 0.5 = MPS pixels have more events.
    """
    y = binary_labels.ravel().astype(bool)
    s = scores.ravel().astype(np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    pos_rank_sum = ranks[y].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bin_events_in_crop(
    t_arr: np.ndarray,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    t_center_us: float,
    window_ms: float,
    crop_bounds: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Center a fixed event window of `window_ms` around `t_center_us` -> [t - W/2, t + W/2],
    and return a 2D integer array of shape (h, w) giving raw event counts per pixel inside
    the GT crop bounds (x1, y1, x2, y2).
    """
    half_us = 0.5 * window_ms * 1000.0
    lo = int(np.searchsorted(t_arr, t_center_us - half_us, side="left"))
    hi = int(np.searchsorted(t_arr, t_center_us + half_us, side="right"))

    x1, y1, x2, y2 = crop_bounds
    w = x2 - x1
    h = y2 - y1
    counts = np.zeros((h, w), dtype=np.int32)
    if hi <= lo:
        return counts

    xs = x_arr[lo:hi]
    ys = y_arr[lo:hi]
    mask = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    if not np.any(mask):
        return counts

    cx = xs[mask].astype(np.int32) - x1
    cy = ys[mask].astype(np.int32) - y1
    np.add.at(counts, (cy, cx), 1)
    return counts


def score_mps_vs_events(mps_native: np.ndarray, count_map: np.ndarray) -> Dict[str, float]:
    """
    Exact overlap metrics inside the GT crop:
      - p(n) = count(n) / total_events
      - event_recall = sum_{n in MPS} p(n)
      - mps_area_frac = |MPS| / |Crop|
      - fold_enrichment = event_recall / mps_area_frac
      - auc = P(count(MPS pixel) > count(non-MPS pixel)) + 0.5 * P(tie)
    """
    total_events = int(count_map.sum())
    mps_bool = mps_native.astype(bool)
    mps_area_frac = float(mps_bool.mean())

    if total_events == 0 or mps_area_frac <= 0.0 or mps_area_frac >= 1.0:
        return {
            "total_events": total_events,
            "mps_area_frac": mps_area_frac,
            "event_recall": float("nan"),
            "fold_enrichment": float("nan"),
            "auc": float("nan"),
        }

    p_map = count_map.astype(np.float64) / float(total_events)
    event_recall = float(p_map[mps_bool].sum())
    fold_enrichment = float(event_recall / mps_area_frac)
    auc = compute_mann_whitney_auc(count_map, mps_bool)

    return {
        "total_events": total_events,
        "mps_area_frac": mps_area_frac,
        "event_recall": event_recall,
        "fold_enrichment": fold_enrichment,
        "auc": auc,
    }


def make_classifier_pred_fn(model: torch.nn.Module, device: torch.device):
    def pred_fn(mutants: torch.Tensor, target=None, raw: bool = False) -> List[Prediction]:
        if mutants.ndim == 3:
            mutants = mutants.unsqueeze(0)
        mutants = mutants.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(mutants)
        probs = F.softmax(logits.float(), dim=-1)
        if raw:
            return probs
        top_conf, top_cls = probs.max(dim=-1)
        preds = []
        for i in range(mutants.shape[0]):
            p = Prediction(pred=int(top_cls[i].item()), conf=float(top_conf[i].item()))
            if target is not None:
                p.target = target.classification
                p.target_confidence = float(probs[i, target.classification].item())
            preds.append(p)
        return preds

    return pred_fn


def run_rex_extract_mps(
    model: torch.nn.Module,
    crop_pil: Image.Image,
    crop_tensor: torch.Tensor,
    device: torch.device,
    iters: int = 20,
    batch_size: int = 32,
    chunk_size: int = 128,
    conf_threshold: float = 0.75,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Run native rex_xai on a 224x224 RGB GT-box crop and extract the binary Minimal
    Sufficient Pixel Set (MPS) via `Explanation.extract()`.
    """
    args = CausalArgs()
    args.iters = iters
    args.batch_size = batch_size
    args.tree_depth = 8
    args.min_box_size = 10
    args.mask_value = 0.0  # ImageNet mean in normalized tensor space
    args.confidence_filter = conf_threshold
    args.minimum_confidence_threshold = conf_threshold
    args.responsibility_style = ResponsibilityStyle.Additive
    args.strategy = Strategy.Global
    args.chunk_size = chunk_size
    args.progress_bar = False
    args.verbosity = 0
    args.seed = seed

    data = Data(crop_pil, (1, 3, 224, 224), device=device, mode="RGB")
    data.data = crop_tensor.unsqueeze(0).to(device)
    data.transposed = True
    data.set_mask_value(0.0)

    pred_fn = make_classifier_pred_fn(model, device)
    data.target = predict_target(data, args, pred_fn)
    maps, stats = calculate_responsibility(data, args, pred_fn)

    exp = Explanation(maps, pred_fn, data, args, stats)
    exp.extract()

    rmap = maps.get(data.target.classification)
    rmap_np = np.zeros((224, 224), dtype=np.float32) if rmap is None else np.asarray(rmap, dtype=np.float32)

    if exp.sufficiency_mask is not None:
        sm = exp.sufficiency_mask.detach().cpu()
        if sm.ndim == 4:
            mps_224 = sm[0, 0].numpy().astype(bool)
        elif sm.ndim == 3:
            mps_224 = sm[0].numpy().astype(bool)
        else:
            mps_224 = sm.numpy().astype(bool)
        assert mps_224.shape == (224, 224), f"Unexpected MPS shape: {mps_224.shape}"
        suff_conf = float(exp.sufficiency_confidence) if exp.sufficiency_confidence is not None else 0.0
    else:
        mps_224 = np.zeros((224, 224), dtype=bool)
        suff_conf = 0.0

    return rmap_np, mps_224, suff_conf, stats


def save_mps_panel(
    out_path: Path,
    seq: str,
    fi: int,
    cat_id: int,
    pred_conf: float,
    crop_pil: Image.Image,
    rmap_224: np.ndarray,
    mps_224: np.ndarray,
    count_map_10ms: np.ndarray,
    mps_native: np.ndarray,
    m10: Dict[str, float],
):
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0))
    fig.suptitle(
        f"{seq} | frame {fi} | cat={cat_id} (conf={pred_conf:.2f}) | 10ms bin ({m10['total_events']} events) | "
        f"MPS={100*m10['mps_area_frac']:.1f}% | Recall={100*m10['event_recall']:.1f}% | "
        f"Enrichment={m10['fold_enrichment']:.2f}x | AUC={m10['auc']:.3f}",
        fontsize=10.5,
        fontweight="bold",
    )

    axes[0].imshow(crop_pil)
    axes[0].set_title("1. RGB GT Crop (224x224)")
    axes[0].axis("off")

    rm = np.maximum(rmap_224, 0.0)
    rm = rm / rm.max() if rm.max() > 1e-12 else rm
    axes[1].imshow(crop_pil)
    axes[1].imshow(rm, cmap="inferno", alpha=0.60, vmin=0.0, vmax=1.0)
    axes[1].set_title("2. ReX Responsibility Map")
    axes[1].axis("off")

    # Binary MPS overlaid on RGB (kept pixels bright, masked pixels darkened)
    arr = np.asarray(crop_pil).astype(np.float32) / 255.0
    vis_mps = arr * np.where(mps_224[..., None], 1.0, 0.20)
    axes[2].imshow(vis_mps)
    axes[2].contour(mps_224.astype(float), levels=[0.5], colors=["#00ff66"], linewidths=1.5)
    axes[2].set_title(f"3. Minimal Sufficient Set ({100*mps_224.mean():.1f}%)")
    axes[2].axis("off")

    # Native 10 ms raw event counts (log1p for visibility) + MPS boundary in green
    axes[3].imshow(np.log1p(count_map_10ms), cmap="magma", interpolation="nearest")
    if 0.0 < mps_native.mean() < 1.0:
        axes[3].contour(mps_native.astype(float), levels=[0.5], colors=["#00ff66"], linewidths=1.5)
    axes[3].set_title(f"4. 10ms Event Counts + MPS ({count_map_10ms.shape[1]}x{count_map_10ms.shape[0]})")
    axes[3].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Steps 2-4: ReX MPS & Centered Event-Window Overlap Scoring")
    parser.add_argument("--data_root", type=str, default=str(_PROJ_ROOT / "data" / "FEOT_100"))
    parser.add_argument("--ckpt", type=str, default=str(_PROJ_ROOT / "checkpoints" / "resnet50_classifier_best.pt"))
    parser.add_argument("--out_dir", type=str, default=str(_PROJ_ROOT / "outputs" / "rex_mps"))
    parser.add_argument("--frames_per_seq", type=int, default=10)
    parser.add_argument("--windows_ms", type=float, nargs="+", default=[10.0, 20.0, 40.0])
    parser.add_argument("--min_events", type=int, default=20)
    parser.add_argument("--rex_iters", type=int, default=20)
    parser.add_argument("--rex_batch", type=int, default=32)
    parser.add_argument("--conf_threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cat_ids = ckpt["cat_ids"]
    cat_to_idx = ckpt["cat_to_idx"]
    idx_to_cat = ckpt["idx_to_cat"]

    model = build_resnet50_classifier(num_classes=len(cat_ids), pretrained=False, freeze_early=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded resnet-50 classifier (epoch {ckpt['epoch']}, {len(cat_ids)} classes)", flush=True)

    data_root = Path(args.data_root)
    ds_dir = data_root / "FEOT_dataset"
    out_dir = Path(args.out_dir)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    test_seqs = [l.strip() for l in (data_root / "test_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    # Track detailed exclusion accounting across all 20 test sequences
    exclusions = {
        "total_test_sequences": len(test_seqs),
        "total_test_frames": 0,
        "seqs_with_zero_occ0": 0,
        "frames_excluded_occluded": 0,
        "frames_occ0_total": 0,
        "frames_occ0_misclassified": 0,
        "frames_occ0_correctly_classified": 0,
        "seqs_with_zero_correct_occ0": 0,
        "frames_sampled_for_rex": 0,
        "frames_excluded_degenerate_mps": 0,
        "frames_valid_mps": 0,
        "per_window_excluded_low_events": {str(w): 0 for w in args.windows_ms},
        "per_window_scored_frames": {str(w): 0 for w in args.windows_ms},
        "per_window_scored_sequences": {str(w): 0 for w in args.windows_ms},
        "per_sequence_log": [],
    }

    frame_records = []
    panels_saved = 0
    t_start = time.time()

    for seq_idx, seq in enumerate(test_seqs, 1):
        meta = load_sequence_meta(ds_dir / seq)
        N = len(meta["frames"])
        exclusions["total_test_frames"] += N

        occ0_idxs = [i for i in range(N) if int(meta["occs"][i]) == 0]
        exclusions["frames_excluded_occluded"] += (N - len(occ0_idxs))
        exclusions["frames_occ0_total"] += len(occ0_idxs)

        if not occ0_idxs:
            exclusions["seqs_with_zero_occ0"] += 1
            exclusions["per_sequence_log"].append({
                "seq": seq, "total_frames": N, "occ0_frames": 0,
                "correct_occ0": 0, "misclassified_occ0": 0, "sampled_for_rex": 0, "valid_mps": 0,
                "note": "No unoccluded (occ==0) frames",
            })
            print(f"[{seq_idx:02d}/20] {seq}: 0 occ==0 frames (excluded)", flush=True)
            continue

        # Classify all occ==0 frames in this sequence to find correctly classified ones
        correct_items = []
        misclf = 0
        for fi in occ0_idxs:
            gt_cat = int(meta["cats"][fi])
            gt_cls = cat_to_idx[gt_cat]
            with Image.open(meta["frames"][fi]) as im:
                img = im.convert("RGB")
            crop_pil, bounds = crop_gt_box(img, meta["boxes"][fi], out_size=224)
            t_in = TF.normalize(TF.to_tensor(crop_pil), IMAGENET_MEAN, IMAGENET_STD)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(t_in.unsqueeze(0).to(device))
                probs = F.softmax(logits.float(), dim=-1)[0]
            pred_cls = int(probs.argmax().item())
            pred_conf = float(probs[pred_cls].item())
            if pred_cls == gt_cls:
                correct_items.append((fi, gt_cat, pred_conf, crop_pil, t_in, bounds))
            else:
                misclf += 1

        exclusions["frames_occ0_misclassified"] += misclf
        exclusions["frames_occ0_correctly_classified"] += len(correct_items)

        if not correct_items:
            exclusions["seqs_with_zero_correct_occ0"] += 1
            exclusions["per_sequence_log"].append({
                "seq": seq, "total_frames": N, "occ0_frames": len(occ0_idxs),
                "correct_occ0": 0, "misclassified_occ0": misclf, "sampled_for_rex": 0, "valid_mps": 0,
                "note": "All occ==0 frames misclassified",
            })
            print(f"[{seq_idx:02d}/20] {seq}: {len(occ0_idxs)} occ==0 frames, 0 correctly classified (excluded)", flush=True)
            continue

        # Sample up to `frames_per_seq` evenly spaced correctly classified occ==0 frames
        if len(correct_items) > args.frames_per_seq:
            pick = np.linspace(0, len(correct_items) - 1, args.frames_per_seq, dtype=int)
            sampled = [correct_items[i] for i in pick]
        else:
            sampled = correct_items

        exclusions["frames_sampled_for_rex"] += len(sampled)

        # Load aligned_event.h5 ONCE for this sequence
        h5_path = ds_dir / seq / "aligned_event.h5"
        with h5py.File(h5_path, "r") as hf:
            t_ev = hf["t"][:]
            x_ev = hf["x"][:]
            y_ev = hf["y"][:]

        seq_valid_mps = 0
        seq_panel_done = False

        for k_idx, (fi, gt_cat, pred_conf, crop_pil, t_in, bounds) in enumerate(sampled):
            rmap_224, mps_224, suff_conf, rex_stats = run_rex_extract_mps(
                model=model,
                crop_pil=crop_pil,
                crop_tensor=t_in,
                device=device,
                iters=args.rex_iters,
                batch_size=args.rex_batch,
                chunk_size=128,
                conf_threshold=args.conf_threshold,
                seed=args.seed + seq_idx * 100 + k_idx,
            )

            x1, y1, x2, y2 = bounds
            w_nat, h_nat = x2 - x1, y2 - y1
            # Map binary 224x224 MPS back to native GT-box pixel grid (h_nat, w_nat) via nearest neighbor
            mps_native = np.asarray(
                Image.fromarray(mps_224.astype(np.uint8) * 255).resize((w_nat, h_nat), Image.NEAREST)
            ) > 127

            mps_frac = float(mps_native.mean())
            if mps_frac <= 0.0 or mps_frac >= 1.0:
                exclusions["frames_excluded_degenerate_mps"] += 1
                continue

            seq_valid_mps += 1
            exclusions["frames_valid_mps"] += 1

            t_rgb_us = float(meta["timestamps"][fi])
            win_scores = {}
            count_10ms = None
            for w_ms in args.windows_ms:
                w_key = str(w_ms)
                cmap = bin_events_in_crop(t_ev, x_ev, y_ev, t_rgb_us, w_ms, bounds)
                if w_ms == args.windows_ms[0]:
                    count_10ms = cmap
                tot_e = int(cmap.sum())
                if tot_e < args.min_events:
                    exclusions["per_window_excluded_low_events"][w_key] += 1
                    win_scores[w_key] = {"skipped_low_events": True, "total_events": tot_e}
                else:
                    exclusions["per_window_scored_frames"][w_key] += 1
                    sc = score_mps_vs_events(mps_native, cmap)
                    sc["skipped_low_events"] = False
                    win_scores[w_key] = sc

            frame_records.append({
                "seq": seq,
                "frame_idx": fi,
                "cat_id": gt_cat,
                "pred_conf": round(pred_conf, 4),
                "suff_conf": round(suff_conf, 4),
                "crop_w": w_nat,
                "crop_h": h_nat,
                "mps_area_frac": round(mps_frac, 4),
                "windows": win_scores,
            })

            # Save 1 representative visual panel per test sequence
            w0_key = str(args.windows_ms[0])
            if (not seq_panel_done) and panels_saved < 20 and (not win_scores[w0_key]["skipped_low_events"]):
                save_mps_panel(
                    out_path=panels_dir / f"mps_{seq}_f{fi:04d}.png",
                    seq=seq, fi=fi, cat_id=gt_cat, pred_conf=pred_conf,
                    crop_pil=crop_pil, rmap_224=rmap_224, mps_224=mps_224,
                    count_map_10ms=count_10ms, mps_native=mps_native, m10=win_scores[w0_key],
                )
                seq_panel_done = True
                panels_saved += 1

        exclusions["per_sequence_log"].append({
            "seq": seq,
            "total_frames": N,
            "occ0_frames": len(occ0_idxs),
            "correct_occ0": len(correct_items),
            "misclassified_occ0": misclf,
            "sampled_for_rex": len(sampled),
            "valid_mps": seq_valid_mps,
        })
        print(
            f"[{seq_idx:02d}/20] {seq}: occ0={len(occ0_idxs):3d}, correct={len(correct_items):3d}, "
            f"misclf={misclf:3d} -> sampled={len(sampled):2d}, valid_mps={seq_valid_mps:2d} ({time.time()-t_start:.0f}s)",
            flush=True,
        )

    # Hierarchical aggregation: average within each sequence first, then across sequences!
    window_summaries = {}
    for w_ms in args.windows_ms:
        w_key = str(w_ms)
        by_seq: Dict[str, List[Dict]] = {}
        for rec in frame_records:
            ws = rec["windows"][w_key]
            if not ws["skipped_low_events"] and not np.isnan(ws["auc"]):
                by_seq.setdefault(rec["seq"], []).append(ws)

        seq_means = []
        for s in sorted(by_seq.keys()):
            lst = by_seq[s]
            seq_means.append({
                "seq": s,
                "n_frames": len(lst),
                "mean_events": float(np.mean([x["total_events"] for x in lst])),
                "mps_area_frac": float(np.mean([x["mps_area_frac"] for x in lst])),
                "event_recall": float(np.mean([x["event_recall"] for x in lst])),
                "fold_enrichment": float(np.mean([x["fold_enrichment"] for x in lst])),
                "auc": float(np.mean([x["auc"] for x in lst])),
            })

        exclusions["per_window_scored_sequences"][w_key] = len(seq_means)

        def stats_over_seqs(field: str, null_val: Optional[float] = None) -> Dict:
            vals = np.array([sm[field] for sm in seq_means], dtype=np.float64)
            out_s = {
                "mean": float(vals.mean()) if len(vals) else float("nan"),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)) if len(vals) else float("nan"),
            }
            if null_val is not None and len(vals) >= 5:
                try:
                    stat, pval = wilcoxon(vals - null_val, alternative="greater")
                    out_s["wilcoxon_p_greater_than_chance"] = float(pval)
                except Exception:
                    pass
            return out_s

        window_summaries[w_key] = {
            "window_ms": w_ms,
            "n_sequences": len(seq_means),
            "n_frames_scored": sum(sm["n_frames"] for sm in seq_means),
            "mps_area_frac": stats_over_seqs("mps_area_frac"),
            "event_recall": stats_over_seqs("event_recall"),
            "fold_enrichment": stats_over_seqs("fold_enrichment", null_val=1.0),
            "auc": stats_over_seqs("auc", null_val=0.5),
            "per_sequence": seq_means,
        }

    final_out = {
        "args": vars(args),
        "exclusions": exclusions,
        "window_summaries": window_summaries,
        "frames": frame_records,
    }
    (out_dir / "rex_mps_overlap_results.json").write_text(json.dumps(final_out, indent=2), encoding="utf-8")

    print("\n================ EXCLUSION SUMMARY ================", flush=True)
    print(f"  Test sequences total                : {exclusions['total_test_sequences']} ({exclusions['total_test_frames']} frames)", flush=True)
    print(f"  Excluded: occluded (occ > 0)        : {exclusions['frames_excluded_occluded']} frames ({exclusions['seqs_with_zero_occ0']} seqs had 0 occ==0 frames)", flush=True)
    print(f"  Unoccluded (occ == 0) frames        : {exclusions['frames_occ0_total']}", flush=True)
    print(f"  Excluded: misclassified by CNN      : {exclusions['frames_occ0_misclassified']} frames ({exclusions['seqs_with_zero_correct_occ0']} seqs had 0 correct occ==0)", flush=True)
    print(f"  Correctly classified occ==0 frames  : {exclusions['frames_occ0_correctly_classified']}", flush=True)
    print(f"  Sampled for ReX (<= {args.frames_per_seq}/seq)       : {exclusions['frames_sampled_for_rex']} frames", flush=True)
    print(f"  Excluded: degenerate MPS (0%/100%)  : {exclusions['frames_excluded_degenerate_mps']} frames", flush=True)
    print(f"  Valid MPS frames                    : {exclusions['frames_valid_mps']} frames", flush=True)

    print("\n=== REX MPS OVERLAP METRICS (Averaged Within Sequence -> Across Sequences) ===", flush=True)
    for w_ms in args.windows_ms:
        w_key = str(w_ms)
        ws = window_summaries[w_key]
        fe = ws["fold_enrichment"]
        au = ws["auc"]
        er = ws["event_recall"]
        ma = ws["mps_area_frac"]
        p_fe = f" (p={fe['wilcoxon_p_greater_than_chance']:.4f})" if "wilcoxon_p_greater_than_chance" in fe else ""
        p_au = f" (p={au['wilcoxon_p_greater_than_chance']:.4f})" if "wilcoxon_p_greater_than_chance" in au else ""
        print(
            f"  Window {w_ms:4.0f} ms | seqs={ws['n_sequences']:2d}, frames={ws['n_frames_scored']:3d} "
            f"(skipped low-evt={exclusions['per_window_excluded_low_events'][w_key]:2d}) | "
            f"MPS Area={100*ma['mean']:4.1f}% | "
            f"Event Recall={100*er['mean']:5.1f}% ± {100*er['std']:4.1f}% | "
            f"Fold Enrich={fe['mean']:.2f}x ± {fe['std']:.2f}{p_fe} | "
            f"AUC={au['mean']:.3f} ± {au['std']:.3f}{p_au}",
            flush=True,
        )

    # Plot window sensitivity comparison chart
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    w_labels = [f"{int(w)} ms" for w in args.windows_ms]
    x = np.arange(len(args.windows_ms))

    # 1. Event Recall vs MPS Area
    rec_means = [100 * window_summaries[str(w)]["event_recall"]["mean"] for w in args.windows_ms]
    rec_stds  = [100 * window_summaries[str(w)]["event_recall"]["std"] for w in args.windows_ms]
    mps_means = [100 * window_summaries[str(w)]["mps_area_frac"]["mean"] for w in args.windows_ms]
    axes[0].bar(x - 0.18, rec_means, yerr=rec_stds, width=0.34, capsize=4, label="Event Recall (%)")
    axes[0].bar(x + 0.18, mps_means, width=0.34, label="MPS Area Fraction (%)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(w_labels)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Event Recall vs. MPS Area")
    axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")

    # 2. Fold Enrichment (1.0 = chance)
    fe_means = [window_summaries[str(w)]["fold_enrichment"]["mean"] for w in args.windows_ms]
    fe_stds  = [window_summaries[str(w)]["fold_enrichment"]["std"] for w in args.windows_ms]
    axes[1].bar(x, fe_means, yerr=fe_stds, width=0.45, capsize=5, color="#0d9488")
    axes[1].axhline(1.0, color="gray", ls="--", lw=1.2, label="Random Chance (1.0x)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(w_labels)
    axes[1].set_ylabel("Fold Enrichment (Recall / Area)")
    axes[1].set_title("Fold Enrichment Across Event Windows")
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

    # 3. MPS vs Non-MPS AUC (0.5 = chance)
    auc_means = [window_summaries[str(w)]["auc"]["mean"] for w in args.windows_ms]
    auc_stds  = [window_summaries[str(w)]["auc"]["std"] for w in args.windows_ms]
    axes[2].bar(x, auc_means, yerr=auc_stds, width=0.45, capsize=5, color="#2563eb")
    axes[2].axhline(0.5, color="gray", ls="--", lw=1.2, label="Random Chance (0.50)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(w_labels)
    axes[2].set_ylim(0.35, 0.85)
    axes[2].set_ylabel("AUC (MPS vs. Non-MPS Event Count)")
    axes[2].set_title("Pixel-Level Event AUC Across Windows")
    axes[2].legend(); axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_dir / "window_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved summary chart to {out_dir / 'window_comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
