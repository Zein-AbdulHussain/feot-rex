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
import torch

from rex_xai.explanation.rex import calculate_responsibility, predict_target
from rex_xai.input.config import CausalArgs
from rex_xai.input.input_data import Data
from rex_xai.responsibility.prediction import Prediction
from rex_xai.utils._utils import ResponsibilityStyle

from losses import box_iou_and_giou
from models import ResNet50LSTMTracker, ResNet50TrackerCNN, TemporalLSTMHead
from run_rex_eval import (
    collect_test_frame_snapshots,
    compute_alignment_metrics,
    run_rex_on_frame,
)


def run_rex_stage1_cnn(
    cnn: ResNet50TrackerCNN,
    srch_crop_pil: Image.Image,
    srch_tensor: torch.Tensor,
    tmpl_enc: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    iters: int = 16,
    batch_size: int = 16,
    seed: int = 42,
    iou_thresh: float = 0.60,
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """Run ReX on Stage-1 ResNet-50 CNN alone on the exact same 256x256 RGB search crop."""
    srch_dev = srch_tensor.to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        orig_out = cnn.forward_with_template(tmpl_enc, srch_dev)
    orig_box = orig_out["box"][0].float()
    orig_conf = float(orig_out["confidence"][0].item())

    args = CausalArgs()
    args.iters = iters
    args.batch_size = batch_size
    args.tree_depth = 8
    args.min_box_size = 12
    args.mask_value = 0.0
    args.responsibility_style = ResponsibilityStyle.Additive
    args.progress_bar = False
    args.verbosity = 0
    args.seed = seed

    data = Data(srch_crop_pil, (1, 3, 256, 256), device=device, mode="RGB")
    data.data = srch_dev
    data.transposed = True
    data.set_mask_value(0.0)

    def pred_fn(mutants: torch.Tensor, target=None, raw: bool = False) -> List[Prediction]:
        if mutants.ndim == 3:
            mutants = mutants.unsqueeze(0)
        mutants = mutants.to(device, non_blocking=True)
        B = mutants.shape[0]
        k_exp = tmpl_enc[0].expand(B, -1, -1, -1)
        g_exp = tmpl_enc[1].expand(B, -1, -1, -1)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = cnn.forward_with_template((k_exp, g_exp), mutants)

        pred_boxes = out["box"].float()
        confs = out["confidence"].float()
        ious, _ = box_iou_and_giou(pred_boxes, orig_box.expand(B, -1))
        conf_ratio = (confs / max(orig_conf, 1e-4)).clamp(0.0, 1.0)
        passed = (ious >= iou_thresh) & (confs >= 0.55 * orig_conf)
        score_1 = (0.5 * ious.clamp(0.0, 1.0) + 0.5 * conf_ratio).clamp(0.0, 1.0)

        if raw:
            return torch.stack([1.0 - score_1, score_1], dim=-1)

        preds = []
        for i in range(B):
            cls_id = 1 if bool(passed[i].item()) else 0
            s1 = float(score_1[i].item())
            c = s1 if cls_id == 1 else float(1.0 - s1)
            p = Prediction(pred=cls_id, conf=c)
            if target is not None:
                p.target = target.classification
                p.target_confidence = s1 if target.classification == 1 else (1.0 - s1)
            preds.append(p)
        return preds

    data.target = predict_target(data, args, pred_fn)
    maps, stats = calculate_responsibility(data, args, pred_fn)
    rmap = maps.get(data.target.classification)
    rmap = np.zeros((256, 256), dtype=np.float32) if rmap is None else np.asarray(rmap, dtype=np.float32)
    return rmap, orig_box.cpu().numpy(), orig_conf, stats


def save_s1_vs_s2_panel(
    out_path: Path,
    seq: str,
    fi: int,
    occ: int,
    srch_pil: Image.Image,
    rmap_s1: np.ndarray,
    rmap_s2: np.ndarray,
    evt_rgb: np.ndarray,
    gt_box: np.ndarray,
    box_s1: np.ndarray,
    box_s2: np.ndarray,
    m1: Dict[str, float],
    m2: Dict[str, float],
    s1_s2_cc: float,
):
    H, W = rmap_s1.shape
    r1 = np.maximum(rmap_s1, 0.0)
    r1 = r1 / r1.max() if r1.max() > 1e-12 else r1
    r2 = np.maximum(rmap_s2, 0.0)
    r2 = r2 / r2.max() if r2.max() > 1e-12 else r2

    def draw_box(ax, b, color, ls="-"):
        cx, cy, bw, bh = b
        ax.add_patch(
            patches.Rectangle(
                ((cx - 0.5 * bw) * W, (cy - 0.5 * bh) * H),
                bw * W, bh * H,
                linewidth=2.0, edgecolor=color, facecolor="none", linestyle=ls,
            )
        )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    fig.suptitle(
        f"{seq} | frame {fi} | occ={occ}% | ReX(S1,S2) CC={s1_s2_cc:+.2f} | "
        f"CNN-only: AUC={m1['roc_auc']:.2f}, NSS={m1['nss']:+.2f}  vs  "
        f"CNN+LSTM: AUC={m2['roc_auc']:.2f}, NSS={m2['nss']:+.2f}",
        fontsize=10.5,
        fontweight="bold",
    )

    axes[0].imshow(srch_pil)
    draw_box(axes[0], gt_box, "#00ff44", "--")
    draw_box(axes[0], box_s1, "#ffaa00", "-")
    draw_box(axes[0], box_s2, "#ff2233", "-")
    axes[0].set_title("1. RGB (GT=green, S1=orange, S2=red)")
    axes[0].axis("off")

    axes[1].imshow(srch_pil)
    axes[1].imshow(r1, cmap="inferno", alpha=0.60, vmin=0.0, vmax=1.0)
    draw_box(axes[1], gt_box, "#00ff44", "--")
    axes[1].set_title(f"2. Stage-1 CNN ReX (AUC={m1['roc_auc']:.2f})")
    axes[1].axis("off")

    axes[2].imshow(srch_pil)
    axes[2].imshow(r2, cmap="inferno", alpha=0.60, vmin=0.0, vmax=1.0)
    draw_box(axes[2], gt_box, "#00ff44", "--")
    axes[2].set_title(f"3. Stage-2 CNN+LSTM ReX (AUC={m2['roc_auc']:.2f})")
    axes[2].axis("off")

    axes[3].imshow(evt_rgb)
    draw_box(axes[3], gt_box, "#00aa33", "--")
    axes[3].set_title(f"4. FEOT Event Picture ({m2['n_event_pixels']} px)")
    axes[3].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare Stage-1 CNN vs Stage-2 CNN+LSTM ReX Maps on Identical Test Frames")
    parser.add_argument("--data_root", type=str, default=str(_PROJ_ROOT / "data" / "FEOT_100"))
    parser.add_argument("--out_dir", type=str, default=str(_PROJ_ROOT / "outputs" / "rex_compare"))
    parser.add_argument("--ckpt", type=str, default=str(_PROJ_ROOT / "checkpoints" / "resnet50_lstm_best.pt"))
    parser.add_argument("--max_per_bucket", type=int, default=10)
    parser.add_argument("--rex_iters", type=int, default=16)
    parser.add_argument("--rex_batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cnn = ResNet50TrackerCNN(pretrained=False, feat_dim=256, freeze_early=True)
    lstm_head = TemporalLSTMHead(hidden_size=256, num_layers=2, dropout=0.3)
    tracker = ResNet50LSTMTracker(cnn, lstm_head, freeze_cnn=True).to(device)
    tracker.load_state_dict(ckpt["full_state_dict"])
    tracker.eval()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Collecting balanced test frame snapshots across all 4 occlusion regimes...", flush=True)
    snapshots = collect_test_frame_snapshots(tracker, data_root, device, max_per_bucket=args.max_per_bucket)
    print(f"Comparing Stage-1 CNN vs Stage-2 CNN+LSTM ReX on {len(snapshots)} identical frames...", flush=True)

    records = []
    saved_per_bk = {"occ_0": 0, "occ_10_40": 0, "occ_50_90": 0, "occ_100": 0}
    t0 = time.time()

    for idx, snap in enumerate(snapshots, 1):
        tmpl_enc = (snap["tmpl_enc"][0].to(device), snap["tmpl_enc"][1].to(device))
        prev_h = (
            (snap["prev_hidden"][0].to(device), snap["prev_hidden"][1].to(device))
            if snap["prev_hidden"] is not None
            else None
        )

        # Stage-1 CNN only ReX
        rmap_s1, box_s1, conf_s1, _ = run_rex_stage1_cnn(
            cnn=tracker.cnn,
            srch_crop_pil=snap["srch_pil"],
            srch_tensor=snap["srch_t"],
            tmpl_enc=tmpl_enc,
            device=device,
            iters=args.rex_iters,
            batch_size=args.rex_batch,
            seed=args.seed + idx,
        )

        # Stage-2 CNN + LSTM ReX (exact same crop & seed)
        rmap_s2, _ = run_rex_on_frame(
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

        m1 = compute_alignment_metrics(rmap_s1, snap["evt_bin"], snap["gt_box_norm"], box_s1)
        m2 = compute_alignment_metrics(rmap_s2, snap["evt_bin"], snap["gt_box_norm"], snap["pred_box_norm"])

        r1_flat = np.maximum(rmap_s1, 0.0).ravel()
        r2_flat = np.maximum(rmap_s2, 0.0).ravel()
        if r1_flat.std() > 1e-12 and r2_flat.std() > 1e-12:
            s1_s2_cc = float(np.corrcoef(r1_flat, r2_flat)[0, 1])
        else:
            s1_s2_cc = 0.0

        records.append({
            "seq": snap["seq"],
            "frame_idx": snap["frame_idx"],
            "occ": snap["occ"],
            "bucket": snap["bucket"],
            "s1_s2_rex_cc": s1_s2_cc,
            "s1": m1,
            "s2": m2,
        })

        bk = snap["bucket"]
        if saved_per_bk[bk] < 2:
            save_s1_vs_s2_panel(
                out_path=panels_dir / f"cmp_{bk}_{snap['seq']}_f{snap['frame_idx']:04d}.png",
                seq=snap["seq"],
                fi=snap["frame_idx"],
                occ=snap["occ"],
                srch_pil=snap["srch_pil"],
                rmap_s1=rmap_s1,
                rmap_s2=rmap_s2,
                evt_rgb=snap["evt_rgb"],
                gt_box=snap["gt_box_norm"],
                box_s1=box_s1,
                box_s2=snap["pred_box_norm"],
                m1=m1,
                m2=m2,
                s1_s2_cc=s1_s2_cc,
            )
            saved_per_bk[bk] += 1

        if idx % 10 == 0 or idx == len(snapshots):
            print(f"  [{idx:02d}/{len(snapshots)}] elapsed {time.time()-t0:.1f}s | last={snap['seq']} ({bk}) S1_AUC={m1['roc_auc']:.3f} vs S2_AUC={m2['roc_auc']:.3f} | S1-S2 CC={s1_s2_cc:+.2f}", flush=True)

    keys = ["nss", "roc_auc", "auc_target_events", "cc", "sim", "top20_iou", "rex_in_gt_frac"]

    def agg(sub: List[Dict]) -> Dict:
        out = {"count": len(sub)}
        if not sub:
            return out
        out["s1_s2_rex_cc"] = float(np.mean([r["s1_s2_rex_cc"] for r in sub]))
        for tag in ("s1", "s2"):
            out[tag] = {}
            for k in keys:
                vals = [r[tag][k] for r in sub if not np.isnan(r[tag][k])]
                out[tag][k] = float(np.mean(vals)) if vals else float("nan")
        return out

    summary = {
        "overall": agg(records),
        "by_occlusion": {
            bk: agg([r for r in records if r["bucket"] == bk])
            for bk in ["occ_0", "occ_10_40", "occ_50_90", "occ_100"]
        },
        "frames": records,
    }
    (out_dir / "stage1_vs_stage2_rex.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Stage-1 (CNN Only) vs. Stage-2 (CNN + LSTM) ReX Alignment Comparison ===", flush=True)
    for label, d in [("OVERALL", summary["overall"])] + [(bk, summary["by_occlusion"][bk]) for bk in ["occ_0", "occ_10_40", "occ_50_90", "occ_100"]]:
        if d["count"] == 0:
            continue
        s1, s2 = d["s1"], d["s2"]
        print(
            f"  {label:10s} (n={d['count']:2d}) | S1-S2 CC={d['s1_s2_rex_cc']:+.3f} | "
            f"NSS: {s1['nss']:+.2f}->{s2['nss']:+.2f} | "
            f"AUC: {s1['roc_auc']:.3f}->{s2['roc_auc']:.3f} | "
            f"TgtAUC: {s1['auc_target_events']:.3f}->{s2['auc_target_events']:.3f} | "
            f"ReX-GT: {100*s1['rex_in_gt_frac']:.1f}%->{100*s2['rex_in_gt_frac']:.1f}%",
            flush=True,
        )

    # Save comparison chart
    bks = [bk for bk in ["occ_0", "occ_10_40", "occ_50_90", "occ_100"] if summary["by_occlusion"][bk]["count"] > 0]
    x = np.arange(len(bks))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    w = 0.32
    ax1.bar(x - w/2, [summary["by_occlusion"][b]["s1"]["nss"] for b in bks], width=w, label="Stage-1 CNN Only")
    ax1.bar(x + w/2, [summary["by_occlusion"][b]["s2"]["nss"] for b in bks], width=w, label="Stage-2 CNN + LSTM")
    ax1.axhline(0.0, color="gray", ls="--", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels(bks)
    ax1.set_ylabel("NSS (ReX vs Event Pixels)")
    ax1.set_title("Event Alignment (NSS): CNN Only vs. CNN+LSTM")
    ax1.legend(); ax1.grid(True, alpha=0.3, axis="y")

    ax2.plot(x, [summary["by_occlusion"][b]["s1"]["auc_target_events"] for b in bks], "o--", lw=2, label="Stage-1 CNN (Target Events AUC)")
    ax2.plot(x, [summary["by_occlusion"][b]["s2"]["auc_target_events"] for b in bks], "s-", lw=2, label="Stage-2 CNN+LSTM (Target Events AUC)")
    ax2.plot(x, [summary["by_occlusion"][b]["s1_s2_rex_cc"] for b in bks], "^:", lw=2, label="ReX(S1) vs ReX(S2) Spatial CC")
    ax2.set_xticks(x); ax2.set_xticklabels(bks)
    ax2.set_ylim(0.0, 1.05)
    ax2.set_ylabel("Score")
    ax2.set_title("Target-Event AUC & Stage-1 vs Stage-2 ReX Similarity")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "stage1_vs_stage2_rex_summary.png", dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart to {out_dir / 'stage1_vs_stage2_rex_summary.png'}", flush=True)


if __name__ == "__main__":
    main()
