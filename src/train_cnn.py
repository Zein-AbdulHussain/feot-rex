import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Use Windows root CA bundle and project-local torch cache for ImageNet pretrained weights
_PROJ_ROOT = Path(__file__).resolve().parent.parent
_WIN_CA = _PROJ_ROOT / ".venv" / "win_roots.pem"
if _WIN_CA.exists() and "SSL_CERT_FILE" not in os.environ:
    os.environ["SSL_CERT_FILE"] = str(_WIN_CA)
os.environ.setdefault("TORCH_HOME", str(_PROJ_ROOT / ".cache" / "torch"))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from feot_dataset import (
    FEOTPairDataset,
    compute_crop_size,
    crop_norm_to_frame_box,
    extract_square_crop,
    load_sequence_meta,
    pil_to_norm_tensor,
)
from losses import TrackerLoss, box_iou_and_giou
from models import ResNet50TrackerCNN


def compute_iou_xywh(b1: np.ndarray, b2: np.ndarray) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(1e-6, b1[2] * b1[3] + b2[2] * b2[3] - inter)
    return float(inter / union)


def compute_center_err(b1: np.ndarray, b2: np.ndarray) -> float:
    c1x, c1y = b1[0] + 0.5 * b1[2], b1[1] + 0.5 * b1[3]
    c2x, c2y = b2[0] + 0.5 * b2[2], b2[1] + 0.5 * b2[3]
    return float(np.hypot(c1x - c2x, c1y - c2y))


@torch.no_grad()
def evaluate_pairs(model: ResNet50TrackerCNN, loader: DataLoader, criterion: TrackerLoss, device: torch.device) -> Dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "giou": 0.0, "l1": 0.0, "hm": 0.0, "iou": 0.0}
    n_batches = 0
    for batch in loader:
        tmpl = batch["template"].to(device, non_blocking=True)
        srch = batch["search"].to(device, non_blocking=True)
        box_gt = batch["box"].to(device, non_blocking=True)
        hm_gt = batch["heatmap"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(tmpl, srch)
            _, metrics = criterion(out["box"], out["heatmap_logits"], box_gt, hm_gt)

        for k in sums:
            sums[k] += metrics[k]
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in sums.items()}


@torch.no_grad()
def evaluate_sequential_tracking(
    model: ResNet50TrackerCNN,
    data_root: Path,
    split_file: str,
    device: torch.device,
    template_size: int = 128,
    search_size: int = 256,
    conf_update_thresh: float = 0.25,
) -> Dict:
    """
    True online single-object tracking evaluation:
    Initializes ONLY from frame 0 GT box, then tracks frame-by-frame using the model's own
    predicted boxes as the next search anchor.
    """
    model.eval()
    ds_dir = data_root / "FEOT_dataset"
    seq_names = [
        l.strip()
        for l in (data_root / split_file).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    all_ious = []
    all_errs = []
    all_occs = []
    per_seq = []

    for name in seq_names:
        meta = load_sequence_meta(ds_dir / name)
        gt_boxes = meta["boxes"]
        occs = meta["occs"]
        frames = meta["frames"]

        # Encode frame 0 template once
        init_box = gt_boxes[0].copy()
        img0 = Image.open(frames[0]).convert("RGB")
        W, H = img0.size
        tcx = init_box[0] + 0.5 * init_box[2]
        tcy = init_box[1] + 0.5 * init_box[3]
        t_side = compute_crop_size(init_box[2], init_box[3], scale_factor=2.0, min_size=64.0)
        tmpl_crop, _ = extract_square_crop(img0, tcx, tcy, t_side, template_size)
        tmpl_t = pil_to_norm_tensor(tmpl_crop).unsqueeze(0).to(device)
        tmpl_encoded = model.encode_template(tmpl_t)

        cur_box = init_box.copy()
        seq_ious = [1.0]
        seq_errs = [0.0]

        for fi in range(1, len(frames)):
            img = Image.open(frames[fi]).convert("RGB")
            acx = cur_box[0] + 0.5 * cur_box[2]
            acy = cur_box[1] + 0.5 * cur_box[3]
            s_side = compute_crop_size(cur_box[2], cur_box[3], scale_factor=4.0, min_size=256.0)
            srch_crop, crop_meta = extract_square_crop(img, acx, acy, s_side, search_size)
            srch_t = pil_to_norm_tensor(srch_crop).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model.forward_with_template(tmpl_encoded, srch_t)

            pred_norm = out["box"][0].float().cpu().numpy()
            conf = float(out["confidence"][0].item())
            pred_box = crop_norm_to_frame_box(pred_norm, crop_meta, img_w=W, img_h=H)

            # Smooth size updates slightly and keep anchor stable if confidence is very low
            if conf >= conf_update_thresh:
                pred_box[2] = 0.7 * cur_box[2] + 0.3 * pred_box[2]
                pred_box[3] = 0.7 * cur_box[3] + 0.3 * pred_box[3]
                cur_box = pred_box
            else:
                # Low confidence (likely occluded): keep center close, gently drift toward prediction
                cur_box[0] = 0.7 * cur_box[0] + 0.3 * pred_box[0]
                cur_box[1] = 0.7 * cur_box[1] + 0.3 * pred_box[1]

            iou = compute_iou_xywh(cur_box, gt_boxes[fi])
            err = compute_center_err(cur_box, gt_boxes[fi])
            seq_ious.append(iou)
            seq_errs.append(err)

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
    parser = argparse.ArgumentParser(description="Stage 1: Train ResNet-50 RGB CNN Tracker on FEOT_100")
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent.parent / "data" / "FEOT_100"))
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples_per_epoch", type=int, default=4200)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    res_dir = out_dir / "outputs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FEOTPairDataset(
        data_root=data_root,
        split_file="train_list.txt",
        max_occ=80,
        samples_per_epoch=args.samples_per_epoch,
        augment=True,
        seed=args.seed,
    )
    val_ds = FEOTPairDataset(
        data_root=data_root,
        split_file="val_list.txt",
        max_occ=80,
        samples_per_epoch=None,
        augment=False,
        seed=args.seed,
    )
    print(f"Train dataset: {len(train_ds.sequences)} seqs ({len(train_ds)} pairs/epoch) | Val dataset: {len(val_ds.sequences)} seqs ({len(val_ds)} pairs)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = ResNet50TrackerCNN(pretrained=True, feat_dim=256, freeze_early=True).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet50TrackerCNN params: total={n_total/1e6:.2f}M, trainable={n_train/1e6:.2f}M")

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("layer3")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("layer3")]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = TrackerLoss(w_giou=2.0, w_l1=5.0, w_hm=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_iou = -1.0
    history: List[Dict] = []
    best_ckpt = ckpt_dir / "resnet50_cnn_best.pt"

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        # Keep frozen early BN layers in eval mode so ImageNet stats aren't corrupted
        model.stem.eval()
        model.layer1.eval()
        model.layer2.eval()

        tr_sums = {"loss": 0.0, "giou": 0.0, "l1": 0.0, "hm": 0.0, "iou": 0.0}
        n_steps = 0

        for batch in train_loader:
            tmpl = batch["template"].to(device, non_blocking=True)
            srch = batch["search"].to(device, non_blocking=True)
            box_gt = batch["box"].to(device, non_blocking=True)
            hm_gt = batch["heatmap"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(tmpl, srch)
                loss, metrics = criterion(out["box"], out["heatmap_logits"], box_gt, hm_gt)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            for k in tr_sums:
                tr_sums[k] += metrics[k]
            n_steps += 1

        scheduler.step()
        tr_avg = {k: v / max(1, n_steps) for k, v in tr_sums.items()}
        val_avg = evaluate_pairs(model, val_loader, criterion, device)

        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "lr_head": optimizer.param_groups[1]["lr"],
            "train": tr_avg,
            "val": val_avg,
            "sec": round(dt, 1),
        }
        history.append(row)

        saved_tag = ""
        if val_avg["iou"] > best_val_iou:
            best_val_iou = val_avg["iou"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_avg,
                    "args": vars(args),
                },
                best_ckpt,
            )
            saved_tag = " *BEST*"

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:4.1f}s) | "
            f"Train loss={tr_avg['loss']:.3f} IoU={tr_avg['iou']:.3f} | "
            f"Val loss={val_avg['loss']:.3f} IoU={val_avg['iou']:.3f}{saved_tag}",
            flush=True,
        )

    print(f"\nTraining finished in {(time.time() - t_start)/60:.1f} min. Best Val Pair IoU: {best_val_iou:.4f}")

    # Load best checkpoint for full online sequential tracking evaluation on val and test
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nRunning online sequential tracking evaluation (Stage-1 ResNet-50 CNN)...")
    val_track = evaluate_sequential_tracking(model, data_root, "val_list.txt", device)
    test_track = evaluate_sequential_tracking(model, data_root, "test_list.txt", device)

    summary = {
        "best_epoch": ckpt["epoch"],
        "best_val_pair_metrics": ckpt["val_metrics"],
        "history": history,
        "val_tracking": val_track,
        "test_tracking": test_track,
    }
    eval_path = res_dir / "stage1_cnn_eval.json"
    eval_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for label, res in [("VAL (10 seqs)", val_track), ("TEST (20 seqs)", test_track)]:
        ov = res["overall"]
        print(f"\n=== Stage-1 ResNet-50 CNN Sequential Tracking — {label} ===")
        print(f"  Overall:   Mean IoU={ov['mean_iou']:.4f} | IoU@0.5={ov['iou_50']:.4f} | Prec@20px={ov['prec_20px']:.4f} (n={ov['count']})")
        for bk, st in res["by_occlusion"].items():
            print(f"  {bk:10s}: Mean IoU={st['mean_iou']:.4f} | IoU@0.5={st['iou_50']:.4f} | Prec@20px={st['prec_20px']:.4f} (n={st['count']})")

    # Save training curve plot if matplotlib is installed
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(ep, [h["train"]["loss"] for h in history], label="Train Loss")
        ax1.plot(ep, [h["val"]["loss"] for h in history], label="Val Loss")
        ax1.set_title("Stage-1 ResNet-50 CNN Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(ep, [h["train"]["iou"] for h in history], label="Train Pair IoU")
        ax2.plot(ep, [h["val"]["iou"] for h in history], label="Val Pair IoU")
        ax2.set_title("Stage-1 ResNet-50 CNN Pair IoU")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(res_dir / "stage1_cnn_curves.png", dpi=150)
        plt.close(fig)
        print(f"Saved curves to {res_dir / 'stage1_cnn_curves.png'}")
    except Exception as e:
        print(f"Skipping curve plot: {e}")


if __name__ == "__main__":
    main()
