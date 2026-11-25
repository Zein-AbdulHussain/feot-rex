import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_PROJ_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_PROJ_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_HOME", str(_PROJ_ROOT / ".cache" / "torch"))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision.models import ResNet50_Weights, resnet50

from feot_dataset import IMAGENET_MEAN, IMAGENET_STD, load_sequence_meta


def crop_gt_box(img: Image.Image, box_xywh: np.ndarray, out_size: int = 224) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Crop the exact annotated GT bounding box [x, y, w, h] clamped to image bounds,
    and resize to (out_size, out_size).
    Returns:
      - crop_pil: resized RGB PIL image
      - (x1, y1, x2, y2): exact integer pixel crop bounds in original 1070x610 frame
    """
    W, H = img.size
    x, y, w, h = box_xywh
    x1 = max(0, min(W - 2, int(round(x))))
    y1 = max(0, min(H - 2, int(round(y))))
    x2 = max(x1 + 2, min(W, int(round(x + w))))
    y2 = max(y1 + 2, min(H, int(round(y + h))))
    crop = img.crop((x1, y1, x2, y2)).resize((out_size, out_size), Image.BILINEAR)
    return crop, (x1, y1, x2, y2)


class FEOTClassifierDataset(Dataset):
    """Frame-level GT-box RGB crop dataset for category classification (sequence-disjoint)."""

    def __init__(
        self,
        data_root: Path,
        split_file: str,
        cat_to_idx: Dict[int, int],
        exclude_full_occ: bool = True,
        img_size: int = 224,
        samples_per_epoch: int = 0,
        augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.ds_dir = self.data_root / "FEOT_dataset"
        self.cat_to_idx = cat_to_idx
        self.img_size = img_size
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch

        seq_names = [
            l.strip()
            for l in (self.data_root / split_file).read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]

        self.items: List[Dict] = []
        self.by_class: Dict[int, List[int]] = defaultdict(list)

        for name in seq_names:
            meta = load_sequence_meta(self.ds_dir / name)
            for fi in range(len(meta["frames"])):
                occ = int(meta["occs"][fi])
                if exclude_full_occ and occ >= 100:
                    continue
                cat_id = int(meta["cats"][fi])
                if cat_id not in cat_to_idx:
                    continue
                cls_idx = cat_to_idx[cat_id]
                item_idx = len(self.items)
                self.items.append({
                    "seq": name,
                    "frame_idx": fi,
                    "frame_path": meta["frames"][fi],
                    "box": meta["boxes"][fi],
                    "occ": occ,
                    "cat_id": cat_id,
                    "label": cls_idx,
                })
                self.by_class[cls_idx].append(item_idx)

        self.active_classes = sorted(self.by_class.keys())

    def __len__(self) -> int:
        if self.samples_per_epoch > 0:
            return self.samples_per_epoch
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        if self.samples_per_epoch > 0:
            # Class-balanced sampling so rare categories are learned cleanly
            cls_idx = self.active_classes[idx % len(self.active_classes)]
            item = self.items[random.choice(self.by_class[cls_idx])]
        else:
            item = self.items[idx]

        with Image.open(item["frame_path"]) as im:
            img = im.convert("RGB")

        box = item["box"].copy()
        if self.augment:
            # Slight box jitter (±6%) so classifier isn't brittle to tight annotation edges
            jx = random.uniform(-0.06, 0.06) * box[2]
            jy = random.uniform(-0.06, 0.06) * box[3]
            jw = random.uniform(0.92, 1.08) * box[2]
            jh = random.uniform(0.92, 1.08) * box[3]
            box = np.array([box[0] + jx, box[1] + jy, jw, jh], dtype=np.float32)

        crop, _ = crop_gt_box(img, box, out_size=self.img_size)

        if self.augment:
            if random.random() < 0.5:
                crop = TF.hflip(crop)
            if random.random() < 0.5:
                crop = TF.adjust_brightness(crop, random.uniform(0.85, 1.15))
                crop = TF.adjust_contrast(crop, random.uniform(0.85, 1.15))
                crop = TF.adjust_saturation(crop, random.uniform(0.85, 1.15))

        t = TF.normalize(TF.to_tensor(crop), IMAGENET_MEAN, IMAGENET_STD)
        return {
            "image": t,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "occ": torch.tensor(item["occ"], dtype=torch.int32),
            "seq": item["seq"],
            "frame_idx": item["frame_idx"],
        }


def build_resnet50_classifier(num_classes: int, pretrained: bool = True, freeze_early: bool = True) -> nn.Module:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    if freeze_early:
        for m in (model.conv1, model.bn1, model.layer1, model.layer2):
            for p in m.parameters():
                p.requires_grad = False
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, num_classes),
    )
    return model


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_to_cat: Dict[int, int],
) -> Dict:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    correct_all = 0
    correct_occ0 = 0
    n_occ0 = 0

    # Group frame probabilities by sequence for sequence-level prediction (average of frame probs)
    seq_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    seq_probs_occ0: Dict[str, List[np.ndarray]] = defaultdict(list)
    seq_label: Dict[str, int] = {}
    seq_frame_correct: Dict[str, List[int]] = defaultdict(list)
    seq_occ0_correct: Dict[str, List[int]] = defaultdict(list)

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        occs = batch["occ"].numpy()
        seqs = batch["seq"]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
        logits_f = logits.float()
        loss = F.cross_entropy(logits_f, y)
        probs = F.softmax(logits_f, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)
        y_np = y.cpu().numpy()

        B = len(y_np)
        total_loss += float(loss.item()) * B
        n_samples += B
        correct_all += int((preds == y_np).sum())

        for i in range(B):
            s = seqs[i]
            seq_label[s] = int(y_np[i])
            seq_probs[s].append(probs[i])
            is_ok = int(preds[i] == y_np[i])
            seq_frame_correct[s].append(is_ok)
            if int(occs[i]) == 0:
                n_occ0 += 1
                correct_occ0 += is_ok
                seq_probs_occ0[s].append(probs[i])
                seq_occ0_correct[s].append(is_ok)

    # Sequence-level prediction = average of its frame-level predicted probability vectors
    seq_results = []
    seq_correct = 0
    for s in sorted(seq_probs.keys()):
        mean_prob = np.mean(seq_probs[s], axis=0)
        seq_pred = int(mean_prob.argmax())
        gt = seq_label[s]
        ok = int(seq_pred == gt)
        seq_correct += ok
        occ0_list = seq_occ0_correct.get(s, [])
        seq_results.append({
            "seq": s,
            "gt_cat": idx_to_cat[gt],
            "pred_cat": idx_to_cat[seq_pred],
            "seq_correct": bool(ok),
            "seq_conf": float(mean_prob[seq_pred]),
            "n_frames_eval": len(seq_frame_correct[s]),
            "frame_acc": float(np.mean(seq_frame_correct[s])),
            "n_occ0": len(occ0_list),
            "occ0_acc": float(np.mean(occ0_list)) if occ0_list else None,
        })

    return {
        "loss": total_loss / max(1, n_samples),
        "n_frames": n_samples,
        "frame_acc": correct_all / max(1, n_samples),
        "n_occ0": n_occ0,
        "occ0_frame_acc": correct_occ0 / max(1, n_occ0),
        "n_sequences": len(seq_results),
        "seq_acc": seq_correct / max(1, len(seq_results)),
        "per_sequence": seq_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 1: Train resnet-50 GT-Box Category Classifier on FEOT_100")
    parser.add_argument("--data_root", type=str, default=str(_PROJ_ROOT / "data" / "FEOT_100"))
    parser.add_argument("--out_dir", type=str, default=str(_PROJ_ROOT))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--samples_per_epoch", type=int, default=4000)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
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
    res_dir = out_dir / "outputs" / "rex_mps"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Discover all categories in train_list.txt (so train->val/test mapping is deterministic)
    ds_dir = data_root / "FEOT_dataset"
    train_seqs = [l.strip() for l in (data_root / "train_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    cats_found = set()
    for s in train_seqs:
        m = load_sequence_meta(ds_dir / s)
        cats_found.update(int(c) for c in np.unique(m["cats"]))
    cat_ids = sorted(cats_found)
    cat_to_idx = {c: i for i, c in enumerate(cat_ids)}
    idx_to_cat = {i: c for i, c in enumerate(cat_ids)}
    print(f"Categories ({len(cat_ids)}): {cat_to_idx}", flush=True)

    # Exclude fully occluded (occ == 100) frames from training!
    train_ds = FEOTClassifierDataset(
        data_root, "train_list.txt", cat_to_idx,
        exclude_full_occ=True, samples_per_epoch=args.samples_per_epoch, augment=True,
    )
    val_ds = FEOTClassifierDataset(
        data_root, "val_list.txt", cat_to_idx,
        exclude_full_occ=True, samples_per_epoch=0, augment=False,
    )
    test_ds = FEOTClassifierDataset(
        data_root, "test_list.txt", cat_to_idx,
        exclude_full_occ=True, samples_per_epoch=0, augment=False,
    )
    print(
        f"Train: {len(train_ds.items)} non-full-occ frames ({args.samples_per_epoch} class-balanced/epoch) | "
        f"Val: {len(val_ds.items)} frames | Test: {len(test_ds.items)} frames",
        flush=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), drop_last=True, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0))

    model = build_resnet50_classifier(num_classes=len(cat_ids), pretrained=True, freeze_early=True).to(device)
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("fc")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=1e-3,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_score = -1.0
    best_ckpt = ckpt_dir / "resnet50_classifier_best.pt"
    history = []

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        model.conv1.eval(); model.bn1.eval(); model.layer1.eval(); model.layer2.eval()

        tr_loss = 0.0
        tr_correct = 0
        tr_n = 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=0.05)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            B = x.shape[0]
            tr_loss += float(loss.item()) * B
            tr_correct += int((logits.argmax(dim=-1) == y).sum().item())
            tr_n += B

        scheduler.step()
        tr_avg_loss = tr_loss / max(1, tr_n)
        tr_acc = tr_correct / max(1, tr_n)

        val_res = evaluate_split(model, val_loader, device, idx_to_cat)
        # Select on combined Val frame accuracy + sequence accuracy
        sel_score = 0.5 * val_res["frame_acc"] + 0.5 * val_res["seq_acc"]
        tag = ""
        if sel_score > best_score:
            best_score = sel_score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "cat_ids": cat_ids,
                    "cat_to_idx": cat_to_idx,
                    "idx_to_cat": idx_to_cat,
                    "val_metrics": {k: v for k, v in val_res.items() if k != "per_sequence"},
                    "args": vars(args),
                },
                best_ckpt,
            )
            tag = " *BEST*"

        history.append({
            "epoch": epoch,
            "train_loss": tr_avg_loss,
            "train_acc": tr_acc,
            "val_loss": val_res["loss"],
            "val_frame_acc": val_res["frame_acc"],
            "val_occ0_acc": val_res["occ0_frame_acc"],
            "val_seq_acc": val_res["seq_acc"],
        })
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({time.time()-t0:4.1f}s) | "
            f"Tr loss={tr_avg_loss:.3f} acc={100*tr_acc:5.1f}% | "
            f"Va loss={val_res['loss']:.3f} frame={100*val_res['frame_acc']:5.1f}% "
            f"occ0={100*val_res['occ0_frame_acc']:5.1f}% seq={100*val_res['seq_acc']:5.1f}%{tag}",
            flush=True,
        )

    print(f"\nClassifier training finished in {(time.time()-t_start)/60:.1f} min.", flush=True)

    # Evaluate best checkpoint on Val and Test
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_eval = evaluate_split(model, val_loader, device, idx_to_cat)
    test_eval = evaluate_split(model, test_loader, device, idx_to_cat)

    summary = {
        "best_epoch": ckpt["epoch"],
        "cat_ids": cat_ids,
        "history": history,
        "val_eval": val_eval,
        "test_eval": test_eval,
    }
    (res_dir / "classifier_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for lbl, ev in [("VAL (10 seqs)", val_eval), ("TEST (20 seqs)", test_eval)]:
        print(f"\n=== resnet-50 GT-Crop Classifier — {lbl} ===", flush=True)
        print(f"  Frame Acc (occ<100): {100*ev['frame_acc']:.1f}% ({ev['n_frames']} frames)", flush=True)
        print(f"  Frame Acc (occ==0) : {100*ev['occ0_frame_acc']:.1f}% ({ev['n_occ0']} frames)", flush=True)
        print(f"  Seq Acc (avg prob) : {100*ev['seq_acc']:.1f}% ({ev['n_sequences']} seqs)", flush=True)
        if lbl.startswith("TEST"):
            for r in ev["per_sequence"]:
                o0 = f"{100*r['occ0_acc']:5.1f}%" if r["occ0_acc"] is not None else "  n/a"
                mark = "OK" if r["seq_correct"] else "XX"
                print(f"    [{mark}] {r['seq']}: gt={r['gt_cat']:2d} pred={r['pred_cat']:2d} (conf={r['seq_conf']:.2f}) | frame={100*r['frame_acc']:5.1f}% | occ0={o0} (n={r['n_occ0']:3d})", flush=True)

    # Save curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ep, [h["train_loss"] for h in history], label="Train Loss")
    ax1.plot(ep, [h["val_loss"] for h in history], label="Val Loss")
    ax1.set_title("resnet-50 Category Classifier Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(ep, [100*h["train_acc"] for h in history], label="Train Frame Acc")
    ax2.plot(ep, [100*h["val_frame_acc"] for h in history], label="Val Frame Acc")
    ax2.plot(ep, [100*h["val_seq_acc"] for h in history], "--", label="Val Seq Acc (Avg)")
    ax2.set_title("resnet-50 Category Classifier Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(res_dir / "classifier_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved curves to {res_dir / 'classifier_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
