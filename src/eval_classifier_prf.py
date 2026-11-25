import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

_PROJ_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TORCH_HOME", str(_PROJ_ROOT / ".cache" / "torch"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_classifier import FEOTClassifierDataset, build_resnet50_classifier


def compute_prf(y_true, y_pred, idx_to_cat):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(set(y_true.tolist()))
    per_cls = {}
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        sup = int((y_true == c).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_cls[str(idx_to_cat[c])] = {
            "support": sup,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    total_sup = sum(v["support"] for v in per_cls.values())
    macro = {
        "precision": round(float(np.mean([v["precision"] for v in per_cls.values()])), 4),
        "recall": round(float(np.mean([v["recall"] for v in per_cls.values()])), 4),
        "f1": round(float(np.mean([v["f1"] for v in per_cls.values()])), 4),
    }
    weighted = {
        "precision": round(sum(v["precision"] * v["support"] for v in per_cls.values()) / max(1, total_sup), 4),
        "recall": round(sum(v["recall"] * v["support"] for v in per_cls.values()) / max(1, total_sup), 4),
        "f1": round(sum(v["f1"] * v["support"] for v in per_cls.values()) / max(1, total_sup), 4),
    }
    return {
        "support": total_sup,
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro": macro,
        "weighted": weighted,
        "per_class": per_cls,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute precision, recall, and F1 for the resnet-50 classifier")
    parser.add_argument("--data_root", type=str, default=str(_PROJ_ROOT / "data" / "FEOT_100"))
    parser.add_argument("--ckpt", type=str, default=str(_PROJ_ROOT / "checkpoints" / "resnet50_classifier_best.pt"))
    parser.add_argument("--out", type=str, default=str(_PROJ_ROOT / "outputs" / "rex_mps" / "classifier_prf_metrics.json"))
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cat_ids = ckpt["cat_ids"]
    cat_to_idx = ckpt["cat_to_idx"]
    idx_to_cat = ckpt["idx_to_cat"]

    model = build_resnet50_classifier(num_classes=len(cat_ids), pretrained=False, freeze_early=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = {}
    for split_name, split_file in [("val", "val_list.txt"), ("test", "test_list.txt")]:
        ds = FEOTClassifierDataset(
            Path(args.data_root), split_file, cat_to_idx, exclude_full_occ=True, samples_per_epoch=0, augment=False
        )
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        yt_all, yp_all, yt_0, yp_0 = [], [], [], []
        seq_probs = defaultdict(list)
        seq_gt = {}
        with torch.no_grad():
            for b in dl:
                x = b["image"].to(device)
                y = b["label"].numpy()
                occs = b["occ"].numpy()
                seqs = b["seq"]
                probs = F.softmax(model(x).float(), dim=-1).cpu().numpy()
                preds = probs.argmax(axis=-1)
                for i in range(len(y)):
                    yt_all.append(int(y[i]))
                    yp_all.append(int(preds[i]))
                    seq_probs[seqs[i]].append(probs[i])
                    seq_gt[seqs[i]] = int(y[i])
                    if int(occs[i]) == 0:
                        yt_0.append(int(y[i]))
                        yp_0.append(int(preds[i]))

        yt_seq = [seq_gt[s] for s in sorted(seq_gt)]
        yp_seq = [int(np.mean(seq_probs[s], axis=0).argmax()) for s in sorted(seq_gt)]

        results[split_name] = {
            "frame_occ_lt_100": compute_prf(yt_all, yp_all, idx_to_cat),
            "frame_occ_eq_0": compute_prf(yt_0, yp_0, idx_to_cat),
            "sequence_avg_prob": compute_prf(yt_seq, yp_seq, idx_to_cat),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved precision/recall/F1 metrics to {out_path}", flush=True)


if __name__ == "__main__":
    main()
