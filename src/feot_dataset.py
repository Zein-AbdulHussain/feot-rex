from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import math

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_sequence_meta(seq_dir: Path) -> Dict:
    """Load sorted frame/event paths, GT boxes [x, y, w, h], occlusion levels, and timestamps."""
    frame_dir = seq_dir / "frame"
    event_dir = seq_dir / "event"
    frames = sorted([p for p in frame_dir.iterdir() if p.is_file()])
    events = sorted([p for p in event_dir.iterdir() if p.is_file()])

    boxes = []
    occs = []
    cats = []
    for line in (seq_dir / "gt.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [float(v) for v in line.replace(",", " ").split()]
        cats.append(int(parts[0]))
        boxes.append(parts[2:6])  # x, y, w, h (top-left)
        occs.append(int(parts[6]))

    ts = [
        float(l.strip())
        for l in (seq_dir / "timestamp.txt").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    assert len(frames) == len(boxes) == len(ts), (
        f"Mismatch in {seq_dir.name}: frames={len(frames)}, gt={len(boxes)}, ts={len(ts)}"
    )

    return {
        "name": seq_dir.name,
        "frames": frames,
        "events": events,
        "boxes": np.asarray(boxes, dtype=np.float32),  # (N, 4) xywh
        "occs": np.asarray(occs, dtype=np.int32),      # (N,)
        "cats": np.asarray(cats, dtype=np.int32),      # (N,)
        "timestamps": np.asarray(ts, dtype=np.float64),
    }


def compute_crop_size(w: float, h: float, scale_factor: float, min_size: float, max_size: float = 1070.0) -> float:
    """SiamFC-style context-padded square crop side length."""
    pad = 0.5 * (w + h)
    s = math.sqrt((w + pad) * (h + pad)) * scale_factor
    return float(max(min_size, min(s, max_size)))


def extract_square_crop(
    img: Image.Image,
    cx: float,
    cy: float,
    crop_side: float,
    out_size: int,
    fill_rgb: Tuple[int, int, int] = (124, 116, 104),
) -> Tuple[Image.Image, Tuple[float, float, float]]:
    """
    Extract a square crop centered at (cx, cy) with side `crop_side`, padding with
    ImageNet-mean color if it extends outside image boundaries, and resize to `out_size`.

    Returns:
        crop_img: PIL.Image of size (out_size, out_size)
        crop_meta: (x1, y1, crop_side) in original image coordinates
    """
    W, H = img.size
    half = crop_side / 2.0
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    ix1, iy1 = int(math.floor(x1)), int(math.floor(y1))
    ix2, iy2 = int(math.ceil(x2)), int(math.ceil(y2))
    side_i = max(1, ix2 - ix1)

    # Fast path when fully inside image
    if ix1 >= 0 and iy1 >= 0 and ix2 <= W and iy2 <= H:
        crop = img.crop((ix1, iy1, ix2, iy2))
    else:
        canvas = Image.new("RGB", (side_i, side_i), fill_rgb)
        src_x1 = max(0, ix1)
        src_y1 = max(0, iy1)
        src_x2 = min(W, ix2)
        src_y2 = min(H, iy2)
        if src_x2 > src_x1 and src_y2 > src_y1:
            patch = img.crop((src_x1, src_y1, src_x2, src_y2))
            canvas.paste(patch, (src_x1 - ix1, src_y1 - iy1))
        crop = canvas

    crop = crop.resize((out_size, out_size), Image.BILINEAR)
    return crop, (float(ix1), float(iy1), float(side_i))


def frame_box_to_crop_norm(box_xywh: np.ndarray, crop_meta: Tuple[float, float, float]) -> np.ndarray:
    """Convert full-frame [x, y, w, h] to crop-normalized [cx, cy, w, h] in [0, 1]."""
    x, y, w, h = box_xywh
    x1, y1, side = crop_meta
    cx = (x + 0.5 * w - x1) / side
    cy = (y + 0.5 * h - y1) / side
    nw = w / side
    nh = h / side
    return np.array([cx, cy, nw, nh], dtype=np.float32)


def crop_norm_to_frame_box(
    box_cxcywh: np.ndarray,
    crop_meta: Tuple[float, float, float],
    img_w: float = 1070.0,
    img_h: float = 610.0,
) -> np.ndarray:
    """Convert crop-normalized [cx, cy, w, h] back to full-frame [x, y, w, h]."""
    cx, cy, nw, nh = box_cxcywh
    x1, y1, side = crop_meta
    w = max(4.0, float(nw) * side)
    h = max(4.0, float(nh) * side)
    fcx = x1 + float(cx) * side
    fcy = y1 + float(cy) * side
    x = max(0.0, min(img_w - 4.0, fcx - 0.5 * w))
    y = max(0.0, min(img_h - 4.0, fcy - 0.5 * h))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return np.array([x, y, w, h], dtype=np.float32)


def make_gaussian_heatmap(cx_norm: float, cy_norm: float, grid_size: int = 16, sigma: float = 1.2) -> torch.Tensor:
    """Create a 2D Gaussian target heatmap of shape (1, grid_size, grid_size) in [0, 1]."""
    xs = torch.arange(grid_size, dtype=torch.float32) + 0.5
    ys = torch.arange(grid_size, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gx = cx_norm * grid_size
    gy = cy_norm * grid_size
    dist2 = (xx - gx) ** 2 + (yy - gy) ** 2
    hm = torch.exp(-dist2 / (2.0 * sigma * sigma))
    return hm.unsqueeze(0)


def pil_to_norm_tensor(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)


class FEOTPairDataset(Dataset):
    """
    Stage-1 CNN dataset: samples (template, search_frame) pairs from FEOT RGB sequences.
    Uses realistic anchor-center jitter & scale jitter so the CNN learns true spatial
    localization rather than center-bias.
    """

    def __init__(
        self,
        data_root: Path,
        split_file: str,
        template_size: int = 128,
        search_size: int = 256,
        grid_size: int = 16,
        max_occ: int = 80,
        samples_per_epoch: Optional[int] = None,
        augment: bool = True,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.ds_dir = self.data_root / "FEOT_dataset"
        self.template_size = template_size
        self.search_size = search_size
        self.grid_size = grid_size
        self.augment = augment
        self.rng = random.Random(seed)

        seq_names = [
            l.strip()
            for l in (self.data_root / split_file).read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        self.sequences: List[Dict] = []
        self.valid_indices: List[Tuple[int, int]] = []  # (seq_idx, frame_idx)

        for name in seq_names:
            meta = load_sequence_meta(self.ds_dir / name)
            # Pick best clear template indices (low occlusion)
            clear_idxs = np.where(meta["occs"] <= 20)[0]
            if len(clear_idxs) == 0:
                clear_idxs = np.array([int(np.argmin(meta["occs"]))])
            meta["clear_idxs"] = clear_idxs

            # Valid search frames for Stage-1 appearance training
            search_idxs = np.where(meta["occs"] <= max_occ)[0]
            if len(search_idxs) == 0:
                search_idxs = np.arange(len(meta["frames"]))
            meta["search_idxs"] = search_idxs

            seq_idx = len(self.sequences)
            self.sequences.append(meta)
            for fi in search_idxs:
                self.valid_indices.append((seq_idx, int(fi)))

        # Balanced sampling per epoch (if specified) prevents 900-frame sequences from dominating 60-frame ones
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.valid_indices)

    def _get_pair_indices(self, idx: int) -> Tuple[int, int, int]:
        if self.samples_per_epoch is not None:
            # Uniform over sequences, then random valid frame within that sequence
            seq_idx = idx % len(self.sequences)
            meta = self.sequences[seq_idx]
            search_fi = int(random.choice(meta["search_idxs"]))
            # Prefer frame 0 / early clear frame 50% of time, any clear frame 50% of time
            if random.random() < 0.5:
                tmpl_fi = int(meta["clear_idxs"][0])
            else:
                tmpl_fi = int(random.choice(meta["clear_idxs"]))
            return seq_idx, tmpl_fi, search_fi
        else:
            seq_idx, search_fi = self.valid_indices[idx]
            meta = self.sequences[seq_idx]
            tmpl_fi = int(meta["clear_idxs"][0])
            return seq_idx, tmpl_fi, search_fi

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_idx, tmpl_fi, search_fi = self._get_pair_indices(idx)
        meta = self.sequences[seq_idx]

        tmpl_box = meta["boxes"][tmpl_fi]
        srch_box = meta["boxes"][search_fi]
        occ = int(meta["occs"][search_fi])

        tmpl_img = Image.open(meta["frames"][tmpl_fi]).convert("RGB")
        srch_img = Image.open(meta["frames"][search_fi]).convert("RGB")

        # Template crop around GT box
        tcx = tmpl_box[0] + 0.5 * tmpl_box[2]
        tcy = tmpl_box[1] + 0.5 * tmpl_box[3]
        t_side = compute_crop_size(tmpl_box[2], tmpl_box[3], scale_factor=2.0, min_size=64.0)
        if self.augment:
            t_side *= random.uniform(0.92, 1.08)
            tcx += random.uniform(-0.04, 0.04) * t_side
            tcy += random.uniform(-0.04, 0.04) * t_side
        tmpl_crop, _ = extract_square_crop(tmpl_img, tcx, tcy, t_side, self.template_size)

        # Search crop around jittered anchor (simulating previous-frame tracking offset)
        scx = srch_box[0] + 0.5 * srch_box[2]
        scy = srch_box[1] + 0.5 * srch_box[3]
        s_side = compute_crop_size(srch_box[2], srch_box[3], scale_factor=4.0, min_size=256.0)

        if self.augment:
            s_side *= random.uniform(0.85, 1.15)
            # Keep target center inside [0.15, 0.85] of search crop
            max_shift = 0.30 * s_side
            anchor_cx = scx + random.uniform(-max_shift, max_shift)
            anchor_cy = scy + random.uniform(-max_shift, max_shift)
        else:
            # Deterministic small offset for val so validation measures real localization
            prev_fi = max(0, search_fi - 1)
            prev_box = meta["boxes"][prev_fi]
            anchor_cx = prev_box[0] + 0.5 * prev_box[2]
            anchor_cy = prev_box[1] + 0.5 * prev_box[3]
            s_side = compute_crop_size(prev_box[2], prev_box[3], scale_factor=4.0, min_size=256.0)

        srch_crop, crop_meta = extract_square_crop(srch_img, anchor_cx, anchor_cy, s_side, self.search_size)
        box_norm = frame_box_to_crop_norm(srch_box, crop_meta)

        # Horizontal flip (consistent across template & search)
        if self.augment and random.random() < 0.5:
            tmpl_crop = TF.hflip(tmpl_crop)
            srch_crop = TF.hflip(srch_crop)
            box_norm[0] = 1.0 - box_norm[0]

        # Subtle photometric jitter
        if self.augment and random.random() < 0.5:
            b = random.uniform(0.85, 1.15)
            c = random.uniform(0.85, 1.15)
            s = random.uniform(0.85, 1.15)
            srch_crop = TF.adjust_brightness(srch_crop, b)
            srch_crop = TF.adjust_contrast(srch_crop, c)
            srch_crop = TF.adjust_saturation(srch_crop, s)

        box_norm[0] = float(np.clip(box_norm[0], 0.02, 0.98))
        box_norm[1] = float(np.clip(box_norm[1], 0.02, 0.98))
        box_norm[2] = float(np.clip(box_norm[2], 0.02, 0.98))
        box_norm[3] = float(np.clip(box_norm[3], 0.02, 0.98))

        heatmap = make_gaussian_heatmap(box_norm[0], box_norm[1], grid_size=self.grid_size, sigma=1.2)

        return {
            "template": pil_to_norm_tensor(tmpl_crop),
            "search": pil_to_norm_tensor(srch_crop),
            "box": torch.from_numpy(box_norm),
            "heatmap": heatmap,
            "occ": torch.tensor(occ, dtype=torch.int32),
        }
