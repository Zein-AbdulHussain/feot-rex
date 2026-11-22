import torch
import torch.nn as nn
import torch.nn.functional as F


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def box_iou_and_giou(pred_cxcywh: torch.Tensor, target_cxcywh: torch.Tensor):
    """Compute element-wise IoU and GIoU between boxes in [cx, cy, w, h] format."""
    b1 = cxcywh_to_xyxy(pred_cxcywh)
    b2 = cxcywh_to_xyxy(target_cxcywh)

    x1 = torch.maximum(b1[..., 0], b2[..., 0])
    y1 = torch.maximum(b1[..., 1], b2[..., 1])
    x2 = torch.minimum(b1[..., 2], b2[..., 2])
    y2 = torch.minimum(b1[..., 3], b2[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    union = (area1 + area2 - inter).clamp(min=1e-6)
    iou = inter / union

    cx1 = torch.minimum(b1[..., 0], b2[..., 0])
    cy1 = torch.minimum(b1[..., 1], b2[..., 1])
    cx2 = torch.maximum(b1[..., 2], b2[..., 2])
    cy2 = torch.maximum(b1[..., 3], b2[..., 3])
    c_area = ((cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)).clamp(min=1e-6)

    giou = iou - (c_area - union) / c_area
    return iou, giou


def gaussian_focal_loss(pred_logits: torch.Tensor, target_hm: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CornerNet / CenterNet modified focal loss for continuous Gaussian heatmaps in [0, 1]."""
    pred = torch.sigmoid(pred_logits).clamp(1e-6, 1.0 - 1e-6)
    pos_mask = target_hm.ge(0.99).float()
    # Ensure at least the max pixel per sample counts as positive
    B = target_hm.shape[0]
    flat_idx = target_hm.view(B, -1).argmax(dim=1)
    peak_mask = torch.zeros_like(target_hm.view(B, -1))
    peak_mask.scatter_(1, flat_idx.unsqueeze(1), 1.0)
    pos_mask = torch.maximum(pos_mask, peak_mask.view_as(target_hm))
    neg_mask = 1.0 - pos_mask

    pos_loss = -torch.log(pred) * ((1.0 - pred) ** alpha) * pos_mask
    neg_weight = (1.0 - target_hm) ** beta
    neg_loss = -torch.log(1.0 - pred) * (pred ** alpha) * neg_weight * neg_mask

    num_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


class TrackerLoss(nn.Module):
    def __init__(self, w_giou: float = 2.0, w_l1: float = 5.0, w_hm: float = 1.0):
        super().__init__()
        self.w_giou = w_giou
        self.w_l1 = w_l1
        self.w_hm = w_hm

    def forward(
        self,
        pred_box: torch.Tensor,
        pred_hm_logits: torch.Tensor,
        target_box: torch.Tensor,
        target_hm: torch.Tensor,
    ):
        # Always compute losses in fp32 to avoid fp16 log(1 - sigmoid(x)) saturation -> NaN
        pred_box = pred_box.float()
        pred_hm_logits = pred_hm_logits.float()
        target_box = target_box.float()
        target_hm = target_hm.float()

        iou, giou = box_iou_and_giou(pred_box, target_box)
        loss_giou = (1.0 - giou).mean()
        loss_l1 = F.l1_loss(pred_box, target_box)
        loss_hm = gaussian_focal_loss(pred_hm_logits, target_hm)
        total = self.w_giou * loss_giou + self.w_l1 * loss_l1 + self.w_hm * loss_hm
        return total, {
            "loss": float(total.item()),
            "giou": float(loss_giou.item()),
            "l1": float(loss_l1.item()),
            "hm": float(loss_hm.item()),
            "iou": float(iou.mean().item()),
        }
