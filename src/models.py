from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50TrackerCNN(nn.Module):
    """
    Stage-1 RGB CNN Tracker built on ImageNet-pretrained ResNet-50 (up to layer3, stride 16).

    Given a template crop z (B, 3, 128, 128) and a search crop x (B, 3, 256, 256):
      1. Extracts 256-ch spatial features Z (8x8) and X (16x16).
      2. Computes target-conditioned depthwise correlation + channel modulation -> F_fused (256, 16, 16).
      3. Predicts:
         - heatmap_logits (B, 1, 16, 16): spatial target center logits (also directly usable as 256-way
           spatial classification for ReX!).
         - box (B, 4): [cx, cy, w, h] normalized to [0, 1] inside the search crop via soft-argmax center +
           learned residual offset and size regression.
         - frame_feat (B, 512): compact visual-spatial descriptor ready to feed into the Stage-2 LSTM.
    """

    def __init__(self, pretrained: bool = True, feat_dim: int = 256, freeze_early: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # Keep conv1..layer3 (stride 16 -> 1024 channels)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 256 ch, stride 4
        self.layer2 = backbone.layer2  # 512 ch, stride 8
        self.layer3 = backbone.layer3  # 1024 ch, stride 16

        if freeze_early:
            for m in (self.stem, self.layer1, self.layer2):
                for p in m.parameters():
                    p.requires_grad = False

        self.feat_dim = feat_dim
        self.neck = nn.Sequential(
            nn.Conv2d(1024, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        # Template pooling for 3x3 depthwise correlation kernel & global channel gate
        self.tmpl_kernel_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.tmpl_gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

        # Fusion of [depthwise_corr (256), gated_search (256)] -> 256
        self.fusion = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        # Spatial center heatmap head -> (B, 1, 16, 16)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )
        # Initialize heatmap bias for sparse positive peak stability
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

        # Box regression head from target-attended 256-d vector -> [dx, dy, w_logit, h_logit]
        self.box_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )

        # Precompute 16x16 normalized grid centers in (0, 1)
        grid = (torch.arange(16, dtype=torch.float32) + 0.5) / 16.0
        gy, gx = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("grid_x", gx.reshape(1, 256))
        self.register_buffer("grid_y", gy.reshape(1, 256))

    def _extract_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.neck(x)

    def encode_template(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode template once per sequence: returns (kernel_3x3, channel_gate)."""
        fz = self._extract_backbone(z)  # (B, 256, 8, 8)
        # Center 4x4 of the 8x8 template holds the target (since scale_factor=2.0)
        fz_center = fz[:, :, 2:6, 2:6]
        kernel = self.tmpl_kernel_pool(fz_center)  # (B, 256, 3, 3)
        z_gap = fz_center.mean(dim=(2, 3))         # (B, 256)
        gate = self.tmpl_gate(z_gap).unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)
        return kernel, gate

    def _depthwise_xcorr(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Batched depthwise cross-correlation of x (B, C, H, W) with kernel (B, C, 3, 3)."""
        B, C, H, W = x.shape
        x_flat = x.reshape(1, B * C, H, W)
        k_flat = kernel.reshape(B * C, 1, 3, 3)
        # Normalize kernel energy for stable correlation magnitude
        k_norm = k_flat / (k_flat.norm(dim=(2, 3), keepdim=True).clamp(min=1e-5))
        out = F.conv2d(x_flat, k_norm, padding=1, groups=B * C)
        return out.reshape(B, C, H, W)

    def forward_with_template(
        self,
        tmpl_encoded: Tuple[torch.Tensor, torch.Tensor],
        search: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        kernel, gate = tmpl_encoded
        fx = self._extract_backbone(search)  # (B, 256, 16, 16)
        corr = self._depthwise_xcorr(fx, kernel)
        gated = fx * gate
        fused = self.fusion(torch.cat([corr, gated], dim=1))  # (B, 256, 16, 16)

        hm_logits = self.heatmap_head(fused)  # (B, 1, 16, 16)
        B = search.shape[0]
        hm_flat = hm_logits.view(B, 256)
        attn = F.softmax(hm_flat, dim=-1)  # (B, 256)

        # Soft-argmax center prediction
        cx_base = (attn * self.grid_x).sum(dim=-1)
        cy_base = (attn * self.grid_y).sum(dim=-1)

        # Target-attended feature vector (B, 256)
        fused_flat = fused.view(B, self.feat_dim, 256)
        v_att = torch.bmm(fused_flat, attn.unsqueeze(-1)).squeeze(-1)

        box_raw = self.box_head(v_att)
        # Small residual refinement on soft-argmax center (±1 grid cell = ±0.0625)
        cx = (cx_base + 0.0625 * torch.tanh(box_raw[:, 0])).clamp(0.01, 0.99)
        cy = (cy_base + 0.0625 * torch.tanh(box_raw[:, 1])).clamp(0.01, 0.99)
        w = torch.sigmoid(box_raw[:, 2]).clamp(0.02, 0.99)
        h = torch.sigmoid(box_raw[:, 3]).clamp(0.02, 0.99)
        box = torch.stack([cx, cy, w, h], dim=-1)

        # 512-d frame descriptor for Stage-2 LSTM: [attended visual features (256), spatial response (256)]
        frame_feat = torch.cat([v_att, torch.sigmoid(hm_flat)], dim=-1)

        return {
            "box": box,
            "heatmap_logits": hm_logits,
            "frame_feat": frame_feat,
            "v_att": v_att,
            "confidence": torch.sigmoid(hm_flat).max(dim=-1).values,
        }

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> Dict[str, torch.Tensor]:
        tmpl_encoded = self.encode_template(template)
        return self.forward_with_template(tmpl_encoded, search)


def compute_cnn_summary_stats(hm_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute 3 explicit uncertainty/occlusion features from (B, 1, 16, 16) heatmap logits:
      - max_prob: peak sigmoid confidence
      - mean_prob: average sigmoid activation
      - norm_entropy: normalized spatial entropy of softmax distribution in [0, 1]
    """
    B = hm_logits.shape[0]
    flat = hm_logits.float().view(B, 256)
    sig = torch.sigmoid(flat)
    max_p = sig.max(dim=-1).values
    mean_p = sig.mean(dim=-1)
    probs = F.softmax(flat, dim=-1).clamp(min=1e-7)
    entropy = -(probs * probs.log()).sum(dim=-1) / 5.54517744  # log(256)
    return torch.stack([max_p, mean_p, entropy], dim=-1)


class TemporalLSTMHead(nn.Module):
    """
    Stage-2 Temporal LSTM module attached on top of ResNet50TrackerCNN.

    Per timestep input (524-d):
      - frame_feat (512): [v_att (256), sigmoid(heatmap) (256)] from ResNet-50 CNN
      - cnn_box (4): CNN predicted [cx, cy, w, h] in search-crop coords
      - cnn_stats (3): [max_prob, mean_prob, norm_entropy]
      - prev_vel_crop (2): previous velocity (vx, vy) / crop_side
      - prev_size_crop (2): previous box (w, h) / crop_side
      - dt_norm (1): normalized frame interval (dt_us / 40000.0)

    Outputs gated fusion of visual refinement and motion-propagated prediction so the tracker
    relies on ResNet-50 when visible and coasts smoothly through occlusions via LSTM memory.
    """

    def __init__(self, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        in_dim = 512 + 4 + 3 + 2 + 2 + 1  # 524
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_drop = nn.Dropout(dropout)

        # Visibility / trust gate logit (1 = target clearly visible, 0 = occluded)
        self.vis_head = nn.Linear(hidden_size, 1)
        # Visual box residual refinement when target is visible
        self.vis_delta_head = nn.Linear(hidden_size, 4)
        # Motion-propagated box residual around constant-velocity prior when occluded
        self.mot_delta_head = nn.Linear(hidden_size, 4)
        # Spatial 16x16 heatmap residual refinement
        self.hm_refine_head = nn.Linear(hidden_size, 256)

        # Start near identity so early training inherits Stage-1 accuracy cleanly
        nn.init.zeros_(self.vis_delta_head.weight)
        nn.init.zeros_(self.vis_delta_head.bias)
        nn.init.zeros_(self.mot_delta_head.weight)
        nn.init.zeros_(self.mot_delta_head.bias)
        nn.init.zeros_(self.hm_refine_head.weight)
        nn.init.zeros_(self.hm_refine_head.bias)

    def forward(
        self,
        frame_feat: torch.Tensor,      # (B, T, 512)
        cnn_box: torch.Tensor,         # (B, T, 4)
        cnn_hm_logits: torch.Tensor,   # (B, T, 1, 16, 16)
        cnn_stats: torch.Tensor,       # (B, T, 3)
        prev_vel_crop: torch.Tensor,   # (B, T, 2)
        prev_size_crop: torch.Tensor,  # (B, T, 2)
        dt_norm: torch.Tensor,         # (B, T, 1)
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x = torch.cat(
            [frame_feat, cnn_box, cnn_stats, prev_vel_crop, prev_size_crop, dt_norm],
            dim=-1,
        )
        x = self.input_proj(x)
        h_seq, next_hidden = self.lstm(x, hidden)
        h_seq = self.out_drop(self.out_norm(h_seq))

        vis_logit = self.vis_head(h_seq)  # (B, T, 1)
        vis_prob = torch.sigmoid(vis_logit)

        # Visual-refined box around Stage-1 CNN prediction
        dv = self.vis_delta_head(h_seq)
        vis_cx = (cnn_box[..., 0] + 0.08 * torch.tanh(dv[..., 0])).clamp(0.01, 0.99)
        vis_cy = (cnn_box[..., 1] + 0.08 * torch.tanh(dv[..., 1])).clamp(0.01, 0.99)
        vis_w = (cnn_box[..., 2] * torch.exp(0.15 * torch.tanh(dv[..., 2]))).clamp(0.02, 0.99)
        vis_h = (cnn_box[..., 3] * torch.exp(0.15 * torch.tanh(dv[..., 3]))).clamp(0.02, 0.99)
        box_vis = torch.stack([vis_cx, vis_cy, vis_w, vis_h], dim=-1)

        # Motion-propagated box around constant-velocity prior (crop center = 0.5 + prev_vel * dt)
        dm = self.mot_delta_head(h_seq)
        dt_clamped = dt_norm[..., 0].clamp(0.25, 3.0)
        prior_cx = 0.5 + (prev_vel_crop[..., 0] * dt_clamped).clamp(-0.4, 0.4)
        prior_cy = 0.5 + (prev_vel_crop[..., 1] * dt_clamped).clamp(-0.4, 0.4)
        mot_cx = (prior_cx + 0.15 * torch.tanh(dm[..., 0])).clamp(0.01, 0.99)
        mot_cy = (prior_cy + 0.15 * torch.tanh(dm[..., 1])).clamp(0.01, 0.99)
        mot_w = (prev_size_crop[..., 0].clamp(0.02, 0.99) * torch.exp(0.12 * torch.tanh(dm[..., 2]))).clamp(0.02, 0.99)
        mot_h = (prev_size_crop[..., 1].clamp(0.02, 0.99) * torch.exp(0.12 * torch.tanh(dm[..., 3]))).clamp(0.02, 0.99)
        box_mot = torch.stack([mot_cx, mot_cy, mot_w, mot_h], dim=-1)

        # Gated fusion: trusts ResNet-50 when visible, coasts on LSTM motion when occluded
        box_pred = vis_prob * box_vis + (1.0 - vis_prob) * box_mot

        # Refined 16x16 spatial heatmap logits
        B, T = frame_feat.shape[:2]
        hm_delta = self.hm_refine_head(h_seq).view(B, T, 1, 16, 16)
        hm_refined = cnn_hm_logits + 0.5 * hm_delta

        return {
            "box": box_pred,
            "box_vis": box_vis,
            "box_mot": box_mot,
            "heatmap_logits": hm_refined,
            "vis_logit": vis_logit.squeeze(-1),
            "vis_prob": vis_prob.squeeze(-1),
            "hidden": next_hidden,
        }


class ResNet50LSTMTracker(nn.Module):
    """
    Unified End-to-End Stage-2 Model: Frozen/Trained ResNet-50 CNN + Temporal LSTM Head.
    Exposes `step(...)` for online tracking and Stage-3 ReX explainability with cached hidden state.
    """

    def __init__(self, cnn: ResNet50TrackerCNN, lstm_head: TemporalLSTMHead, freeze_cnn: bool = True):
        super().__init__()
        self.cnn = cnn
        self.lstm_head = lstm_head
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

    def encode_template(self, z: torch.Tensor):
        return self.cnn.encode_template(z)

    def step(
        self,
        tmpl_encoded: Tuple[torch.Tensor, torch.Tensor],
        search_rgb: torch.Tensor,          # (B, 3, 256, 256)
        prev_vel_crop: torch.Tensor,       # (B, 2)
        prev_size_crop: torch.Tensor,      # (B, 2)
        dt_norm: torch.Tensor,             # (B, 1)
        hidden: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Single-frame end-to-end step (ideal for online tracking and ReX mutant evaluation)."""
        cnn_out = self.cnn.forward_with_template(tmpl_encoded, search_rgb)
        stats = compute_cnn_summary_stats(cnn_out["heatmap_logits"])
        out = self.lstm_head(
            frame_feat=cnn_out["frame_feat"].unsqueeze(1),
            cnn_box=cnn_out["box"].unsqueeze(1),
            cnn_hm_logits=cnn_out["heatmap_logits"].unsqueeze(1),
            cnn_stats=stats.unsqueeze(1),
            prev_vel_crop=prev_vel_crop.unsqueeze(1),
            prev_size_crop=prev_size_crop.unsqueeze(1),
            dt_norm=dt_norm.unsqueeze(1),
            hidden=hidden,
        )
        return {
            "box": out["box"][:, 0],
            "box_cnn": cnn_out["box"],
            "box_mot": out["box_mot"][:, 0],
            "heatmap_logits": out["heatmap_logits"][:, 0],
            "cnn_heatmap_logits": cnn_out["heatmap_logits"],
            "vis_prob": out["vis_prob"][:, 0],
            "hidden": out["hidden"],
        }

