"""
DeepGuard AI V2 - Neural Network Modules
==========================================
Reusable PyTorch nn.Module classes for the V2 architectural upgrade.

Phases covered:
  5A: TanhBoundedCosineClassifier
  1A: MultiScaleFFT
  1B: FreqNetDCT
  1C: HaarWaveletResidual, SPSLEncoder
  2A: DualSSLAudioBackbone
  2B: AMFF, NeXtTDNN, ECAModule
  2C: AASIST2GraphAttention
  3A: FuseMoE
  3B: GAMEDVeto
  3C: CASTCrossAttention
  4A: AVHuBERTLipSync
  4B: RPPGModule
  5B: adversarial training helpers
  5C: feature squeezing
  5D: EWC continual learning
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Phase 5A: Tanh-Bounded Cosine Classifier
# ======================================================================

class TanhBoundedCosineClassifier(nn.Module):
    """
    score = (tanh(k * (cos(f, W_real) - cos(f, W_fake))) + 1) / 2
    Parameters per instance: 2 * feature_dim + 1 (sharpness k)
    """

    def __init__(self, feature_dim: int, k_init: float = 2.0):
        super().__init__()
        self.W_real = nn.Parameter(torch.randn(feature_dim))
        self.W_fake = nn.Parameter(torch.randn(feature_dim))
        self.k = nn.Parameter(torch.tensor(k_init))
        nn.init.xavier_uniform_(self.W_real.unsqueeze(0))
        nn.init.xavier_uniform_(self.W_fake.unsqueeze(0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        f_norm = F.normalize(features, p=2, dim=-1)
        w_real_norm = F.normalize(self.W_real, p=2, dim=0)
        w_fake_norm = F.normalize(self.W_fake, p=2, dim=0)
        cos_real = torch.matmul(f_norm, w_real_norm)
        cos_fake = torch.matmul(f_norm, w_fake_norm)
        raw = cos_real - cos_fake
        k_clamped = torch.clamp(self.k, min=0.5, max=10.0)
        score = (torch.tanh(k_clamped * raw) + 1.0) / 2.0
        return score


# ======================================================================
# Phase 1A: Multi-Scale FFT with Channel Attention
# ======================================================================

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class MultiScaleFFT(nn.Module):
    """MSCA-FFT: Multi-Scale Channel Attention on FFT features. ~280K params."""
    def __init__(self, out_features: int = 32):
        super().__init__()
        self.se1 = SEBlock(2, 4)
        self.se2 = SEBlock(2, 4)
        self.se3 = SEBlock(2, 4)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(6, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, out_features)

    @staticmethod
    def _fft_mag_phase(x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(x, norm="ortho")
        mag = torch.log1p(torch.abs(fft))
        phase = torch.angle(fft)
        return torch.cat([mag, phase], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        s1 = F.avg_pool2d(gray, 3, 3, 1)
        s1_fft = self.se1(self._fft_mag_phase(s1))
        s1_fft = F.adaptive_avg_pool2d(s1_fft, 32)
        s2 = F.avg_pool2d(gray, 7, 7, 3)
        s2_fft = self.se2(self._fft_mag_phase(s2))
        s2_fft = F.adaptive_avg_pool2d(s2_fft, 32)
        s3 = F.avg_pool2d(gray, 15, 15, 7)
        s3_fft = self.se3(self._fft_mag_phase(s3))
        s3_fft = F.adaptive_avg_pool2d(s3_fft, 32)
        fused = torch.cat([s1_fft, s2_fft, s3_fft], dim=1)
        fused = self.fusion_conv(fused)
        out = self.global_pool(fused).flatten(1)
        return self.fc(out)


# ======================================================================
# Phase 1B: FreqNet DCT Frequency-Domain Plugin
# ======================================================================

class FreqNetDCT(nn.Module):
    """DCT-based detection head. ~1.9M params."""
    def __init__(self, block_size: int = 8, out_features: int = 8):
        super().__init__()
        self.block_size = block_size
        dct_dim = 3 * block_size * block_size  # 3 channels
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(dct_dim, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Linear(32, out_features))

    def _block_dct(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        bs = self.block_size
        pad_h = (bs - H % bs) % bs
        pad_w = (bs - W % bs) % bs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H_p, W_p = x.shape
        x = x.reshape(B * C, 1, H_p // bs, bs, W_p // bs, bs)
        x = x.permute(0, 1, 2, 4, 3, 5)
        N = bs
        n = torch.arange(N, dtype=x.dtype, device=x.device)
        k = n.unsqueeze(1)
        dct_matrix = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0] *= 1 / math.sqrt(2)
        dct_matrix *= math.sqrt(2.0 / N)
        x = torch.matmul(dct_matrix, x)
        x = torch.matmul(x, dct_matrix.transpose(-1, -2))
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(B * C, bs * bs, H_p // bs, W_p // bs)
        x = x.reshape(B, C * bs * bs, H_p // bs, W_p // bs)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dct_coeffs = self._block_dct(x)
        feat = self.conv_blocks(dct_coeffs)
        feat = self.global_pool(feat).flatten(1)
        return self.fc(feat)


# ======================================================================
# Phase 1C: Haar Wavelet Residual & SPSL Autoencoder
# ======================================================================

class HaarWaveletResidual(nn.Module):
    """Computes wavelet high-frequency residual energy per level. Returns (B, 3)."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if x.shape[1] == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        elif x.shape[1] == 1:
            gray = x
        else:
            gray = x.mean(dim=1, keepdim=True)
        energies = []
        current = gray
        kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                              dtype=x.dtype, device=x.device)
        for level in range(3):
            laplacian = F.conv2d(current, kernel, padding=1)
            energies.append(laplacian.abs().mean(dim=(1, 2, 3)))
            if level < 2:
                current = F.avg_pool2d(current, 2)
        return torch.stack(energies, dim=1)


class SPSLEncoder(nn.Module):
    """Self-Projected Spectral Learning autoencoder. ~1.2M params. Output: 64-dim."""
    def __init__(self, bottleneck_dim: int = 64):
        super().__init__()
        self.channel_reduce = nn.Conv2d(3, 1, 1) if True else nn.Identity()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.bottleneck = nn.Linear(128 * 4 * 4, bottleneck_dim)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128 * 4 * 4), nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[1] > 1:
            x = self.channel_reduce(x)
        enc = self.encoder(x).flatten(1)
        embedding = self.bottleneck(enc)
        recon = self.decoder(embedding)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return embedding, recon


# ======================================================================
# Phase 2A: Dual-SSL Audio Backbone
# ======================================================================

class DualSSLAudioBackbone(nn.Module):
    """Cross-attention fusion of WavLM + HuBERT features. Output: 256-dim."""
    def __init__(self, hidden_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )

    def forward(self, wavlm_features: torch.Tensor, hubert_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attended, _ = self.cross_attention(wavlm_features, hubert_features, hubert_features,
                                           key_padding_mask=mask)
        return self.projection(attended.mean(dim=1))


# ======================================================================
# Phase 2B: AMFF + NeXt-TDNN with ECA
# ======================================================================

class ECAModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.sigmoid(self.conv(self.avg_pool(x).transpose(1, 2))).transpose(1, 2)
        return x * y


class AMFF(nn.Module):
    """Attention-based Multi-scale Feature Fusion. ~120K params."""
    def __init__(self, in_dim: int = 256, scale_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(in_dim, scale_dim, 1),
            nn.Conv1d(in_dim, scale_dim, 3, padding=1),
            nn.Conv1d(in_dim, scale_dim, 5, padding=2),
        ])
        self.fusion_attn = nn.MultiheadAttention(scale_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(scale_dim * 3, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        scale_feats = [conv(xt).transpose(1, 2) for conv in self.scale_convs]
        min_len = min(s.shape[1] for s in scale_feats)
        scale_feats = [s[:, :min_len] for s in scale_feats]
        concat = torch.cat(scale_feats, dim=-1)
        attended, _ = self.fusion_attn(concat, concat, concat)
        return self.out_proj(attended.mean(dim=1))


class NeXtTDNN(nn.Module):
    """NeXt-TDNN with ECA. Output: hidden_dim*2 (mean+std pooled)."""
    def __init__(self, in_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 5, dilation=1, padding=2), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, dilation=2, padding=2), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, dilation=3, padding=3), nn.BatchNorm1d(hidden_dim), nn.PReLU(),
        )
        self.eca = ECAModule(hidden_dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.eca(self.tdnn(x.transpose(1, 2)))
        return torch.cat([h.mean(dim=2), h.std(dim=2)], dim=1)


# ======================================================================
# Phase 2C: AASIST2 Graph Attention
# ======================================================================

class AASIST2GraphAttention(nn.Module):
    """Graph attention for anti-spoofing. ~12M base. Output: 32-dim."""
    def __init__(self, in_features: int = 64, hidden_dim: int = 128, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.gal_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gal_layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                "ffn": nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim)),
                "norm1": nn.LayerNorm(hidden_dim), "norm2": nn.LayerNorm(hidden_dim),
            }))
        self.learn_t = nn.Parameter(torch.ones(1))
        self.output_proj = nn.Linear(hidden_dim, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.gal_layers:
            attn_out, _ = layer["attn"](h, h, h)
            attn_out = attn_out * torch.clamp(self.learn_t, min=0.1, max=10.0)
            h = layer["norm1"](h + attn_out)
            h = layer["norm2"](h + layer["ffn"](h))
        return self.output_proj(h.mean(dim=1))


# ======================================================================
# Phase 3A: FuseMoE
# ======================================================================

class FuseMoE(nn.Module):
    """Mixture of Experts with Laplace distance gating."""
    def __init__(self, input_dim: int = 8, num_experts: int = 4):
        super().__init__()
        self.expert_centroids = nn.Parameter(torch.randn(num_experts, input_dim))
        self.expert_bandwidths = nn.Parameter(torch.ones(num_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(x.unsqueeze(1), self.expert_centroids.unsqueeze(0)).squeeze(1)
        bandwidths = torch.clamp(self.expert_bandwidths, min=0.1)
        gate_weights = F.softmax(-dists / bandwidths.unsqueeze(0), dim=-1)
        expert_outputs = torch.cat([expert(x) for expert in self.experts], dim=-1)
        return torch.sigmoid((gate_weights * expert_outputs).sum(dim=-1))


# ======================================================================
# Phase 3B: GAMED Veto Voting
# ======================================================================

class GAMEDVeto(nn.Module):
    def __init__(self, strong_real_threshold: float = 0.1, strong_real_quality: float = 0.7,
                 quality_threshold: float = 0.3):
        super().__init__()
        self.strong_real_threshold = strong_real_threshold
        self.strong_real_quality = strong_real_quality
        self.quality_threshold = quality_threshold

    def forward(self, modality_scores: List[torch.Tensor],
                quality_scores: List[torch.Tensor]) -> torch.Tensor:
        B = modality_scores[0].shape[0]
        device = modality_scores[0].device
        votes = torch.cat([s.unsqueeze(1) for s in modality_scores], dim=1)
        majority = (votes > 0.5).float().mean(dim=1)
        verdict = torch.where(majority > 0.5, torch.ones(B, device=device),
                              torch.where(majority < 0.3, torch.zeros(B, device=device),
                                          torch.full((B,), 0.5, device=device)))
        for score, quality in zip(modality_scores, quality_scores):
            veto_mask = (score < self.strong_real_threshold) & (quality > self.strong_real_quality)
            verdict = torch.where(veto_mask, torch.clamp(verdict, max=0.5), verdict)
        return verdict


# ======================================================================
# Phase 3C: CAST Cross-Attention
# ======================================================================

class CASTCrossAttention(nn.Module):
    def __init__(self, dim_a: int = 32, dim_b: int = 32, hidden_dim: int = 32, num_heads: int = 2):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, hidden_dim) if dim_a != hidden_dim else nn.Identity()
        self.proj_b = nn.Linear(dim_b, hidden_dim) if dim_b != hidden_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.consistency_proj = nn.Linear(hidden_dim, 1)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.proj_a(feat_a).unsqueeze(1)
        b = self.proj_b(feat_b).unsqueeze(1)
        attended, _ = self.cross_attn(a, b, b)
        attended = attended.squeeze(1)
        consistency = torch.sigmoid(self.consistency_proj(attended)).squeeze(-1)
        return consistency, attended


# ======================================================================
# Phase 4A: AV-HuBERT Lip-Sync
# ======================================================================

class AVHuBERTLipSync(nn.Module):
    def __init__(self, input_dim: int = 1024, lora_rank: int = 8, out_dim: int = 128):
        super().__init__()
        self.lora_A = nn.Linear(input_dim, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, input_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self.sync_head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, 1), nn.Tanh(),
        )
        self.embedding_proj = nn.Linear(input_dim, out_dim)

    def forward(self, av_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        adapted = av_features + self.lora_B(self.lora_A(av_features))
        return self.sync_head(adapted).squeeze(-1), self.embedding_proj(adapted)


# ======================================================================
# Phase 4B: Localized Facial rPPG
# ======================================================================

class RPPGModule(nn.Module):
    def __init__(self, spatial_dim: int = 32, out_dim: int = 128):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )
        self.temporal_proj = nn.Sequential(nn.Linear(spatial_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, out_dim))
        self.pulse_head = nn.Sequential(nn.Linear(out_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 2))

    def forward(self, face_sequences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = face_sequences.shape
        sw = self.spatial_conv(face_sequences.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)
        weighted = face_sequences * sw
        r_mean = weighted[:, :, 0].mean(dim=(2, 3))
        g_mean = weighted[:, :, 1].mean(dim=(2, 3))
        b_mean = weighted[:, :, 2].mean(dim=(2, 3))
        X = 3 * r_mean - 2 * g_mean
        Y = 1.5 * r_mean + g_mean - 1.5 * b_mean
        xy_flat = torch.stack([X, Y], dim=-1).reshape(B, -1)
        rppg_embed = self.temporal_proj(xy_flat)
        pulse = self.pulse_head(rppg_embed)
        return rppg_embed, pulse


# ======================================================================
# Phase 5B: FGSM/PGD Adversarial Training
# ======================================================================

def fgsm_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                epsilon: float = 8.0 / 255.0, loss_fn=None) -> torch.Tensor:
    """Fast Gradient Sign Method attack."""
    if loss_fn is None:
        loss_fn = nn.BCELoss()
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = loss_fn(output, y.float())
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    return x_adv.detach()


def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               epsilon: float = 8.0 / 255.0, alpha: float = None,
               num_steps: int = 7, loss_fn=None) -> torch.Tensor:
    """Projected Gradient Descent attack."""
    if loss_fn is None:
        loss_fn = nn.BCELoss()
    if alpha is None:
        alpha = epsilon / (2 * num_steps)
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = loss_fn(output, y.float())
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = x_adv.detach()
    return x_adv


# ======================================================================
# Phase 5C: Feature Squeezing
# ======================================================================

def feature_squeeze(x: torch.Tensor, bit_depth: int = 5, kernel_size: int = 3) -> torch.Tensor:
    """Inference-time feature squeezing: median filter + quantization."""
    smoothed = x
    if kernel_size > 1:
        pad = kernel_size // 2
        smoothed = F.avg_pool2d(
            F.pad(x, (pad, pad, pad, pad), mode="reflect"),
            kernel_size, stride=1
        )
    step = 2 ** (8 - bit_depth)
    quantized = (smoothed / step).floor() * step
    return (x + smoothed + quantized) / 3.0


# ======================================================================
# Phase 5D: EWC Continual Learning
# ======================================================================

class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""

    def __init__(self, model: nn.Module, importance: float = 1000.0):
        self.model = model
        self.importance = importance
        self.fisher_information = {}
        self.old_params = {}
        self._estimated = False

    @torch.no_grad()
    def estimate_fisher(self, data_loader, loss_fn=None):
        """Estimate Fisher information from data."""
        if loss_fn is None:
            loss_fn = nn.BCELoss()
        self.model.eval()
        for name, param in self.model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)
            self.old_params[name] = param.clone()
        for batch in data_loader:
            self.model.zero_grad()
            output = self.model(batch["input"])
            loss = loss_fn(output, batch["target"].float())
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
        for name in self.fisher_information:
            self.fisher_information[name] /= max(len(data_loader), 1)
        self._estimated = True

    def penalty(self) -> torch.Tensor:
        """Compute EWC regularization penalty."""
        if not self._estimated:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                loss += (self.fisher_information[name] * (param - self.old_params[name]) ** 2).sum()
        return self.importance * loss
