# -*- coding: utf-8 -*-
"""ContinuSyncFormer — sub-frame synchronization model.

Encoder (temporal self-attention + RoPE, per view) → Cross-view attention (soft-alignment)
→ Hierarchical regression head (Δt ∈ ℝ). Ablation flags expose each component so the paper's
ablation study is a matter of config. Tasks #11–#18 (WS2).
"""
from __future__ import annotations
from ..core.interfaces import SyncMethod
from ..core.types import KeypointSequence, SyncResult

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

if _HAS_TORCH:
    from .attention import EncoderBlock, CrossViewBlock
    from .heads import HierarchicalHead
    from .rope import sinusoidal_pe, rope_cache
    from .occlusion import gate_keypoints, reliability_bias

    class ContinuSyncFormer(nn.Module):
        def __init__(self, n_joints=17, d_model=128, n_heads=4, n_layers=3, max_offset=8,
                     pos_encoding="rope", cross_view=True, occlusion_aware=False,
                     normalize=True, motion_input=False):
            super().__init__()
            assert pos_encoding in ("rope", "sincos", "none")
            self.pos_encoding = pos_encoding
            self.cross_view = cross_view
            self.occlusion_aware = occlusion_aware
            self.normalize = normalize
            # motion_input: per-joint speed magnitudes instead of raw coords —
            # view-tolerant (rotation/translation/scale-invariant after z-score),
            # which is what makes cross-camera pairs (B1) learnable.
            # motion_input=True or "speed": speed only; "speed+vy": speed and the signed vertical
            # image velocity, which separates the up- and down-strokes of a limb (gait aliasing).
            self.motion_input = bool(motion_input)
            self.motion_mode = ("speed" if motion_input is True else motion_input) if motion_input else None
            assert self.motion_mode in (None, "speed", "speed+vy")
            self.deroll = True     # speed+vy: take "vertical" along the body axis (roll-invariant);
                                   # trained on upright Panoptic views, where both coincide
            use_rope = (pos_encoding == "rope")

            self.head_dim = d_model // n_heads
            self.max_offset = max_offset
            self.embed = nn.Linear(n_joints * (1 if self.motion_mode == "speed" else 2), d_model)
            self.enc = nn.ModuleList(
                [EncoderBlock(d_model, n_heads, use_rope) for _ in range(n_layers)])
            if cross_view:
                self.cross = CrossViewBlock(d_model, n_heads, use_rope)
            self.head = HierarchicalHead(max_offset)

        # -- helpers --
        def _norm(self, kp):
            """Center + scale keypoints per sample ('pose' input). normalize=False => 'raw'."""
            if not self.normalize:
                return kp
            mu = kp.mean(dim=(1, 2), keepdim=True)
            sd = kp.std(dim=(1, 2), keepdim=True) + 1e-6
            return (kp - mu) / sd

        def _motion(self, kp):
            """Per-joint speed, z-scored over time (cf. baselines.motion_features)."""
            vel = torch.gradient(kp, dim=1)[0]                 # (B,T,J,2)
            feat = vel.norm(dim=-1)                            # (B,T,J)
            if self.motion_mode == "speed+vy":
                if self.deroll:                                # vertical = body axis (neck -> mid-hip)
                    a = (kp[:, :, 2] - kp[:, :, 0]).median(dim=1).values          # (B,2)
                    d = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    vy = (vel * d[:, None, None, :]).sum(-1)
                else:
                    vy = vel[..., 1]
                feat = torch.cat([feat, vy], dim=-1)           # (B,T,2J)
            mu = feat.mean(dim=1, keepdim=True)
            sd = feat.std(dim=1, keepdim=True) + 1e-6
            return (feat - mu) / sd

        def _encode(self, kp, vis, rope):
            if self.motion_input:
                feat = self._motion(kp)                        # (B,T,J)
                if vis is not None and vis.numel():
                    feat = feat * (vis.repeat(1, 1, 2) if self.motion_mode == "speed+vy" else vis)
                x = self.embed(feat)                           # (B,T,d)
            else:
                kp = gate_keypoints(self._norm(kp), vis)       # (B,T,J,2)
                B, T, J, _ = kp.shape
                x = self.embed(kp.reshape(B, T, J * 2))        # (B,T,d)
            if self.pos_encoding == "sincos":
                x = x + sinusoidal_pe(x.shape[1], x.shape[-1], x.device)[None]
            bias = reliability_bias(vis) if self.occlusion_aware else None
            for blk in self.enc:
                x = blk(x, rope=rope, key_bias=bias)
            return x, bias

        def _xcorr(self, za, zb):
            """Soft cross-correlation of two encoded sequences over integer lags.

            b lagged by Δt ⇒ za[t+Δt] matches zb[t] ⇒ the profile peaks at lag = Δt.
            Random/linear projections approximately preserve inner products (JL), so this
            carries the offset signal even before training and the encoder only sharpens it.
            Returns corr: (B, 2K+1)."""
            K = self.max_offset
            zan, zbn = F.normalize(za, dim=-1), F.normalize(zb, dim=-1)
            B, T, _ = za.shape
            cors = []
            for L in range(-K, K + 1):
                if L >= 0:
                    x, y = zan[:, L:], zbn[:, :T - L]
                else:
                    x, y = zan[:, :T + L], zbn[:, -L:]
                cors.append((x * y).sum(-1).mean(-1))          # (B,)
            return torch.stack(cors, dim=-1)                   # (B, 2K+1)

        def forward(self, kp_a, kp_b, vis_a=None, vis_b=None):
            B, T = kp_a.shape[0], kp_a.shape[1]
            rope = None
            if self.pos_encoding == "rope":
                pos = torch.arange(T, device=kp_a.device, dtype=torch.float32)
                rope = rope_cache(pos, self.head_dim)
            za, ba = self._encode(kp_a, vis_a, rope)
            zb, bb = self._encode(kp_b, vis_b, rope)
            S = None
            if self.cross_view:                                # enrich features across views
                za2, S = self.cross(za, zb, rope=None, key_bias_b=bb)
                zb2, _ = self.cross(zb, za, rope=None, key_bias_b=ba)
                za, zb = za2, zb2
            corr = self._xcorr(za, zb)                         # (B, nbins) coarse profile
            dt_soft, logits = self.head(corr)
            dt_hard = self.head.dt_hard(corr)                  # peak + parabolic sub-frame
            return {"dt": dt_soft, "dt_hard": dt_hard, "logits": logits,
                    "align": S, "corr": corr}

    def load_csf(checkpoint, device="cpu", **kw):
        """ContinuSyncFormer (19 joints, motion input) from a checkpoint; the input variant
        (speed, or speed + vertical velocity) is read from the embedding width."""
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        state = state["model"] if "model" in state else state
        mode = "speed+vy" if state["embed.weight"].shape[1] == 38 else True
        model = ContinuSyncFormer(n_joints=19, motion_input=mode, **kw).to(device).eval()
        model.load_state_dict(state)
        return model
else:
    class ContinuSyncFormer:  # pragma: no cover
        def __init__(self, *a, **k):
            raise ImportError("Install torch: pip install -r requirements-model.txt")


class ContinuSyncMethod(SyncMethod):
    """Wrap a trained ContinuSyncFormer as a SyncMethod so it plugs into benchmark.eval."""
    name = "ContinuSyncFormer"

    def __init__(self, checkpoint: str | None = None, device: str = "cpu",
                 model_kwargs: dict | None = None):
        if not _HAS_TORCH:
            raise ImportError("Install torch: pip install -r requirements-model.txt")
        self.device = device
        self.model = ContinuSyncFormer(**(model_kwargs or {})).to(device).eval()
        if checkpoint:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            self.model.load_state_dict(state["model"] if "model" in state else state)

    @torch.no_grad()
    def predict(self, a: KeypointSequence, b: KeypointSequence) -> SyncResult:
        T = min(a.T, b.T)
        ka = torch.tensor(a.xy[:T], dtype=torch.float32, device=self.device)[None]
        kb = torch.tensor(b.xy[:T], dtype=torch.float32, device=self.device)[None]
        out = self.model(ka, kb)
        dt = out["dt_hard"][0].item()
        conf = out["logits"].softmax(-1).max().item()
        return SyncResult(dt_frames=float(dt), confidence=float(conf))
