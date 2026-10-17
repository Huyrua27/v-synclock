# -*- coding: utf-8 -*-
"""Training loop + evaluation for ContinuSyncFormer. Tasks #15, #16 (WS2).

    from vsynclock.model.train import train
    train(epochs=8, device="cuda")

Data is materialized once into GPU tensors (generation happens a single time), then
mini-batched manually — far faster than regenerating splines every epoch.
Reports Accin@0.1 / Frm.err / MAE(ms) on a held-out val set and saves the best checkpoint.
"""
from __future__ import annotations
import os
import torch

from .continusyncformer import ContinuSyncFormer
from .dataset import KeypointPairDataset
from .losses import hierarchical_loss
from ..benchmark import metrics

FPS = 30.0


def _materialize(ds: KeypointPairDataset, device):
    """Generate the whole set once → tensors on `device`. Returns ka,kb,dt,va,vb (va/vb may be None)."""
    has_vis = ds.occlusion_p > 0
    kas, kbs, dts, vas, vbs = [], [], [], [], []
    for i in range(len(ds)):
        ka, kb, dt, va, vb = ds[i]
        kas.append(ka); kbs.append(kb); dts.append(dt)
        if has_vis:
            vas.append(va); vbs.append(vb)
    ka = torch.stack(kas).to(device); kb = torch.stack(kbs).to(device)
    dt = torch.stack(dts).to(device)
    va = torch.stack(vas).to(device) if has_vis else None
    vb = torch.stack(vbs).to(device) if has_vis else None
    return ka, kb, dt, va, vb


def _batches(n, batch, shuffle, device):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for s in range(0, n, batch):
        yield idx[s:s + batch]


def _sel(t, ib): return None if t is None else t[ib]


@torch.no_grad()
def evaluate(model, ka, kb, dt, va, vb, batch=256):
    model.eval()
    preds = []
    for ib in _batches(len(dt), batch, False, ka.device):
        out = model(ka[ib], kb[ib], _sel(va, ib), _sel(vb, ib))
        preds += out["dt_hard"].cpu().numpy().tolist()
    gts = dt.cpu().numpy().tolist()
    return {"Accin@0.1": metrics.accin(preds, gts, 0.1),
            "Frm.err": metrics.frm_err(preds, gts),
            "MAE_ms": metrics.mae_ms(preds, gts, FPS)}


def train(epochs=8, batch=64, lr=3e-4, device=None, out="checkpoints/csf.pt",
          model_kwargs=None, n_train=6000, n_val=1000, occlusion_p=0.0, log=True,
          train_ds=None, val_ds=None, select="accin", init=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ContinuSyncFormer(**(model_kwargs or {})).to(device)
    if init:                                     # fine-tune from an existing checkpoint
        state = torch.load(init, map_location=device, weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state)
        if log:
            print(f"initialized from {init}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    centers = model.head.centers

    if train_ds is None:
        train_ds = KeypointPairDataset(n=n_train, seed=1, occlusion_p=occlusion_p)
    if val_ds is None:
        val_ds = KeypointPairDataset(n=n_val, seed=999, occlusion_p=occlusion_p)
    n_train, n_val = len(train_ds), len(val_ds)
    if log:
        print(f"[{device}] materializing data ({n_train} train / {n_val} val)...", flush=True)
    tka, tkb, tdt, tva, tvb = _materialize(train_ds, device)
    vka, vkb, vdt, vva, vvb = _materialize(val_ds, device)

    best, best_state = float("-inf"), None
    for ep in range(1, epochs + 1):
        model.train(); run, nb = 0.0, 0
        for ib in _batches(n_train, batch, True, device):
            res = model(tka[ib], tkb[ib], _sel(tva, ib), _sel(tvb, ib))
            loss, _ = hierarchical_loss(res["logits"], res["dt"], tdt[ib], centers)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); run += loss.item(); nb += 1
        m = evaluate(model, vka, vkb, vdt, vva, vvb)
        if log:
            k = min(500, n_train)  # train-subset metric: overfit vs underfit
            tm = evaluate(model, tka[:k], tkb[:k], tdt[:k], _sel(tva, slice(0, k)),
                          _sel(tvb, slice(0, k)))
            print(f"ep{ep:02d} loss={run/nb:.4f} Accin@0.1={m['Accin@0.1']:.3f} "
                  f"Frm.err={m['Frm.err']:.4f} MAE={m['MAE_ms']:.2f}ms "
                  f"(train Accin@0.1={tm['Accin@0.1']:.3f})", flush=True)
        # Drift line-fitting cares about mean error, not threshold accuracy →
        # allow selecting the checkpoint by lowest Frm.err instead of Accin@0.1.
        score = -m["Frm.err"] if select == "frmerr" else m["Accin@0.1"]
        if score > best:
            best = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if out:                              # save as we go: survives interruption
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                torch.save({"model": best_state, "best_score": best, "select": select,
                            "epoch": ep}, out)
    if out and best_state is not None:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({"model": best_state, "best_score": best, "select": select}, out)
        if log: print(f"saved {out}  (best {select} score={best:.3f})", flush=True)
    return best
