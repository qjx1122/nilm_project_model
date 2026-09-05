import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import regression_metrics


def run_epoch(model, loader, device, optimizer=None, loss_name="mse", grad_clip=1.0,
              event_weight=1.0, on_thr_norm=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    ys, ps = [], []
    elem_fn = nn.MSELoss(reduction="none") if loss_name == "mse" else nn.L1Loss(reduction="none")

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        per = elem_fn(pred, yb)
        if event_weight != 1.0 and on_thr_norm is not None:
            w = 1.0 + (yb >= on_thr_norm).to(per.dtype) * (float(event_weight) - 1.0)
            loss = (per * w).mean()
        else:
            loss = per.mean()
        if train:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(float(loss.item()))
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    return float(np.mean(losses)), np.concatenate(ys), np.concatenate(ps)


def fit(model, train_loader, val_loader, device, optimizer, epochs, patience,
        y_mean, y_std, checkpoint, on_threshold=500.0, loss_name="mse",
        grad_clip=1.0, scheduler=None, event_weight=1.0, select_on="mae", weight_avg=1):
    best = float("inf")
    if select_on in ("f1", "minprf"):
        best = -float("inf")
    bad = 0
    history = []
    top_states = []  # (key, epoch, cpu state_dict copies) for checkpoint weight averaging

    def _key_of(vm):
        if select_on == "f1":
            return vm["f1"]
        if select_on == "minprf":
            return min(vm["precision"], vm["recall"])
        return -vm["mae"]  # higher-is-better view for the top-k list
    on_thr_norm = None
    if event_weight != 1.0 and y_std > 1e-9:
        on_thr_norm = (float(on_threshold) - float(y_mean)) / float(y_std)

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.dataset, "resample"):
            train_loader.dataset.resample(epoch)   # no-op unless stochastic mode configured
        train_loss, yt, yp = run_epoch(model, train_loader, device, optimizer, loss_name, grad_clip,
                                       event_weight, on_thr_norm)
        val_loss, yv, pv = run_epoch(model, val_loader, device, None, loss_name, grad_clip)

        # Inverse normalization for human-readable metrics.
        yt_w = yt * y_std + y_mean
        yp_w = yp * y_std + y_mean
        yv_w = yv * y_std + y_mean
        pv_w = pv * y_std + y_mean

        tm = regression_metrics(yt_w, yp_w, on_threshold)
        vm = regression_metrics(yv_w, pv_w, on_threshold)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"train_{k}": v for k, v in tm.items()},
               **{f"val_{k}": v for k, v in vm.items()}}
        if scheduler is not None:
            row["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train MAE={tm['mae']:.2f} | val MAE={vm['mae']:.2f} | "
            f"val F1={vm['f1']:.3f} | "
            f"train R2={tm['r2']:.4f} | val R2={vm['r2']:.4f}"
        )

        key = vm["f1"] if select_on == "f1" else (
            min(vm["precision"], vm["recall"]) if select_on == "minprf" else vm["mae"])
        improved = key > best if select_on in ("f1", "minprf") else key < best
        if weight_avg > 1:
            k = _key_of(vm)
            top_states.append((k, epoch, {kk: vv.detach().cpu().clone().float()
                                          for kk, vv in model.state_dict().items()}))
            top_states.sort(key=lambda t: -t[0])
            del top_states[weight_avg:]
        if improved:
            best = key
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad += 1
            if bad >= patience:
                if scheduler is not None:
                    scheduler.step()
                break

        if scheduler is not None:
            scheduler.step()

    if weight_avg > 1 and len(top_states) > 1:
        avg = {kk: torch.mean(torch.stack([st[2][kk] for st in top_states]), dim=0)
               for kk in top_states[0][2]}
        torch.save(avg, checkpoint)
        avg_epochs = sorted(st[1] for st in top_states)
        print(f"Weight-averaged checkpoint over epochs {avg_epochs} (top-{weight_avg} by selection key)")
        history[-1]["weight_avg_epochs"] = avg_epochs

    return history, best_epoch
