"""ON/OFF operating-point calibration: scan the prediction threshold and report P/R/F1.

The 500 W threshold in metrics.py is applied to BOTH ground truth and predictions.
Ground-truth ON definition stays fixed (on_threshold_watts); this tool only sweeps the
prediction-side cut-off, i.e. how the regressor's output is converted into ON decisions.
Also reports regression MAE/RMSE (threshold-independent) and nMAE = MAE / std(target).

Usage:
  python scripts/threshold_scan.py --config reports/<run>/config.yaml --ckpt reports/<run>/best.pt \
      --data-path data/ukdale_prepared.npz --max-samples-test 30000 [--save-preds preds.npz] [--out scan.json]
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import load_simple_npz, build_splits
from src.model import NILMTransformer
from src.trainer import run_epoch

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--ckpt", required=True)
p.add_argument("--data-path", required=True)
p.add_argument("--max-samples-test", type=int, default=None)
p.add_argument("--true-threshold", type=float, default=500.0)
p.add_argument("--grid", default="")
p.add_argument("--save-preds", default="")
p.add_argument("--out", default="")
args = p.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
dcfg = dict(cfg["data"])
if args.max_samples_test:
    dcfg["max_samples_test"] = args.max_samples_test

x, y = load_simple_npz(args.data_path)
train_ds, _val, test_ds = build_splits(
    x, y, int(dcfg["window_size"]), dcfg["train_ratio"], dcfg["val_ratio"],
    dcfg.get("max_samples_train"), dcfg.get("max_samples_val"), dcfg.get("max_samples_test"))

model = NILMTransformer(**cfg["model"])
model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
_, yt, yp = run_epoch(model, DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False),
                      torch.device("cpu"), None, cfg["training"].get("loss", "mse"), 0.0)
yt = yt * train_ds.y_std + train_ds.y_mean
yp = yp * train_ds.y_std + train_ds.y_mean
if args.save_preds:
    np.savez_compressed(args.save_preds, y_true=yt, y_pred=yp)

mae = float(np.mean(np.abs(yt - yp)))
rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
nmae = float(mae / (np.std(yt) + 1e-12))

mask = yt >= args.true_threshold
if args.grid:
    grid = [float(t) for t in args.grid.split(",")]
else:
    grid = [float(v) for v in np.linspace(50, 1000, 20)]

rows = []
for t in grid:
    pred_on = yp >= t
    tp = float(np.sum(mask & pred_on)); fp = float(np.sum(~mask & pred_on)); fn = float(np.sum(mask & ~pred_on))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    rows.append({"pred_threshold_w": round(t, 1), "precision": round(prec, 4),
                 "recall": round(rec, 4), "f1": round(f1, 4), "tp": int(tp), "fp": int(fp), "fn": int(fn)})
both = [r for r in rows if r["precision"] > 0.9 and r["recall"] > 0.9]
best_f1 = max(rows, key=lambda r: r["f1"])
out = {"regression": {"mae_w": round(mae, 3), "rmse_w": round(rmse, 3), "nmae": round(nmae, 4),
                      "mae_kw": round(mae / 1000, 5), "n_test": int(len(yt))},
       "true_threshold_w": args.true_threshold,
       "grid": rows, "both_gt_0.9": both, "best_f1_point": best_f1}
s = json.dumps(out, indent=2, ensure_ascii=False)
if args.out:
    Path(args.out).write_text(s, encoding="utf-8")
print(s)
