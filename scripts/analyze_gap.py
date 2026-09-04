import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import load_simple_npz, build_splits
from src.model import NILMTransformer
from src.trainer import run_epoch
from src.metrics import regression_metrics
import argparse

_p = argparse.ArgumentParser(description="Analyze val/test F1 gap for a trained run")
_p.add_argument("--npz", default=r"D:\Work\testPython\datasets\ukdale_prepared.npz",
                help="path to ukdale_prepared.npz")
_p.add_argument("--config", default="configs/baseline.yaml",
                help="config yaml used for the run (for model/split params)")
_p.add_argument("--ckpt", default="reports/baseline/best.pt",
                help="path to best.pt of the run to analyze")
_p.add_argument("--threshold", type=float, default=500.0)
_args = _p.parse_args()
NPZ = _args.npz
CFG = _args.config
CKPT = _args.ckpt
THRESH = _args.threshold


def find_runs(mask):
    """Return list of (start, length) for True runs in bool array."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]])))
    return [(int(s), int(e - s)) for s, e in zip(idx[::2], idx[1::2])]


def seg_stats(name, seg):
    on = seg >= THRESH
    runs = find_runs(on)
    on_pow = seg[on]
    run_lens = [r[1] for r in runs] if runs else []
    print(f"\n=== {name}: n={len(seg)} ===")
    print(f"  ON 点占比: {on.mean():.4f} ({on.sum()}/{len(seg)})")
    print(f"  ON 事件数(连续段): {len(runs)}")
    if run_lens:
        print(f"  事件时长(点,6s/点): mean={np.mean(run_lens):.1f} "
              f"median={np.median(run_lens):.0f} max={max(run_lens)}")
    if on_pow.any():
        print(f"  ON 功率(W): mean={on_pow.mean():.0f} median={np.median(on_pow):.0f} max={on_pow.max():.0f}")
    else:
        print("  ON 功率: n/a (no ON)")


print("=" * 60)
print("PART 1: 全量段 target 分布（时序 70/15/15 划分）")
print("=" * 60)
x, y = load_simple_npz(NPZ)
n = len(x)
a = int(n * 0.70)
b = int(n * (0.70 + 0.15))
print(f"总长 n={n}, train[0,{a}), val[{a},{b}), test[{b},{n})")
for name, (lo, hi) in [("train", (0, a)), ("val", (a, b)), ("test", (b, n))]:
    seg_stats(name, y[lo:hi])

print("\n" + "=" * 60)
print("PART 2: 6000 采样点分布（linspace 等距抽样）")
print("=" * 60)
cfg = yaml.safe_load(Path(CFG).read_text(encoding="utf-8"))
window = int(cfg["data"]["window_size"])
train_ds, val_ds, test_ds = build_splits(
    x, y, window,
    cfg["data"]["train_ratio"], cfg["data"]["val_ratio"],
    cfg["data"]["max_samples_train"], cfg["data"]["max_samples_val"],
    cfg["data"]["max_samples_test"],
)
print(f"train_centers={len(train_ds.indices)} val={len(val_ds.indices)} test={len(test_ds.indices)}")

y_val_true = y[val_ds.indices]
y_test_true = y[test_ds.indices]
print(f"\nval 6000 采样: 真实ON={(y_val_true>=THRESH).sum()} "
      f"占比={(y_val_true>=THRESH).mean():.4f}")
print(f"test 6000 采样: 真实ON={(y_test_true>=THRESH).sum()} "
      f"占比={(y_test_true>=THRESH).mean():.4f}")

print("\n" + "=" * 60)
print("PART 3: 模型预测 + 漏报/误报分析")
print("=" * 60)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NILMTransformer(**cfg["model"]).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)
_, yt, yp = run_epoch(model, test_loader, device, None,
                       cfg["training"]["loss"], cfg["training"]["grad_clip"])
yt_w = yt * train_ds.y_std + train_ds.y_mean
yp_w = yp * train_ds.y_std + train_ds.y_mean

on_true = yt_w >= THRESH
on_pred = yp_w >= THRESH
tp = on_true & on_pred
fp = (~on_true) & on_pred
fn = on_true & (~on_pred)
print(f"TP={tp.sum()} FP={fp.sum()} FN={fn.sum()} "
      f"true_ON={on_true.sum()} pred_ON={on_pred.sum()}")
print(f"\n漏报(FN, 真ON但预测OFF): {fn.sum()} 个点")
if fn.any():
    fn_true_pow = yt_w[fn]
    print(f"  漏报点真实功率: mean={fn_true_pow.mean():.0f} "
          f"median={np.median(fn_true_pow):.0f} max={fn_true_pow.max():.0f}")
    print(f"  漏报点预测功率: mean={yp_w[fn].mean():.0f} "
          f"median={np.median(yp_w[fn]):.0f} max={yp_w[fn].max():.0f}")
print(f"\n命中(TP, 真ON且预测ON): {tp.sum()} 个点")
if tp.any():
    print(f"  命中点真实功率: mean={yt_w[tp].mean():.0f} "
          f"median={np.median(yt_w[tp]):.0f}")
    print(f"  命中点预测功率: mean={yp_w[tp].mean():.0f}")
print(f"\n误报(FP, 真OFF但预测ON): {fp.sum()} 个点")
if fp.any():
    print(f"  误报点真实功率: mean={yt_w[fp].mean():.0f} "
          f"max={yt_w[fp].max():.0f}")
    print(f"  误报点预测功率: mean={yp_w[fp].mean():.0f}")

# 漏报点在 test 段内的位置分布
if fn.any():
    fn_pos = test_ds.indices[fn]
    test_start, test_end = test_ds.indices[0], test_ds.indices[-1]
    rel = (fn_pos - test_start) / max(test_end - test_start, 1)
    print(f"\n漏报点在 test 段相对位置: mean={rel.mean():.2f} "
          f"std={rel.std():.2f}")
    print(f"  前1/3漏报: {(rel<0.33).sum()} 中1/3: {((rel>=0.33)&(rel<0.67)).sum()} 后1/3: {(rel>=0.67).sum()}")

print("\n" + "=" * 60)
print("PART 4: test 段时间定位（从 ukdale.h5 读 mains DatetimeIndex）")
print("=" * 60)
try:
    import h5py, pandas as pd
    with h5py.File(r"D:\Work\testPython\datasets\ukdale.h5", "r") as f:
        df = pd.read_hdf(r"D:\Work\testPython\datasets\ukdale.h5",
                         key="/building1/elec/meter1")
    idx = df.index[~df.index.duplicated(keep="first")].sort_index()
    # 对齐后的 npz 是 resample('6s') inner join，这里用 mains 的 resample 6s 近似定位
    resampled = df.iloc[:, 0].astype(np.float64).resample("6s").mean()
    # npz 长度 n 对应 resampled 内 inner overlap 段。近似：取 resampled 前 n 个对应时间
    if len(resampled) >= n:
        test_idx = resampled.index[b:n]
        print(f"test 段时间范围(近似): {test_idx[0]} .. {test_idx[-1]}")
        val_idx = resampled.index[a:b]
        print(f"val  段时间范围(近似): {val_idx[0]} .. {val_idx[-1]}")
    else:
        print(f"resampled len {len(resampled)} < n {n}, skip")
except Exception as e:
    print(f"time locate skipped: {e}")
