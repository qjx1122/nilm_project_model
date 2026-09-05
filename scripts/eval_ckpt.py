"""Re-evaluate an existing checkpoint under a (possibly denser) test protocol.

Deterministic by design: split boundaries and normalization stats depend only on
the data length, ratios and window size (see src/data.py::build_splits), NOT on
max_samples_*. So passing the original run's config reproduces the exact same
model input distribution, while --max-samples-test can be raised to densify the
test set for apples-to-apples comparisons across runs.

Usage:
  python scripts/eval_ckpt.py --config reports/<run>/config.yaml \
      --ckpt reports/<run>/best.pt --data-path data/ukdale_prepared.npz \
      --max-samples-test 30000 --out reports/<run>/dense_test_eval.json
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
import torch
from torch.utils.data import DataLoader

from src.data import load_simple_npz, build_splits
from src.model import NILMTransformer
from src.metrics import regression_metrics
from src.trainer import run_epoch

p = argparse.ArgumentParser()
p.add_argument("--config", required=True, help="run 的 config.yaml（决定切分与归一化口径）")
p.add_argument("--ckpt", required=True)
p.add_argument("--data-path", required=True)
p.add_argument("--max-samples-test", type=int, default=None, help="加密测试集（默认沿用 config）")
p.add_argument("--out", default="", help="写出 json 路径（默认打印到 stdout）")
args = p.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
dcfg = dict(cfg["data"])
if args.max_samples_test:
    dcfg["max_samples_test"] = args.max_samples_test

x, y = load_simple_npz(args.data_path)
train_ds, _val_ds, test_ds = build_splits(
    x, y, int(dcfg["window_size"]),
    dcfg["train_ratio"], dcfg["val_ratio"],
    dcfg.get("max_samples_train"), dcfg.get("max_samples_val"),
    dcfg.get("max_samples_test"),
)

model = NILMTransformer(**cfg["model"])
model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))

loader = DataLoader(test_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)
_, yt, yp = run_epoch(model, loader, torch.device("cpu"), None,
                     cfg["training"].get("loss", "mse"), 0.0)
yt = yt * train_ds.y_std + train_ds.y_mean
yp = yp * train_ds.y_std + train_ds.y_mean
m = regression_metrics(yt, yp, float(cfg["metrics"]["on_threshold_watts"]))
m["n_test"] = len(test_ds)
m["eval"] = {"config": str(args.config), "ckpt": str(args.ckpt),
             "max_samples_test": dcfg.get("max_samples_test")}
out = json.dumps(m, indent=2)
if args.out:
    Path(args.out).write_text(out, encoding="utf-8")
print(out)
