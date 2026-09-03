import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from src.data import make_synthetic_signal, load_simple_npz
from src.experiment import train_experiment

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--data-path", default="")
p.add_argument("--test-data", default="",
               help="可选：另一住户的 NPZ（aggregate/target）。提供时 train/val 来自 --data-path，Test 用整套目标住户序列（跨家庭评估）")
p.add_argument("--out", default="reports/baseline")
p.add_argument("--synthetic", action="store_true")
args = p.parse_args()

cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
if args.synthetic or not args.data_path:
    x, y = make_synthetic_signal(20000, cfg.get("seed", 42))
    print("WARNING: synthetic mode. This is not a UK-DALE result.")
else:
    # Expected NPZ format: aggregate, target.
    x, y = load_simple_npz(args.data_path)

test_series = None
if args.test_data:
    tx, ty = load_simple_npz(args.test_data)
    test_series = (tx, ty)
    cfg.setdefault("data", {})["cross_test_path"] = str(args.test_data)

result = train_experiment(x, y, cfg, args.out, test_series=test_series)
print(result)
