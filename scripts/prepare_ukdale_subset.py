"""Prepare a UK-DALE low-frequency subset CSV into the simple NPZ format.

Input: one or more CSVs with columns: timestamp, aggregate, <appliance>, ...
Output: NPZ with float32 arrays `aggregate` and `target` on a strict 6-second grid.

Honesty note: this script only reshapes REAL data. It never synthesizes rows;
short gaps are forward-filled (<= gap_limit rows) and long gaps cut the series,
keeping the longest remaining segment. All decisions are written to a JSON report.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_and_align(csv_paths, appliance, gap_limit_rows=10):
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p, parse_dates=["timestamp"])
        if "timestamp" not in df or "aggregate" not in df:
            raise ValueError(f"{p}: need 'timestamp' and 'aggregate' columns")
        if appliance not in df.columns:
            raise ValueError(f"{p}: appliance column '{appliance}' not found; have {list(df.columns)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    step = pd.Timedelta(seconds=6)
    grid = pd.date_range(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], freq=step, tz=df["timestamp"].dt.tz)
    full = pd.DataFrame({"timestamp": grid})
    merged = full.merge(df[["timestamp", "aggregate", appliance]], on="timestamp", how="left")
    missing = merged["aggregate"].isna()
    # Forward-fill short gaps only; long gaps stay NaN and act as segment cuts.
    merged["aggregate"] = merged["aggregate"].ffill(limit=gap_limit_rows)
    merged[appliance] = merged[appliance].ffill(limit=gap_limit_rows)
    valid = merged["aggregate"].notna() & merged[appliance].notna()

    # Longest contiguous valid run (no imputed rows inside).
    ids = (valid != valid.shift()).cumsum()
    runs = valid.groupby(ids).agg(["sum", "size"])
    runs = runs[runs["sum"] > runs["size"] * 0]
    ok_runs = runs[runs["size"] == runs["sum"]]  # fully-valid segments
    if ok_runs.empty:
        raise RuntimeError("no contiguous segment found")
    best = ok_runs["size"].idxmax()
    seg = merged[(ids == best) & valid].reset_index(drop=True)

    report = {
        "input_rows": int(len(df)),
        "grid_rows": int(len(merged)),
        "missing_before_fill": int(missing.sum()),
        "segment_start": str(seg["timestamp"].iloc[0]),
        "segment_end": str(seg["timestamp"].iloc[-1]),
        "segment_rows": int(len(seg)),
        "appliance": appliance,
        "aggregate_mean_w": float(seg["aggregate"].mean()),
        "aggregate_max_w": float(seg["aggregate"].max()),
        "target_mean_w": float(seg[appliance].mean()),
        "target_max_w": float(seg[appliance].max()),
    }
    return seg["aggregate"].to_numpy(np.float32), seg[appliance].to_numpy(np.float32), report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True, help="one or more UK-DALE subset CSVs (same house)")
    p.add_argument("--appliance", required=True)
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--report", default="", help="optional JSON report path")
    args = p.parse_args()

    agg, tgt, report = load_and_align([Path(c) for c in args.csv], args.appliance)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, aggregate=agg, target=tgt)
    report["out"] = str(out)
    report["npz_rows"] = int(len(agg))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
