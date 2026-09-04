"""Prepare UK-DALE ukdale.h5 into a simple NPZ for the Seq2Point pipeline.

Output NPZ contains two aligned 1-D float32 arrays:
    aggregate : whole-house mains power (watts)
    target    : appliance power (watts, default kettle)

Time alignment: resample both meters to `--sample-period` seconds (default 6),
inner-join on the overlapping time range, forward-fill short gaps, drop NaNs.

Meter location is resolved from the NILMTK metadata pickle embedded in
`building{N}.attrs['metadata']` (no nilmtk install required). The first
site_meter is used as mains; the first appliance whose `type` matches
`--appliance` is used as target.

Usage:
    python scripts/prepare_ukdale.py \
        --h5-path D:/Work/testPython/datasets/ukdale.h5 \
        --out D:/Work/testPython/datasets/ukdale_prepared.npz \
        --building 1 --appliance kettle --sample-period 6
"""
import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def find_meters(meta, appliance):
    """Return (mains_meter_id, target_meter_id) from building metadata pickle."""
    elec_meters = meta.get("elec_meters", {})
    mains_ids = [mid for mid, m in elec_meters.items() if m.get("site_meter")]
    if not mains_ids:
        raise RuntimeError("No site_meter found in metadata.elec_meters")

    target_ids = []
    for app in meta.get("appliances", []):
        atype = str(app.get("type", "")).lower().strip()
        orig = str(app.get("original_name", "")).lower().strip()
        if appliance.lower() in atype or appliance.lower() in orig or atype == appliance.lower():
            for mid in app.get("meters", []):
                target_ids.append(mid)
    if not target_ids:
        avail = sorted({str(a.get("type", "")) for a in meta.get("appliances", [])})
        raise RuntimeError(f"Appliance '{appliance}' not in metadata. Available: {avail}")

    return mains_ids[0], target_ids[0]


def load_meter_series(h5_path, building, meter_id, sample_period):
    """Read one meter's power series via pandas (pytables backend)."""
    key = f"/building{building}/elec/meter{meter_id}"
    # pandas.read_hdf returns DataFrame with tz-aware DatetimeIndex
    df = pd.read_hdf(h5_path, key=key)
    # prefer active power, else first column
    cols = list(df.columns)
    if isinstance(df.columns, pd.MultiIndex):
        cols_flat = [("power" if c[0] == "power" else c[0], c[1]) for c in cols]
        if ("power", "active") in cols_flat:
            col = ("power", "active")
        else:
            col = cols[0]
    else:
        col = cols[0]
    s = df[col].astype(np.float64)
    s = s[~s.index.duplicated(keep="first")]
    s = s.sort_index()
    # resample to common grid (mean within bin)
    s = s.resample(f"{sample_period}s").mean()
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--building", type=int, default=1)
    p.add_argument("--appliance", default="kettle")
    p.add_argument("--sample-period", type=int, default=6)
    args = p.parse_args()

    h5_path = Path(args.h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    # 1) locate meters from metadata pickle
    with h5py.File(h5_path, "r") as f:
        meta = pickle.loads(f[f"building{args.building}"].attrs["metadata"])
    mains_id, target_id = find_meters(meta, args.appliance)
    print(f"[meta] building{args.building}: mains=meter{mains_id}, "
          f"{args.appliance}=meter{target_id}")

    # 2) load + resample both series
    print(f"[load] reading mains meter{mains_id} ...")
    mains = load_meter_series(h5_path, args.building, mains_id, args.sample_period)
    print(f"  mains: n={len(mains)} range={mains.index[0]} .. {mains.index[-1]} "
          f"mean={mains.mean():.1f}W max={mains.max():.0f}W")
    print(f"[load] reading {args.appliance} meter{target_id} ...")
    target = load_meter_series(h5_path, args.building, target_id, args.sample_period)
    print(f"  target: n={len(target)} range={target.index[0]} .. {target.index[-1]} "
          f"mean={target.mean():.1f}W max={target.max():.0f}W "
          f"frac>50W={(target>50).mean():.4f}")

    # 3) align: inner join on common time index, forward-fill short gaps, drop NaN
    df = pd.concat({"aggregate": mains, "target": target}, axis=1, join="inner")
    print(f"[align] inner-join overlap: n={len(df)}")
    df = df.ffill(limit=5).dropna()
    print(f"[align] after ffill+dropna: n={len(df)}")

    aggregate = df["aggregate"].to_numpy(dtype=np.float32)
    target_arr = df["target"].to_numpy(dtype=np.float32)
    print(f"[out] aggregate shape={aggregate.shape} dtype={aggregate.dtype}")
    print(f"[out] target    shape={target_arr.shape} dtype={target_arr.dtype}")

    # 4) save NPZ (load_simple_npz expects keys: aggregate, target)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, aggregate=aggregate, target=target_arr)
    print(f"[done] saved -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
