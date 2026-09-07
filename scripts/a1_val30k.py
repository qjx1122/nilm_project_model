"""A1 experiment: val densified to 30k + max-min(P,R) threshold selection, with B1 hysteresis prototypes.

Protocol (pre-registered, REPORT_TEST 2026-09-07 四元约束专题):
  - Members: median4/mean4 over {f8_big, f10_biglong, f12_jit_cont, f13_seed43} power predictions.
  - Labels: point labels target[center]; ON = label >= 500W. Predictions fire at >= t.
  - Selection: scan t in [20,520) step 5 on DENSE VAL 30k, maximize min(P,R). Test evaluated ONCE at t*.
  - Variants: raw / sup60 (a fired window needs >=1 neighbor with pred>=0.6t; endpoints treated as neighbor-absent) / m2of3 (majority of 3 consecutive windows >t).
  - SAE guardrail: |energy_error| of the aggregated power on test must be <= 0.2 for pass.
Outputs: reports/a1_val30k.json (+ raw arrays cached at reports/.a1_preds.npz).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
LANES = ["tune_f8_big", "tune_f10_biglong", "tune_f12_jit_cont", "tune_f13_seed43"]
ON_W = 500.0
GRID = np.arange(20, 520, 5, dtype=np.float64)


def get_preds():
    """Return cached per-lane predictions on dense val30k/test30k, computing them if absent."""
    cache = REP / ".a1_preds.npz"
    if cache.exists():
        z = np.load(cache)
        if set(z.files) == {f"{s}_{l}" for s in ("val", "te") for l in LANES} | {"val_lab", "te_lab"}:
            return z
    import torch, yaml
    from torch.utils.data import DataLoader
    from src.model import NILMTransformer
    from src.data import build_splits
    from src.experiment import seed_everything
    z = np.load(ROOT / "data" / "ukdale_prepared.npz")
    agg, tgt = z["aggregate"], z["target"]
    cfg0 = yaml.safe_load((REP / "tune_f10_biglong" / "config.yaml").read_text(encoding="utf-8"))
    seed_everything(42)
    _tr, va, te = build_splits(agg, tgt, 128, 0.7, 0.15, 30000, 30000, 30000)
    tr = _tr  # normalization stats only
    arrays = {"val_lab": tgt[va.indices].astype(np.float32), "te_lab": tgt[te.indices].astype(np.float32)}

    def infer(ds, lane):
        cfg = yaml.safe_load((REP / lane / "config.yaml").read_text(encoding="utf-8"))
        m = NILMTransformer(**cfg["model"])
        m.load_state_dict(torch.load(REP / lane / "best.pt", map_location="cpu", weights_only=True))
        m.eval()
        ps = []
        with torch.no_grad():
            for x, _y in DataLoader(ds, batch_size=512):
                ps.append((m(x).numpy() * float(tr.y_std) + float(tr.y_mean)).ravel())
        return np.concatenate(ps).astype(np.float32)

    for lane in LANES:
        arrays[f"val_{lane}"] = infer(va, lane)
        arrays[f"te_{lane}"] = infer(te, lane)
        print("inferred", lane, flush=True)
    np.savez_compressed(cache, **arrays)
    return arrays


def prf(p, on):
    tp = float(np.count_nonzero(on & p)); fp = float(np.count_nonzero(~on & p)); fn = float(np.count_nonzero(on & ~p))
    P = tp / max(tp + fp, 1.0); R = tp / max(tp + fn, 1.0)
    return P, R, (2 * P * R / max(P + R, 1e-9))


def f_sup(pred, t, ratio=0.6):
    """support hysteresis: a window fired at >=t survives iff itself or a neighbor >= ratio*t."""
    fired = pred >= t
    weak = pred >= ratio * t
    nb = np.zeros_like(fired)
    nb[1:] |= weak[:-1]
    nb[:-1] |= weak[1:]
    return fired & nb


def f_m23c(pred, t):
    p = (pred >= t).astype(np.int32)
    pp = np.zeros(len(p) + 2, dtype=np.int32); pp[1:-1] = p
    return np.asarray((pp[:-2] + pp[1:-1] + pp[2:]) >= 2, dtype=bool)


def scan(power, lab, filt, grid=GRID):
    """Return list of dicts over t, and best-by-min(P,R) on the given split arrays (val side)."""
    rows = []
    for t in grid:
        f = filt(power, t)
        P, R, F1 = prf(f, lab >= ON_W)
        rows.append((min(P, R), t, P, R, F1))
    best = max(rows, key=lambda r: (r[0], r[2]))
    return best, rows


def main():
    z = get_preds()
    val_lab, te_lab = z["val_lab"].astype(np.float64), z["te_lab"].astype(np.float64)
    V = np.stack([z[f"val_{l}"] for l in LANES]).astype(np.float64)
    T = np.stack([z[f"te_{l}"] for l in LANES]).astype(np.float64)
    aggs = {"med4": (np.median(V, axis=0), np.median(T, axis=0)),
            "mean4": (V.mean(axis=0), T.mean(axis=0))}
    for i, l in enumerate(LANES):
        aggs[l.replace("tune_", "")] = (V[i], T[i])
    filters = {"raw": lambda p, t: p >= t, "sup60": f_sup, "m2of3": f_m23c}
    top_curves = {}
    n_on_val, n_on_te = int(np.count_nonzero(val_lab >= ON_W)), int(np.count_nonzero(te_lab >= ON_W))

    out = {"protocol": {"n_val": len(val_lab), "n_test": len(te_lab), "n_on_val": n_on_val,
                        "n_on_test": n_on_te, "grid": [float(GRID[0]), float(GRID[-1]), 5],
                        "sel": "max-min(P,R) on dense val30k; test evaluated once at t*",
                        "on_def": "target[center]>=500W vs pred>=t"},
           "rows": []}
    for aname, (pv, pt) in aggs.items():
        ee = float(np.sum(pt - te_lab) / np.sum(te_lab))
        mae = float(np.mean(np.abs(pt - te_lab)))
        for fname, filt in filters.items():
            (mm, tstar, Pvv, Rvv, F1v), rows_all = scan(pv, val_lab, filt)
            if aname == "med4":
                top = sorted(rows_all, key=lambda r: -r[0])[:5]
                top_curves[fname] = [[round(x[1]), round(x[0], 4), round(x[2], 4), round(x[3], 4)] for x in top]
            Pte, Rte, F1te = prf(filt(pt, tstar), te_lab >= ON_W)
            P5, R5, F5 = prf(pt >= 500, te_lab >= ON_W)
            out["rows"].append({"agg": aname, "filt": fname,
                                "t_star": float(tstar),
                                "val": dict(zip("P R F1 min".split(), [round(Pvv, 4), round(Rvv, 4), round(F1v, 4), round(mm, 4)])),
                                "test": dict(zip("P R F1 min".split(), [round(Pte, 4), round(Rte, 4), round(F1te, 4), round(min(Pte, Rte), 4)])),
                                "test@500": dict(zip("P R F1".split(), [round(P5, 4), round(R5, 4), round(F5, 4)])),
                                "sae_test": round(abs(ee), 4), "mae_test_W": round(mae, 2)})
            print(f"{aname:>9} {fname:>6} t*={tstar:5.0f}  val min={mm:.3f} | test P={Pte:.4f} R={Rte:.4f} F1={F1te:.4f} "
                  f"min={min(Pte, Rte):.4f} | @500 P={P5:.3f}/R={R5:.3f} | SAE={abs(ee):.3f}", flush=True)
    out["val_top5"] = top_curves
    # binomial 1-sigma on the test min at n_on (statistical honesty)
    p0 = max(n_on_te, 1)
    out["noise_1sigma_pp"] = round(100 * float(np.sqrt(0.9 * 0.1 / p0)), 2)
    (REP / "a1_val30k.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("written reports/a1_val30k.json; n_on val/test:", n_on_val, n_on_te,
          " 1sigma:", out["noise_1sigma_pp"], "pp", flush=True)


if __name__ == "__main__":
    main()
