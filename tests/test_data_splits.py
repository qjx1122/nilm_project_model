import numpy as np
from src.data import build_splits, build_crosshouse_split


def test_crosshouse_split_shapes_and_norm():
    rng = np.random.default_rng(0)
    agg_tr = rng.normal(300, 50, 4000).astype(np.float32)
    tgt_tr = rng.normal(50, 10, 4000).astype(np.float32)
    agg_te = rng.normal(800, 90, 3000).astype(np.float32)
    tgt_te = rng.normal(20, 5, 3000).astype(np.float32)
    tr, va, te = build_crosshouse_split(
        agg_tr, tgt_tr, agg_te, tgt_te, window=64,
        train_ratio=0.7, val_ratio=0.15, max_train=100, max_val=50, max_test=80,
    )
    assert len(tr) == 100 and len(va) == 50 and len(te) == 80
    # Normalization fitted on source-train segment only: x uses agg_tr[:2800]
    assert abs(tr.x_mean - float(agg_tr[:2800].mean())) < 1e-4
    # Test dataset indexes into the TARGET series, not the source.
    x_w, y_w = te[0]
    assert tuple(x_w.shape) == (64, 1)
    center = int(te.indices[0])
    expected = (agg_te[center - 32:center + 32] - tr.x_mean) / tr.x_std
    assert np.allclose(x_w.numpy(), expected[:, None])  # x_w is (window, 1)
    # Same normalizer shared across splits (source train stats).
    assert te.x_mean == tr.x_mean and te.y_std == tr.y_std


def test_regular_splits_unchanged():
    rng = np.random.default_rng(1)
    a = rng.normal(200, 30, 3000).astype(np.float32)
    t = rng.normal(40, 8, 3000).astype(np.float32)
    tr, va, te = build_splits(a, t, 64, 0.7, 0.15, 50, 20, 20)
    assert len(tr) == 50 and len(va) == 20 and len(te) == 20
