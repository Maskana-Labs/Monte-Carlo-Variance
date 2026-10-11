"""Sanity checks. Run with:  python test_experiment.py   (pytest also works)."""

import numpy as np

from experiment import INTEGRANDS, METHODS, halton, replicate, sweep


def test_halton_first_points():
    H = halton(3, 2)
    np.testing.assert_allclose(H[:, 0], [0.5, 0.25, 0.75])  # base 2
    np.testing.assert_allclose(H[:, 1], [1 / 3, 2 / 3, 1 / 9])  # base 3


def test_halton_points_lie_in_unit_cube():
    H = halton(500, 4)
    assert H.min() >= 0.0 and H.max() < 1.0


def test_exact_integrals_by_brute_force():
    # Midpoint grid check of the "truth" constants used by the experiment.
    g = (np.arange(400) + 0.5) / 400
    x1, x2 = np.meshgrid(g, g)
    pts = np.stack([x1.ravel(), x2.ravel()], axis=1)
    assert abs(INTEGRANDS["discontinuous"]["fn"](pts).mean() - np.pi / 4) < 2e-3
    g = (np.arange(12) + 0.5) / 12
    grid = np.stack(np.meshgrid(g, g, g, g), axis=-1).reshape(-1, 4)
    assert abs(INTEGRANDS["smooth"]["fn"](grid).mean() / INTEGRANDS["smooth"]["truth"] - 1) < 1e-2


def test_all_estimators_are_unbiased_to_within_noise():
    reps, N = 300, 256
    H = halton(N, 4)
    for i_int, name in enumerate(INTEGRANDS):
        est = replicate(name, N, reps, seed=1, i_int=i_int, i_n=0, H=H[:, : INTEGRANDS[name]["d"]])
        for m in METHODS:
            err = est[m] - INTEGRANDS[name]["truth"]
            assert abs(err.mean()) < 4 * err.std() / np.sqrt(reps) + 1e-3, (name, m)


def test_sweep_is_reproducible():
    kw = dict(reps=5, min_log2n=4, max_log2n=6, seed=2)
    assert sweep(**kw) == sweep(**kw)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
