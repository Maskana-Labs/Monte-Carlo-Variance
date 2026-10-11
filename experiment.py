"""Experiment 03: how fast does Monte Carlo error shrink, and can we make it shrink faster?

Four estimators of an integral over the unit cube, all given the same budget of N
evaluations of the integrand:

  plain       average of f at N uniform random points
  antithetic  N/2 random points u, each paired with its mirror 1-u
  control     plain average corrected by a cheap correlated function g(x) = sum(x)
              whose mean (d/2) is known exactly; the coefficient is estimated from the sample
  qmc         first N Halton points with a fresh random shift (Cranley-Patterson) per replicate

For each N we repeat the estimate many times, measure the root-mean-square error
against the exact answer, and fit the log-log slope of error against N.
Plain Monte Carlo should give a slope near -1/2. The question is what the others
do, and whether the answer depends on how smooth the integrand is.

Usage:
    python experiment.py                  # default run, writes ./results
    python experiment.py --reps 500 --max-log2n 16 --out results/big
"""

import argparse
import csv
import os

import numpy as np

PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)


# --------------------------------------------------------------------------
# Integrands on [0, 1]^d with known exact integrals.
# --------------------------------------------------------------------------
def smooth_fn(x):
    return np.exp(x.sum(axis=-1))


def disc_fn(x):
    return (x[..., 0] ** 2 + x[..., 1] ** 2 <= 1.0).astype(float)


INTEGRANDS = {
    "smooth": dict(
        fn=smooth_fn, d=4, truth=(np.e - 1.0) ** 4,
        title="smooth: exp(x1 + x2 + x3 + x4), exact (e-1)^4",
    ),
    "discontinuous": dict(
        fn=disc_fn, d=2, truth=np.pi / 4.0,
        title="discontinuous: quarter-disc indicator, exact pi/4",
    ),
}
METHODS = ("plain", "antithetic", "control", "qmc")


# --------------------------------------------------------------------------
# Low-discrepancy points
# --------------------------------------------------------------------------
def halton(n, d):
    """First n points of the d-dimensional Halton sequence (index 0 skipped), shape (n, d)."""
    out = np.empty((n, d))
    for j in range(d):
        base = PRIMES[j]
        i = np.arange(1, n + 1)
        r = np.zeros(n)
        f = 1.0 / base
        while i.any():
            r += f * (i % base)
            i //= base
            f /= base
        out[:, j] = r
    return out


# --------------------------------------------------------------------------
# Estimators. Each returns one estimate using N evaluations of f.
# --------------------------------------------------------------------------
def est_plain(f, d, N, rng, X=None):
    X = rng.random((N, d)) if X is None else X
    return float(f(X).mean())


def est_antithetic(f, d, N, rng):
    U = rng.random((N // 2, d))
    return float(0.5 * (f(U) + f(1.0 - U)).mean())


def est_control(f, d, N, rng, X=None):
    X = rng.random((N, d)) if X is None else X
    fx = f(X)
    g = X.sum(axis=1)
    cov = np.cov(fx, g)
    beta = cov[0, 1] / cov[1, 1]
    return float(fx.mean() - beta * (g.mean() - d / 2.0))


def est_qmc(f, d, N, rng, H):
    shift = rng.random(d)
    return float(f((H[:N] + shift) % 1.0).mean())


def replicate(name, N, reps, seed, i_int, i_n, H):
    """`reps` independent estimates of every method (plain and control share their sample)."""
    I = INTEGRANDS[name]
    f, d = I["fn"], I["d"]
    out = {m: np.empty(reps) for m in METHODS}
    for r in range(reps):
        rng = np.random.default_rng([seed, i_int, i_n, r])
        X = rng.random((N, d))
        out["plain"][r] = est_plain(f, d, N, rng, X)
        out["control"][r] = est_control(f, d, N, rng, X)
        out["antithetic"][r] = est_antithetic(f, d, N, rng)
        out["qmc"][r] = est_qmc(f, d, N, rng, H)
    return out


def sweep(reps=200, min_log2n=4, max_log2n=14, seed=0):
    rows = []
    for i_int, (name, I) in enumerate(INTEGRANDS.items()):
        Ns = [2**k for k in range(min_log2n, max_log2n + 1)]
        H = halton(Ns[-1], I["d"])
        for i_n, N in enumerate(Ns):
            est = replicate(name, N, reps, seed, i_int, i_n, H)
            for m in METHODS:
                err = est[m] - I["truth"]
                rows.append(dict(
                    integrand=name, method=m, N=N,
                    rmse=float(np.sqrt(np.mean(err**2))),
                    bias=float(err.mean()),
                ))
    return rows


def fit_slopes(rows, min_n=64):
    """Log-log slope of RMSE against N for each (integrand, method), using N >= min_n."""
    slopes = {}
    for name in INTEGRANDS:
        for m in METHODS:
            sel = [r for r in rows if r["integrand"] == name and r["method"] == m and r["N"] >= min_n]
            slope = np.polyfit(np.log([r["N"] for r in sel]), np.log([r["rmse"] for r in sel]), 1)[0]
            slopes[(name, m)] = float(slope)
    return slopes


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
COLORS = {"plain": "#555555", "antithetic": "#0072B2", "control": "#E69F00", "qmc": "#009E73"}


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def plot(rows, slopes, path, reps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(INTEGRANDS), figsize=(11, 4.4))
    for ax, (name, I) in zip(axes, INTEGRANDS.items()):
        for m in METHODS:
            sel = [r for r in rows if r["integrand"] == name and r["method"] == m]
            ax.plot([r["N"] for r in sel], [r["rmse"] for r in sel], marker="o", ms=3.5,
                    color=COLORS[m], label=f"{m} (slope {slopes[(name, m)]:.2f})")
        sel = [r for r in rows if r["integrand"] == name and r["method"] == "plain"]
        n0, e0 = sel[0]["N"], sel[0]["rmse"]
        ns = np.array([r["N"] for r in sel], dtype=float)
        ax.plot(ns, e0 * (ns / n0) ** -0.5, "k:", lw=1, label="slope -1/2 reference")
        ax.plot(ns, e0 * (ns / n0) ** -1.0, "k--", lw=1, label="slope -1 reference")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("function evaluations N")
        ax.set_ylabel("RMSE")
        ax.set_title(I["title"], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"Monte Carlo error vs budget ({reps} replicates per point; slopes fitted for N >= 64)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=200, help="replicates per (integrand, N)")
    ap.add_argument("--min-log2n", type=int, default=4)
    ap.add_argument("--max-log2n", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = sweep(args.reps, args.min_log2n, args.max_log2n, args.seed)
    slopes = fit_slopes(rows)
    write_csv(rows, os.path.join(args.out, "summary.csv"))
    plot(rows, slopes, os.path.join(args.out, "error_vs_budget.png"), args.reps)

    nmax = 2**args.max_log2n
    print(f"reps={args.reps} N=2^{args.min_log2n}..2^{args.max_log2n} seed={args.seed}\n")
    print(f"{'integrand':<15}{'method':<12}{'slope':>7}{'RMSE at Nmax':>15}")
    for name in INTEGRANDS:
        for m in METHODS:
            r = next(r for r in rows if r["integrand"] == name and r["method"] == m and r["N"] == nmax)
            print(f"{name:<15}{m:<12}{slopes[(name, m)]:>7.2f}{r['rmse']:>15.3g}")
    print(f"\nWrote summary.csv and error_vs_budget.png to {args.out}/")


if __name__ == "__main__":
    main()
