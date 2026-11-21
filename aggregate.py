#!/usr/bin/env python3
"""
Does the variance signature survive aggregation to a substation meter?

This is the objection that decides whether the finding is useful or a curiosity.
A grid operator does not see one GPU; it sees thousands summed at a meter. If
per-GPU variability averages away, there is nothing to read.

The reason to think it does not average away, and the whole point:

  * data-parallel TRAINING is synchronised by construction. Every GPU runs the
    same step, waits at the same all-reduce, and idles in the same gap. Their
    power swings are CORRELATED, and correlated sums preserve relative variance.

  * INFERENCE nodes serve independent request streams. Uncorrelated sums have
    their relative variance fall as 1/sqrt(N).

If that holds, aggregation does not weaken the signature - it AMPLIFIES it,
because only one of the two workloads smooths out.

This tests it on the measured traces, no GPU required.

  python3 aggregate.py traces_cv.json
"""
import argparse, json, math, pathlib, random, statistics


def detrend(x):
    n = len(x)
    if n < 3:
        return list(x)
    mx, my = (n - 1) / 2, statistics.mean(x)
    sxx = sum((i - mx) ** 2 for i in range(n)) or 1
    b = sum((i - mx) * (v - my) for i, v in enumerate(x)) / sxx
    return [v - (my + b * (i - mx)) for i, v in enumerate(x)]


def cv(x):
    m = statistics.mean(x)
    return statistics.pstdev(x) / m if m else 0.0


def aggregate(trace, n, group, rng):
    """Sum n GPUs at one meter, in synchronised groups of `group`.

    Within a group every GPU shares a phase (a data-parallel job stepping in
    lockstep). Across groups the phase is random (separate jobs, unrelated
    request streams). So group=n is a perfectly synchronised cluster, group=1
    is fully independent, and anything between is partial synchronisation.

    Phase offset is the honest way to model this: every node runs the same
    workload, they are just not in step. Adding fresh gaussian noise instead
    would bake in the answer.
    """
    L = len(trace)
    out = [0.0] * L
    left = n
    while left > 0:
        k = min(group, left)
        off = rng.randrange(L)
        for i in range(L):
            out[i] += k * trace[(i + off) % L]
        left -= k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="?", default="traces_cv.json")
    ap.add_argument("--sizes", default="1,10,100,1000,10000")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    d = json.loads(pathlib.Path(a.traces).read_text())
    rng = random.Random(a.seed)
    sizes = [int(x) for x in a.sizes.split(",")]

    # one representative of each workload, matched as closely as possible on power
    cases = {k: v for k, v in d["cases"].items()}
    def pick(prefix):
        best, bestw = None, None
        for k, v in cases.items():
            if not k.startswith(prefix):
                continue
            w = statistics.mean(v["samples"])
            if best is None or abs(w - 350) < abs(bestw - 350):
                best, bestw = k, w
        return best, bestw

    tk, tw = pick("train")
    ik, iw = pick("infer")
    print(f"matched-power pair: {tk} ({tw:.0f}W)  vs  {ik} ({iw:.0f}W)\n")

    tt = detrend(cases[tk]["samples"])
    ti = detrend(cases[ik]["samples"])
    base_t, base_i = statistics.mean(cases[tk]["samples"]), statistics.mean(cases[ik]["samples"])
    tt = [v + base_t for v in tt]
    ti = [v + base_i for v in ti]

    def trial(trace, n, group, reps=8):
        """Average over phase draws - a single draw of 3 groups is luck, not a result."""
        return statistics.mean(cv(aggregate(trace, n, group, rng)) for _ in range(reps))

    print(f"  {'GPUs':>7}  {'training CV':>13}{'inference CV':>14}{'separation':>12}")
    print("  " + "-" * 50)
    seps = []
    for n in sizes:
        # training: one synchronised job. inference: independent request streams.
        ct = trial(tt, n, n)
        ci = trial(ti, n, 1)
        sep = ct / ci if ci else float("inf")
        seps.append((n, sep))
        print(f"  {n:>7,}  {ct:>13.4f}{ci:>14.4f}{sep:>11.1f}x")

    # Perfect sync is the best case and it is trivially true: summing identical
    # traces multiplies the mean and the deviation alike, so CV is unchanged by
    # construction. The result that carries weight is the partial case below.
    print(f"\n  SENSITIVITY - training split across independent jobs, so the cluster")
    print(f"  is only partly in step. `jobs` = how many unsynchronised jobs share")
    print(f"  the meter. This is the number that decides whether the finding holds.")
    print(f"\n  {'GPUs':>7}{'jobs':>7}{'training CV':>13}{'inference CV':>14}{'separation':>12}")
    print("  " + "-" * 53)
    worst = None
    for n in (100, 1000, 10000):
        ci = trial(ti, n, 1)
        for jobs in (1, 8, 64, 512):
            if jobs > n:
                continue
            ct = trial(tt, n, max(1, n // jobs))
            sep = ct / ci if ci else 0
            worst = sep if worst is None else min(worst, sep)
            print(f"  {n:>7,}{jobs:>7}{ct:>13.4f}{ci:>14.4f}{sep:>11.1f}x")

    print()
    if worst and worst >= 2:
        print(f"  In the worst case tested the two bands stay {worst:.1f}x apart.")
        print(f"  Summing over a fleet makes the difference clearer, because the")
        print(f"  inference nodes average toward flat while synchronised training")
        print(f"  keeps its shape.")
    else:
        print(f"  The bands are only {worst:.1f}x apart in the worst case, so the")
        print(f"  difference washes out at fleet scale and would need per-node data.")


if __name__ == "__main__":
    main()
