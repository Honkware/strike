#!/usr/bin/env python3
"""
Pins the power-signature finding against the measured traces.

Everything here rests on one claim: at matched power draw, training and
inference have different variance. That claim cost GPU time to establish and
would rot silently - a change to detrending, resampling, or the CV definition
could quietly move the bands together and the tool would keep printing GO.

So these tests assert the finding itself, not that the code runs. Each one is
written to FAIL if the analysis stops supporting the conclusion.

  python3 test_signature.py
"""
import json, pathlib, random, statistics, sys

import aggregate as agg
import trace_probe as tp

HERE = pathlib.Path(__file__).parent
D = json.loads((HERE / "traces_cv.json").read_text())


def series(label):
    c = D["cases"][label]
    return tp.resample(list(zip(c["t"], c["samples"])), c["eff_hz"]), c["eff_hz"]


def cvs(prefix):
    out = {}
    for k in D["cases"]:
        if k.startswith(prefix):
            s, hz = series(k)
            p = tp.periodicity(s, hz)
            out[k] = p["cv"]
    return out


T = None
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def main():
    print("measured traces\n")
    tcv, icv = cvs("train"), cvs("infer")

    # 1. The core finding: the bands do not overlap.
    check("training CV band sits entirely above inference",
          min(tcv.values()) > max(icv.values()),
          f"{min(tcv.values()):.3f} > {max(icv.values()):.3f}")

    # 2. The control. Without this the finding is just "training draws more
    #    watts", which is not a signature. This is the test that matters most.
    tw = {k: tp.periodicity(*series(k))["mean_w"] for k in tcv}
    iw = {k: tp.periodicity(*series(k))["mean_w"] for k in icv}
    pairs = [(a, b) for a in tw for b in iw
             if abs(tw[a] - iw[b]) / max(tw[a], iw[b]) < 0.05]
    check("a matched-power pair exists", bool(pairs),
          f"{pairs[0] if pairs else 'none'}")
    if pairs:
        a, b = pairs[0]
        check("separation survives at matched power", tcv[a] / icv[b] >= 2.0,
              f"{tw[a]:.0f}W vs {iw[b]:.0f}W -> {tcv[a]/icv[b]:.1f}x")

    # 3. Variance must track the workload, not the load. If CV rose with watts
    #    we would be measuring utilisation with extra steps. The disproof is a
    #    direct inversion: some inference case draws MORE power than some
    #    training case and still has LOWER CV. Monotonic-in-power forbids that.
    inv = [(b, a) for a in tcv for b in icv if iw[b] > tw[a] and icv[b] < tcv[a]]
    check("CV inverts against power somewhere (so it is not a power proxy)",
          bool(inv),
          f"{inv[0][0]} {iw[inv[0][0]]:.0f}W CV {icv[inv[0][0]]:.3f}  <  "
          f"{inv[0][1]} {tw[inv[0][1]]:.0f}W CV {tcv[inv[0][1]]:.3f}" if inv else "")

    # ...and within a workload, power barely moves CV. Training spans 345-590W
    # with a near-flat CV; if that spread ever approaches the between-workload
    # gap, the two effects are no longer separable.
    spread = (max(tcv.values()) - min(tcv.values())) / min(tcv.values())
    gap = min(tcv.values()) / max(icv.values())
    check("within-workload CV spread is small next to the between-workload gap",
          spread < 0.5 and gap > 2, f"spread {spread:.0%} vs gap {gap:.1f}x")

    # 4. The dead hypothesis stays dead. Periodicity was the original thesis;
    #    if a future change makes it "pass" it is far more likely a bug than a
    #    discovery, because the peaks sat at the bottom of the search band.
    sus = [k for k in D["cases"] if tp.periodicity(*series(k))["suspect"]]
    check("periodicity peaks are still flagged as drift, not rhythm",
          len(sus) >= len(D["cases"]) // 2, f"{len(sus)}/{len(D['cases'])} suspect")

    # 5. Detrending must be load-bearing. It is the fix for the original false
    #    GO, so a regression that removes it has to be caught here.
    s, hz = series("train_b1")
    raw = tp.spectrum([v / statistics.mean(s) for v in s], hz)
    det = tp.spectrum([v / statistics.mean(s) for v in tp._detrend(s)], hz)
    check("detrending removes low-frequency drift energy",
          max(m for _, m in det) < max(m for _, m in raw),
          f"peak {max(m for _,m in det):.5f} < {max(m for _,m in raw):.5f}")

    print("\naggregation to a substation meter\n")
    rng = random.Random(3)
    tt, ti = D["cases"]["train_b1"]["samples"], D["cases"]["infer_b256"]["samples"]

    # 6. Independent summing must wash variance out at the 1/sqrt(N) rate. If it
    #    does not, the aggregation model is wrong and result 7 means nothing.
    c1 = agg.cv(agg.aggregate(ti, 1, 1, rng))
    c100 = statistics.mean(agg.cv(agg.aggregate(ti, 100, 1, rng)) for _ in range(8))
    check("independent nodes average out at ~1/sqrt(N)",
          6 < c1 / c100 < 16, f"100 nodes -> {c1/c100:.1f}x reduction (sqrt(100)=10)")

    # 7. The objection that decides usefulness: does it survive at scale?
    for n, jobs in ((1000, 8), (1000, 64), (10000, 64)):
        ct = statistics.mean(agg.cv(agg.aggregate(tt, n, n // jobs, rng)) for _ in range(8))
        ci = statistics.mean(agg.cv(agg.aggregate(ti, n, 1, rng)) for _ in range(8))
        check(f"holds at {n:,} GPUs / {jobs} jobs", ct / ci >= 2.0, f"{ct/ci:.1f}x")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("all pinned. the finding is what it was when measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
