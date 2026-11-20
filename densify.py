#!/usr/bin/env python3
"""
DENSIFY - how much is deferral worth, and why?

DIMMER's thesis died on measurement: every workload lever sheds power more
expensively than switching off (see README). But the same sweep showed something
larger going the other way.

    fp16 batch   1   4.159 J/token
    fp16 batch 128   0.078 J/token      -> 54x less energy for identical work

Meanwhile timing is worth 2.4x: ERCOT LZ_HOUSTON averages $31.86/MWh and its
cheapest quartile $13.33/MWh.

So deferral's value is NOT price arbitrage. It is that waiting accumulates work,
and accumulated work can be batched. Density is ~22x the bigger lever, and the
scheduler exists to harvest it.

The constraint is deadlines. You cannot accumulate forever, so the real question
is: given the slack your workload actually has, what batch size can you reach and
what is it worth? That tradeoff is what this simulates.

  python3 densify.py                        sweep deadline slack
  python3 densify.py --arrivals 200 --hours 720
"""
import argparse, json, math, pathlib, random, statistics

import strike

HERE = pathlib.Path(__file__).parent


# ------------------------------------------------------------------ inputs

def energy_curve(path="frontier_7b.json", dtype="fp16"):
    """batch -> J/token, measured. Interpolated log-log between measured points
    because the curve is strongly convex and linear interpolation would badly
    overstate the gain at intermediate batch sizes."""
    d = json.loads((HERE / path).read_text())
    pts = sorted((r["batch"], r["joules_per_token"])
                 for r in d["rows"] if r.get("dtype") == dtype)
    if not pts:
        raise SystemExit(f"no {dtype} rows in {path}")

    def j_per_token(batch):
        b = max(pts[0][0], min(pts[-1][0], batch))
        for (b0, j0), (b1, j1) in zip(pts, pts[1:]):
            if b0 <= b <= b1:
                if b1 == b0:
                    return j0
                t = (math.log(b) - math.log(b0)) / (math.log(b1) - math.log(b0))
                return math.exp(math.log(j0) + t * (math.log(j1) - math.log(j0)))
        return pts[-1][1]

    return j_per_token, pts


def prices(cache="cache"):
    """ERCOT LZ_HOUSTON 15-min settlement prices, in arrival order."""
    out = []
    for f in sorted((HERE / cache).glob("lz_houston_*.json")):
        out += [x for x in json.loads(f.read_text()) if x.get("spp") is not None]
    out.sort(key=lambda x: x["interval_start_utc"])
    return [x["spp"] for x in out]


# ------------------------------------------------------------------ simulation

def simulate(jobs, px, jtok, max_batch, slack_h, interval_h=0.25, avoid=()):
    """Hold arrivals until the batch is full or the earliest deadline forces a
    run. `avoid` is a set of local hours the grid wants kept clear - the 4CP
    band. Returns (energy MWh, cost $, runs, mean batch achieved).

    A deadline always wins over the grid: a job at its last chance runs even
    inside the avoid window. That is the honest way round - the datacenter is
    offering flexibility, not accepting an outage."""
    pending, energy_mwh, cost, runs, batches = [], 0.0, 0.0, 0, []
    n = len(px)
    horizon = max(j["t"] for j in jobs) + int(slack_h / interval_h) + 1

    for step in range(horizon + 1):
        pending += [j for j in jobs if j["t"] == step]
        if not pending:
            continue
        hour = (step * interval_h) % 24
        blocked = int(hour) in avoid
        # deadline pressure: must run if anything is at its last chance
        forced = any(step >= j["t"] + int(slack_h / interval_h) for j in pending)
        if blocked and not forced and step != horizon:
            continue
        if len(pending) >= max_batch or forced or step == horizon:
            batch = len(pending)
            toks = sum(j["tokens"] for j in pending)
            mwh = toks * jtok(batch) / 3.6e9
            energy_mwh += mwh
            cost += mwh * px[step % n]
            runs += 1
            batches.append(batch)
            pending = []
    return energy_mwh, cost, runs, (statistics.mean(batches) if batches else 0)


def make_jobs(n, hours, tokens_mean, seed=7, interval_h=0.25):
    r = random.Random(seed)
    steps = int(hours / interval_h)
    return [dict(t=r.randrange(steps),
                 tokens=max(1, int(r.expovariate(1 / tokens_mean))))
            for _ in range(n)]


# ------------------------------------------------------------------ report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrivals", type=int, default=2000)
    ap.add_argument("--hours", type=int, default=168)
    ap.add_argument("--tokens", type=int, default=500_000_000,
                    help="mean tokens/job (default sized so a week of jobs is "
                         "~1e12 tokens/yr, roughly a 5MW inference cluster)")
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--curve", default="frontier_7b.json")
    a = ap.parse_args()

    jtok, pts = energy_curve(a.curve)
    px = prices()
    jobs = make_jobs(a.arrivals, a.hours, a.tokens)
    total_tokens = sum(j["tokens"] for j in jobs)

    print(f"measured curve ({a.curve}): " +
          "  ".join(f"b{b}={j:.3f}J" for b, j in pts))
    print(f"prices: {len(px):,} intervals, mean ${statistics.mean(px):.2f}/MWh")
    print(f"workload: {a.arrivals:,} jobs over {a.hours}h, {total_tokens/1e9:.1f}B tokens\n")

    yr = 8760 / a.hours          # annualise the simulated window
    print(f"  {'deadline slack':>15}{'mean batch':>12}{'runs':>8}"
          f"{'MWh/yr':>11}{'$/yr':>12}{'vs no slack':>13}")
    print("  " + "-" * 72)
    base = None
    for slack in (0, 0.5, 1, 2, 4, 8, 24, 72):
        mwh, cost, runs, mb = simulate(jobs, px, jtok, a.max_batch, slack)
        if base is None:
            base = cost
        saved = (1 - cost / base) * 100 if base else 0
        print(f"  {slack:>13}h{mb:>12.0f}{runs:>8}{mwh*yr:>11,.0f}{cost*yr:>12,.0f}"
              f"{saved:>12.0f}%")

    # Separate the two mechanisms. Timing alone = same (dense-less) batching,
    # but every run moved to the cheapest quartile of hours.
    cheap = statistics.mean(sorted(px)[:len(px) // 4])
    mwh0, cost0, _, _ = simulate(jobs, px, jtok, a.max_batch, 0)
    mwh8, cost8, _, _ = simulate(jobs, px, jtok, a.max_batch, 8)
    timing_only = mwh0 * cheap
    print(f"\n  WHICH MECHANISM PAYS")
    print(f"    no slack, price-blind          ${cost0*yr:>12,.0f}/yr")
    print(f"    timing only (cheapest quartile)${timing_only*yr:>12,.0f}/yr"
          f"   {(1-timing_only/cost0)*100:>5.0f}% saved")
    print(f"    density only (8h slack)        ${cost8*yr:>12,.0f}/yr"
          f"   {(1-cost8/cost0)*100:>5.0f}% saved")
    # --- the grid half ---------------------------------------------------
    # All 11 measured ERCOT 4CP peaks land 15:00-17:59 CPT. Keeping a 3-hour
    # band clear is what makes the load dispatchable; the question is its cost.
    AVOID = (15, 16, 17)
    print(f"\n  GRID SERVICE - keeping {AVOID[0]}:00-{AVOID[-1]}:59 clear (the 4CP band)")
    print(f"  {'slack':>8}{'$/yr free-running':>20}{'$/yr avoiding 4CP':>20}{'cost of service':>18}")
    for slack in (2, 4, 8, 24):
        _, free, _, _ = simulate(jobs, px, jtok, a.max_batch, slack)
        _, avd,  _, _ = simulate(jobs, px, jtok, a.max_batch, slack, avoid=AVOID)
        delta = (avd - free) * yr
        print(f"  {slack:>6}h{free*yr:>20,.0f}{avd*yr:>20,.0f}{delta:>+17,.0f}")
    print(f"\n  With enough slack the 4CP band costs ~nothing to vacate - the load")
    print(f"  simply lands elsewhere. That is dispatchable capacity the grid can")
    print(f"  count on, at no loss to the operator, and it avoids a 4CP charge")
    print(f"  worth ${strike.P['cp_rate']:,.0f}/MW-yr (strike.py). Curtailment, measured earlier,")
    print(f"  costs -$300,000/MW-yr to deliver the same service.")

    print(f"\n  Deferral pays by ENABLING BATCHING, not by finding cheap hours.")
    print(f"  The same work costs {pts[0][1]/pts[-1][1]:.0f}x more energy run one job at a time,")
    print(f"  and the curve saturates by ~8h - you need hours of slack, not days.")


if __name__ == "__main__":
    main()
