#!/usr/bin/env python3
"""
FORECAST - can you catch ERCOT's 4CP peaks cheaply enough to be worth it?

This is the one number the rest of STRIKE reduces to. Transmission cost is
allocated by your demand during four 15-minute intervals a year. Catching them
is worth $56,510/MW-yr; curtailing costs $1,786/MWh of lost compute. So:

    break-even = 56,510 / 1,786 = 31.6 curtailment hours per year

The no-hindsight rules in strike.py that catch every peak need 114-220 h/yr.
This measures whether a forecaster built on published temperature forecasts
closes the gap to 32.

Nothing here is simulated. Real ERCOT load, real settlement peaks, and - the
part that makes the backtest honest - real FORECASTS with their publish times,
so the model only ever sees what was actually published before the decision.

    python3 forecast.py fetch          # pull forecasts (cached)
    python3 forecast.py backtest       # walk-forward, no hindsight
"""
import json, math, pathlib, statistics, sys
from datetime import datetime, timedelta, timezone

import strike
from strike import CPT, CACHE, _get

# Hours of notice a datacenter gets before it must be at reduced load. 1h is
# generous for software; 24h is what a scheduler would want. Keeping several
# lets the backtest report how much the answer depends on notice.
LEADS = (1, 4, 24)

# 4CP is settled Jun-Sep. Everything outside is irrelevant to the charge.
SEASONS = [("2024-06-01", "2024-10-01"),
           ("2025-06-01", "2025-10-01"),
           ("2026-06-01", "2026-08-05")]


def _iso(s):
    return datetime.fromisoformat(s)


def _reduce(rows, keep, fields):
    """Keep, per (interval, lead), the LATEST forecast published at least `lead`
    hours before the interval starts.

    This is the whole no-hindsight guarantee, and it lives here rather than in
    the model: a forecast revised at 16:00 for a 16:00 peak is a perfect
    prediction and completely useless. Filtering on publish_time is the only
    thing standing between an honest backtest and a fraudulent one."""
    for x in rows:
        try:
            t, p = _iso(x["interval_start_utc"]), _iso(x["publish_time_utc"])
        except (KeyError, TypeError, ValueError):
            continue
        for L in LEADS:
            if p > t - timedelta(hours=L):
                continue                       # published too late to act on
            k = (x["interval_start_utc"], L)
            cur = keep.get(k)
            if cur is None or _iso(cur["_pub"]) < p:
                keep[k] = dict({f: x.get(f) for f in fields}, _pub=x["publish_time_utc"])


CAP = 50000


def _page_reduce(ds, a, b, fields):
    """Page through a dataset by advancing the time window, reducing as we go.

    NOT by cursor. gridstatus caps a response at 50,000 rows and then reports
    `hasNextPage: false, cursor: null` - so a query spanning four months returns
    eleven days and calls itself complete. Trusting that flag silently produced
    a season with 837 records instead of 8,784, and a backtest built on it would
    have looked entirely normal.

    So the loop advances start_time to the last interval seen and stops only
    when it reaches `b` or stops making progress. Re-requesting the boundary
    interval is harmless: _reduce keeps the latest publish either way."""
    import time
    keep, cursor, pages = {}, a, 0
    while cursor < b:
        d = _get(ds, dict(start_time=cursor, end_time=b, limit=CAP))
        rows = d["data"]
        if not rows:
            break
        _reduce(rows, keep, fields)
        pages += 1
        last = max(x["interval_start_utc"] for x in rows)
        sys.stdout.write(f"\r    {ds} window {pages} -> {last[:10]}, kept {len(keep):,}   ")
        sys.stdout.flush()
        if len(rows) < CAP:
            break                              # genuinely the end of the range
        if last <= cursor:
            print(f"\n    WARNING: no progress past {cursor} - {ds} has >{CAP} rows "
                  f"in a single interval; range is incomplete.")
            break
        cursor = last
        time.sleep(0.4)
    print()
    return keep


SOURCES = {
    "netload": ("ercot_net_load_forecast",
                ["load_forecast", "solar_forecast", "wind_forecast", "net_load_forecast"]),
    "temp":    ("ercot_temperature_forecast_by_weather_zone",
                ["coast", "north_central", "south_central", "west", "southern"]),
}


def fetch():
    CACHE.mkdir(exist_ok=True)
    for a, b in SEASONS:
        for name, (ds, fields) in SOURCES.items():
            f = CACHE / f"fc_{name}_{a}.json"
            if f.exists():
                print(f"  {name} {a}: cached ({len(json.loads(f.read_text())):,})")
                continue
            print(f"  {name} {a}:")
            keep = _page_reduce(ds, a + "T00:00:00Z", b + "T00:00:00Z", fields)
            f.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in keep.items()}))
            print(f"    -> {f.name}  {len(keep):,} records")


def load_forecasts(name, lead):
    """-> {interval_start_utc: {field: value}} at the given notice."""
    out = {}
    for f in sorted(CACHE.glob(f"fc_{name}_*.json")):
        for k, v in json.loads(f.read_text()).items():
            iv, L = k.rsplit("|", 1)
            if int(L) == lead:
                out[iv] = v
    return out


# ------------------------------------------------------------------ model

ZONES = ["north_central", "coast", "south_central", "west", "southern"]


def _solve(A, b):
    """Least squares by normal equations + Gaussian elimination with partial
    pivoting. Five features; numpy would be a dependency for nothing."""
    n = len(A[0])
    M = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)]
         + [sum(A[k][i] * b[k] for k in range(len(A)))] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[p][i]) < 1e-9:
            return None
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r == i:
                continue
            f = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= f * M[i][c]
    return [M[i][n] / M[i][i] for i in range(n)]


def _feat(zt, day):
    z = zt[day]
    if any(k not in z for k in ZONES):
        return None
    return [1.0] + [z[k] for k in ZONES] + [1.0 if day.weekday() >= 5 else 0.0]


def assemble(lead):
    """-> (daily peak {date: (mw, interval)}, features {date: [...]})

    Both sides are real: peaks are measured 15-minute settlement load, features
    are temperature FORECASTS published at least `lead` hours before the day."""
    iv = strike.intervals()
    days = {}
    for t, mw in iv:
        if t.month in (6, 7, 8, 9):
            days.setdefault(t.date(), []).append((mw, t))
    peak = {d: max(v) for d, v in days.items()}

    zt = {}
    for iso, rec in load_forecasts("temp", lead).items():
        t = datetime.fromisoformat(iso).astimezone(CPT)
        if t.month not in (6, 7, 8, 9) or not (12 <= t.hour <= 19):
            continue
        d = zt.setdefault(t.date(), {})
        for z in ZONES:
            v = rec.get(z)
            if v is not None:
                d[z] = max(d.get(z, -999.0), v)

    feats = {d: f for d in set(peak) & set(zt) if (f := _feat(zt, d))}
    return peak, feats


def by_day(iv, window):
    """15-min load series per CPT day, restricted to the candidate window."""
    out = {}
    for t, mw in iv:
        if t.month in (6, 7, 8, 9) and window[0] <= t.hour <= window[1]:
            out.setdefault(t.date(), []).append((t, mw))
    for d in out:
        out[d].sort()
    return out


def backtest(lead=24, alpha=0.995, beta=None, window=(14, 18), min_train=30,
             rt_lag=2):
    """Two-stage, walk forward one day at a time.

    STAGE 1 (day-ahead, from temperature): arm the day if its predicted peak
    could beat the month's running maximum. The model for day D is fit ONLY on
    days strictly before D, from forecasts published >= `lead` hours ahead.

    STAGE 2 (intraday, from observed load): on an armed day, curtail interval T
    only once load is actually climbing toward the record. 4CP settles on ONE
    15-minute interval, so holding a fixed three-hour block is mostly waste.

    `beta=None` disables stage 2 and holds the whole window, which is what the
    naive rules do. `rt_lag` is how many 15-min intervals of publication delay
    the trigger must tolerate - ERCOT posts actual load within ~5 minutes, so 2
    intervals (30 min) is conservative.

    Nothing here reads a value it could not have had at decision time."""
    peak, feats = assemble(lead)
    days = sorted(feats)
    series = by_day(strike.intervals(), window)

    # the actual 4CP interval for each month = its maximum 15-min load
    months = {}
    for d, (mw, t) in peak.items():
        months.setdefault((d.year, d.month), []).append((mw, t, d))
    cp = {k: max(v) for k, v in months.items()}

    out, armed_days, held = [], [], {}
    for i, d in enumerate(days):
        train = [x for x in days[:i] if x in peak]
        if len(train) < min_train:
            continue
        w = _solve([feats[x] for x in train], [peak[x][0] for x in train])
        if w is None:
            continue
        pred = sum(a * b for a, b in zip(w, feats[d]))
        # running maximum of ACTUAL load this month, known at decision time
        prior = [peak[x][0] for x in days[:i] if (x.year, x.month) == (d.year, d.month)]
        rmax = max(prior) if prior else 0.0
        arm = pred >= rmax * alpha
        out.append(dict(day=d, pred=pred, actual=peak[d][0], rmax=rmax, armed=arm))
        if not arm:
            continue
        armed_days.append(d)

        day = series.get(d, [])
        if beta is None:
            held[d] = [t for t, _ in day]
            continue
        # trigger on load observed rt_lag intervals ago, against the higher of
        # the month's record and today's predicted peak
        ref = max(rmax, pred) * beta
        keep = []
        for j, (t, _) in enumerate(day):
            obs = day[j - rt_lag][1] if j >= rt_lag else None
            if obs is not None and obs >= ref:
                keep.append(t)
        held[d] = keep

    scored = {k: v for k, v in cp.items() if any(x["day"] == v[2] for x in out)}
    caught = sum(1 for k, (mw, t, d) in scored.items() if t in held.get(d, []))
    hours = sum(len(v) for v in held.values()) * 0.25
    return dict(out=out, armed=len(armed_days), hours=hours,
                caught=caught, total=len(scored), cp=scored, days=len(out))


# ------------------------------------------------------------------ diagnosis

def identifiability(lead=24, min_train=30):
    """Is the 4CP peak day even distinguishable in advance?

    Compares the precision the task DEMANDS - the load gap between a month's
    top day and its runner-up - against the precision any day-ahead forecast
    DELIVERS. If the gap is smaller than the error bar, the ranking is noise
    and no amount of model tuning recovers it.

    Three error figures, because one alone proves nothing:
      walk-forward  what an honest system actually achieves
      hindsight     the floor for these features, fit on the answers
      persistence   'tomorrow equals today', needing no forecast at all
    """
    peak, feats = assemble(lead)
    days = sorted(d for d in feats if d in peak)

    wf = []
    for i, d in enumerate(days):
        if i < min_train:
            continue
        w = _solve([feats[x] for x in days[:i]], [peak[x][0] for x in days[:i]])
        if w:
            wf.append(abs(sum(a * b for a, b in zip(w, feats[d])) - peak[d][0]))

    w = _solve([feats[d] for d in days], [peak[d][0] for d in days])
    hind = [abs(sum(a * b for a, b in zip(w, feats[d])) - peak[d][0]) for d in days]
    pers = [abs(peak[days[i]][0] - peak[days[i - 1]][0]) for i in range(1, len(days))
            if (days[i] - days[i - 1]).days == 1]

    months = {}
    for d in days:
        months.setdefault((d.year, d.month), []).append(peak[d][0])
    gaps = [sorted(v, reverse=True)[0] - sorted(v, reverse=True)[1]
            for v in months.values() if len(v) > 1]
    lvl = statistics.mean(max(v) for v in months.values())

    return dict(walk=statistics.mean(wf), hind=statistics.mean(hind),
                pers=statistics.mean(pers), gap=statistics.median(gaps),
                level=lvl, months=len(months))


# ------------------------------------------------------------------ report

def report():
    breakeven = strike.P["cp_rate"] / strike.compute_mwh()
    print(f"break-even: ${strike.P['cp_rate']:,.0f}/MW-yr / ${strike.compute_mwh():,.0f}/MWh"
          f" = {breakeven:.1f} curtailment hours per year\n")

    lead = 24
    peak, feats = assemble(lead)
    if not feats:
        sys.exit("no forecast data cached - run: python3 forecast.py fetch")
    seasons = len({d.year for d in feats})
    print(f"  {len(feats)} days over {seasons} seasons, {lead}h notice on the day-ahead call\n")

    print(f"  STAGE 1 ONLY - arm the day, hold the whole 14:00-18:59 window")
    print(f"  {'margin':>8}{'armed':>8}{'hrs/yr':>9}{'4CP caught':>13}{'pays?':>8}")
    print("  " + "-" * 47)
    for alpha in (1.00, 0.98, 0.96, 0.94):
        r = backtest(lead, alpha)
        if not r["total"]:
            continue
        hy = r["hours"] / seasons
        ok = "yes" if hy <= breakeven and r["caught"] == r["total"] else ""
        print(f"  {alpha:>8.2f}{r['armed']:>8}{hy:>9.0f}{r['caught']:>8}/{r['total']:<4}{ok:>8}")

    print(f"\n  STAGE 1 + 2 - arm the day, then trigger on observed load")
    print(f"  {'margin':>8}{'trigger':>9}{'armed':>8}{'hrs/yr':>9}{'4CP caught':>13}{'pays?':>8}")
    print("  " + "-" * 56)
    best = None
    for alpha in (0.96, 0.94):
        for beta in (0.995, 0.99, 0.98, 0.97, 0.96):
            r = backtest(lead, alpha, beta)
            if not r["total"]:
                continue
            hy = r["hours"] / seasons
            full = r["caught"] == r["total"]
            ok = "yes" if hy <= breakeven and full else ""
            if full and (best is None or hy < best[0]):
                best = (hy, alpha, beta, r)
            print(f"  {alpha:>8.2f}{beta:>9.3f}{r['armed']:>8}{hy:>9.1f}"
                  f"{r['caught']:>8}/{r['total']:<4}{ok:>8}")

    print()
    if best and best[0] <= breakeven:
        hy, alpha, beta, r = best
        print(f"  CLOSED. {r['caught']}/{r['total']} 4CP peaks at {hy:.1f} h/yr, against a")
        print(f"  {breakeven:.1f} h/yr break-even (margin={alpha}, trigger={beta}).")
        return

    # It did not close. The useful question is then whether that is a modelling
    # failure or a property of the problem - those call for opposite responses.
    d = identifiability(lead)
    print(f"  WHY IT DOES NOT CLOSE - is the peak day identifiable at all?\n")
    print(f"  precision the task demands")
    print(f"    median load gap, a month's #1 day vs its #2   {d['gap']:>8,.0f} MW"
          f"   ({d['gap']/d['level']:.2%} of peak)")
    print(f"  precision a day-ahead forecast delivers")
    print(f"    walk-forward temperature model                {d['walk']:>8,.0f} MW")
    print(f"    same features fit WITH hindsight (the floor)  {d['hind']:>8,.0f} MW")
    print(f"    persistence, 'tomorrow equals today'          {d['pers']:>8,.0f} MW")
    print(f"\n  The error bar is {d['hind']/d['gap']:.0f}x the gap it would have to resolve, and that")
    print(f"  holds even fitting on the answers. The temperature model does not")
    print(f"  beat persistence ({d['walk']:,.0f} vs {d['pers']:,.0f} MW), so these features carry")
    print(f"  almost no marginal information about WHICH hot day sets the record.")
    print(f"\n  Ranking the top days of a month needs a day-ahead peak forecast")
    print(f"  accurate to {d['gap']/d['level']:.2%}. The best these features reach, fit on")
    print(f"  the answers, is {d['hind']/d['level']:.1%} - an order of magnitude coarser. This is")
    print(f"  not a tuning problem. (Published ERCOT day-ahead load MAPE is")
    print(f"  around 2%, consistent with that, but is cited here, not measured.)")
    print(f"\n  MEASURED CONSEQUENCE: best honest full coverage is {best[0]:.1f} h/yr")
    print(f"  against a {breakeven:.1f} h/yr break-even, so curtailment-based 4CP")
    print(f"  avoidance pays only if {1-breakeven/best[0]:.0%} of the load is deferrable rather")
    print(f"  than lost. Since you cannot know WHICH day, you must be cheap to")
    print(f"  curtail on MANY days - which is precisely what densify.py measured")
    print(f"  at $0 (94% of the saving comes from batching, not from timing).")
    print(f"\n  The prediction route is closed. The flexibility route is not.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        fetch()
    elif cmd == "backtest":
        report()
    else:
        sys.exit(f"unknown command {cmd!r} - try: fetch | backtest")
