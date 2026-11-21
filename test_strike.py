#!/usr/bin/env python3
"""
Pins the 4CP detection and the curtailment arithmetic that strike.py,
forecast.py, dimmer.py and densify.py all build on.

The ERCOT cache is not shipped, so the fixtures here are small hand-built series
that reuse values from it: the four 2024 monthly peaks as strike.py found them,
and an LZ_HOUSTON price year shaped to reproduce the $100 row of `strike.py
energy`. Runs offline with no key.

  python3 test_strike.py
"""
import json, pathlib, sys, tempfile
from datetime import datetime, timedelta, timezone

import strike

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def utc(cpt):
    """'2024-08-20 16:45' CPT -> gridstatus interval_start_utc string."""
    t = datetime.fromisoformat(cpt).replace(tzinfo=strike.CPT)
    return t.astimezone(timezone.utc).isoformat()


# 2024 monthly system peaks from `strike.py fourcp` on the full cache
PEAKS = {"2024-06-27 16:45": 80274, "2024-07-01 16:30": 81525,
         "2024-08-20 16:45": 85796, "2024-09-19 16:45": 78213}


def load_rows():
    rows = []
    for t, mw in PEAKS.items():
        t0 = datetime.fromisoformat(t)
        # the three 5-min readings ERCOT posts inside the peak interval
        for i, d in enumerate((-30, 0, 30)):
            rows.append(dict(interval_start_utc=utc(str(t0 + timedelta(minutes=5 * i))), load=mw + d))
        # a near-miss the next day, inside 1% of the peak
        rows.append(dict(interval_start_utc=utc(str(t0 + timedelta(days=1))), load=mw * 0.995))
    # May and October are hotter here and must not count toward 4CP
    rows.append(dict(interval_start_utc=utc("2024-05-30 17:00"), load=95000))
    rows.append(dict(interval_start_utc=utc("2024-10-02 16:00"), load=95000))
    rows.append(dict(interval_start_utc=utc("2024-08-21 12:00"), load=None))
    return rows


def main():
    print("4CP detection\n")
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "load_2024-05-15.json").write_text(json.dumps(load_rows()))
    strike.CACHE = tmp
    iv = strike.intervals()
    L = dict(iv)

    k = datetime(2024, 8, 20, 16, 45, tzinfo=strike.CPT)
    check("5-min load averages into its 15-min CPT settlement interval",
          k in L and L[k] == 85796, f"{L.get(k)}")
    check("rows with no load are dropped", all(v for _, v in iv))

    pk = strike.four_cp(iv)
    check("one peak per month, June to September only",
          sorted(pk) == [(2024, 6), (2024, 7), (2024, 8), (2024, 9)], f"{sorted(pk)}")
    got = {f"{t:%Y-%m-%d %H:%M}": round(v) for t, v in pk.values()}
    check("each month's peak is its highest interval", got == PEAKS)

    print("\ncurtailment economics\n")
    check("a MW of H100s earns $1,786/MWh at $2.50/GPU-hr and 1.4 kW/GPU",
          round(strike.compute_mwh()) == 1786, f"{strike.compute_mwh():.1f}")
    be = strike.P["cp_rate"] / strike.curtail_cost()
    check("break-even is 38.5 curtailment h/yr", round(be, 1) == 38.5, f"{be:.2f}")
    half = strike.curtail_cost(dict(strike.P, deferrable=0.5))
    check("deferred work costs nothing, so 50% deferrable halves the cost",
          abs(half - strike.compute_mwh() / 2) < 1e-9)

    # one year of 15-min prices: 757 intervals at $202.40, the rest at $10
    p = [202.40] * 757 + [10.0] * (35040 - 757)
    hrs, saved, lost = strike.price_rule(p, 100, 1.0)
    check("price rule counts hours at or above the threshold", hrs == 189.25, f"{hrs}")
    check("curtailing above $100 saves ~$38k and loses ~$338k",
          round(saved, -3) == 38000 and round(lost, -3) == 338000,
          f"saved {saved:,.0f}  lost {lost:,.0f}")
    hrs, _, _ = strike.price_rule(p, 1000, 1.0)
    check("no interval above the top price, no hours", hrs == 0)

    import dimmer
    check("dimmer.py prices 4CP at strike.py's rate",
          dimmer.CP_RATE == strike.P["cp_rate"], f"{dimmer.CP_RATE:,.0f}")

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print("all pinned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
