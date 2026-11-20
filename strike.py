#!/usr/bin/env python3
"""
STRIKE - is load flexibility worth it for an AI datacenter in ERCOT?

Everything is measured against real data, and every assumption is a parameter.
The whole thesis reduces to one number: HOURS OF CURTAILMENT NEEDED TO CATCH
ALL FOUR 4CP INTERVALS. Get under ~32 h/yr and flexibility pays at any
workload mix; miss it and you destroy value.

  python3 strike.py fetch     pull + cache ERCOT prices and load (needs API key)
  python3 strike.py energy    does curtailing on PRICE pay?      (answer: no)
  python3 strike.py fourcp    find the real 4CP intervals
  python3 strike.py backtest  no-hindsight curtailment rules
  python3 strike.py surface   net $/MW-yr vs forecaster quality x deferability
  python3 strike.py all
"""
import json, os, sys, time, pathlib, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta, timezone

CPT = timezone(timedelta(hours=-5))          # ERCOT settles on Central Prevailing Time
CACHE = pathlib.Path(__file__).parent / "cache"
API = "https://api.gridstatus.io/v1/datasets"

# ---- parameters. every one of these is an assumption; change them and re-run ----
P = dict(
    cp_rate      = 56510.0,   # $/MW-yr  CenterPoint 4CP transmission. VERIFY against current tariff.
    gpu_hr       = 2.50,      # $/GPU-hr on-demand H100
    kw_per_gpu   = 1.40,      # facility kW per GPU incl. PUE and overhead
    deferrable   = 0.0,       # share of load that can be postponed rather than lost
)
def compute_mwh(p=P):
    """$ of revenue riding on one MWh of facility power."""
    return p["gpu_hr"] * (1000.0 / p["kw_per_gpu"])
def curtail_cost(p=P):
    """$/MWh actually lost when curtailed - deferred work is not destroyed work."""
    return compute_mwh(p) * (1 - p["deferrable"])


# ------------------------------------------------------------------ data

def _get(ds, params, tries=6):
    k = os.environ.get("GRIDSTATUS_API_KEY") or _dotenv()
    u = f"{API}/{ds}/query?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            r = urllib.request.Request(u, headers={"x-api-key": k})
            return json.loads(urllib.request.urlopen(r, timeout=180).read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503) and i < tries - 1:
                time.sleep(8 * (i + 1)); continue
            raise

def _dotenv():
    f = pathlib.Path(__file__).parent / ".env"
    for line in f.read_text().splitlines():
        if line.startswith("GRIDSTATUS_API_KEY="):
            return line.split("=", 1)[1].strip()
    sys.exit("no GRIDSTATUS_API_KEY in env or .env")

def _page(ds, a, b, **extra):
    rows, cur = [], None
    while True:
        p = dict(start_time=a, end_time=b, limit=50000, **extra)
        if cur: p["cursor"] = cur
        d = _get(ds, p); rows += d["data"]
        m = d.get("meta") or {}
        cur = m.get("cursor")
        if not cur or not m.get("hasNextPage"): break
        time.sleep(1)
    return rows

def fetch():
    CACHE.mkdir(exist_ok=True)
    for a, b in [("2024-08-01","2025-02-01"),("2025-02-01","2025-08-01"),
                 ("2025-08-01","2026-02-01"),("2026-02-01","2026-08-05")]:
        f = CACHE / f"lz_houston_{a}.json"
        if not f.exists():
            f.write_text(json.dumps(_page("ercot_spp_real_time_15_min", a+"T00:00:00Z", b+"T00:00:00Z",
                filter_column="location", filter_value="LZ_HOUSTON", filter_operator="=")))
        print(f"  prices {a}: {len(json.loads(f.read_text())):,}")
    for a, b in [("2024-05-15","2024-10-15"),("2025-05-15","2025-10-15"),("2026-05-15","2026-08-05")]:
        f = CACHE / f"load_{a}.json"
        if not f.exists():
            f.write_text(json.dumps(_page("ercot_load", a+"T00:00:00Z", b+"T00:00:00Z")))
        print(f"  load   {a}: {len(json.loads(f.read_text())):,}")

def prices():
    r = []
    for f in sorted(CACHE.glob("lz_houston_*.json")): r += json.loads(f.read_text())
    return sorted([x for x in r if x.get("spp") is not None], key=lambda x: x["interval_start_utc"])

def intervals():
    """ERCOT 5-min load -> 15-min settlement intervals in CPT."""
    r = []
    for f in sorted(CACHE.glob("load_*.json")): r += json.loads(f.read_text())
    b = {}
    for x in r:
        if not x.get("load"): continue
        t = datetime.fromisoformat(x["interval_start_utc"]).astimezone(CPT)
        k = t.replace(minute=(t.minute // 15) * 15, second=0, microsecond=0)
        b.setdefault(k, []).append(x["load"])
    return sorted((k, sum(v) / len(v)) for k, v in b.items())

def four_cp(iv):
    out = {}
    for yr in sorted({k.year for k, _ in iv}):
        for mo in (6, 7, 8, 9):
            c = [(k, v) for k, v in iv if k.year == yr and k.month == mo]
            if c: out[(yr, mo)] = max(c, key=lambda z: z[1])
    return out


# ------------------------------------------------------------------ analyses

def energy():
    p = [x["spp"] for x in prices()]; n = len(p); yrs = n * 0.25 / 8760
    print(f"\nLZ_HOUSTON {n:,} intervals, {yrs:.2f} yrs   mean ${sum(p)/n:.2f}/MWh  max ${max(p):,.0f}")
    print(f"compute is worth ${compute_mwh():,.0f}/MWh - {compute_mwh()/(sum(p)/n):.0f}x the average power price\n")
    print(f"  {'curtail when price >=':<26}{'hrs/yr':>8}{'saved':>11}{'lost':>11}{'net $/MW-yr':>14}")
    for thr in (100, 250, 500, 1000, 1750, 2500):
        hrs = sum(1 for x in p if x >= thr) * 0.25 / yrs
        saved = sum(x for x in p if x >= thr) * 0.25 / yrs
        lost = hrs * curtail_cost()
        print(f"  ${thr:<25,}{hrs:>8.1f}{saved:>11,.0f}{lost:>11,.0f}{saved-lost:>14,.0f}")
    print("\n  -> price-driven curtailment is value-destroying for high-value compute.")

def fourcp():
    iv = intervals()
    print(f"\n{len(iv):,} settlement intervals. Actual 4CP (monthly system peak):\n")
    for (yr, mo), (k, v) in sorted(four_cp(iv).items()):
        c = [x for kk, x in iv if kk.year == yr and kk.month == mo]
        near = sum(1 for x in c if x >= v * 0.99)
        print(f"  {yr}-{mo:02d}  {k:%b %d %H:%M} CPT  {v:>7,.0f} MW   {near:>3} intervals within 1%")

def backtest():
    iv = intervals(); L = dict(iv); pk = four_cp(iv)
    be = P["cp_rate"] / curtail_cost() if curtail_cost() else float("inf")
    print(f"\nBREAK-EVEN: at most {be:.1f} curtailment hrs/yr at deferrable={P['deferrable']:.0%}\n")
    print("Rule A: curtail if load 1h ago > T, weekday 14-19h CPT, Jun-Sep. No hindsight.")
    print(f"  {'T (MW)':>9}{'hrs/yr':>9}{'caught':>10}{'net $/MW-yr':>14}")
    for T in (76000, 78000, 80000, 82000):
        h, c, n = 0, 0, 0
        for (yr, mo), (p0, _) in pk.items():
            if yr == 2026 and mo == 9: continue
            n += 1
            for k, _ in iv:
                if k.year != yr or k.month != mo or k.weekday() >= 5 or not (14 <= k.hour < 19): continue
                prev = L.get(k - timedelta(hours=1))
                if prev is not None and prev >= T:
                    h += 0.25
                    if k == p0: c += 1
        y = n / 4
        print(f"  {T:>9,}{h/y:>9.1f}{c:>6}/{n:<3}{c/n*P['cp_rate'] - h/y*curtail_cost():>14,.0f}")
    print("\n  -> every threshold loses. The gap between ~117h needed and ~32h affordable")
    print("     IS the product: better forecasting closes it.")

def surface():
    print(f"\nNET $/MW-yr capturing all four 4CP intervals (4CP rate ${P['cp_rate']:,.0f}/MW-yr)\n")
    hrs = [117, 80, 60, 40, 30, 20, 12, 6]
    print(f"  {'deferrable':>11}" + "".join(f"{h:>9}h" for h in hrs))
    print("  " + "-" * (11 + 10 * len(hrs)))
    for d in (0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 1.0):
        cost = compute_mwh() * (1 - d)
        print(f"  {d*100:>10.0f}%" + "".join(f"{P['cp_rate'] - h*cost:>10,.0f}" for h in hrs))
    print("\nBREAK-EVEN deferrable share by forecaster quality:")
    for h in hrs:
        d = 1 - P["cp_rate"] / (h * compute_mwh())
        print(f"  {h:>4}h/yr -> {max(0,d)*100:>5.1f}% deferrable" + ("   (pays at ANY mix)" if d <= 0 else ""))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "fetch": fetch()
    elif cmd == "all":
        for f in (energy, fourcp, backtest, surface): f()
    else: globals()[cmd]()
