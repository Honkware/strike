#!/usr/bin/env python3
"""
DIMMER - an AI datacenter is not a switch, it is a dial.

Every flexibility analysis (including strike.py) models curtailment
as binary: full load or zero load. For an AI datacenter that is simply wrong.
GPUs have a power-performance frontier, and it is steeply favourable:

  A100 power cap 250W -> 175W : -30% power for +6.7% inference time   [measured, lit]
  H100 FP8 vs BF16            : +30-40% tok/s at identical power draw [measured, lit]
  batch to the utilisation knee (256-512 on H100): energy/token plateaus

So the choice is not "$1,786/MWh of revenue or nothing." It is a continuum, and
the leverage ratio (power shed / revenue lost) is roughly 4.5x at shallow depths.

This re-runs the 4CP economics on the dial instead of the switch.
"""
import strike

CP_RATE   = strike.P["cp_rate"]   # $/MW-yr, one figure for every script
REV_MWH   = strike.compute_mwh()  # $/MWh of revenue riding on facility power

# (power shed, revenue lost) operating points on the frontier.
# shallow points are lit-anchored; deep points need measuring on real hardware.
FRONTIER = [
    (0.00, 0.000, "baseline"),
    (0.15, 0.020, "mild clock cap"),
    (0.30, 0.067, "A100 250->175W [measured]"),
    (0.45, 0.130, "clock cap + FP8 swap"),
    (0.60, 0.250, "+ INT4 + batch reshape"),
    (0.80, 0.500, "deep dim, quality degraded"),
    (1.00, 1.000, "full curtailment (the switch)"),
]

def net(shed, rev_loss, hours):
    """$/MW-yr from dimming during the four 4CP intervals, plus the hours you
    dimmed on false alarms. 4CP charge scales with demand during the interval."""
    saved = shed * CP_RATE
    cost  = hours * REV_MWH * rev_loss
    return saved - cost

def breakeven_hours(shed, rev_loss):
    if rev_loss <= 0: return float("inf")
    return shed * CP_RATE / (REV_MWH * rev_loss)

def load_measured(path="frontier_7b.json"):
    """Replace the literature-anchored FRONTIER with points measured by sweep.py.

    Until this is run the deep rows are extrapolations of mine, and the whole
    argument rests on them - so the measurement is the project, not a detail.
    """
    import json, pathlib as _p
    f = _p.Path(path)
    if not f.exists():
        return None
    import sweep
    data = json.loads(f.read_text())
    pts = [x for x in sweep.frontier(data) if x["throughput_lost"] > 1e-6]
    if not pts:
        return []
    # take the best (highest-leverage) batch size at each cap
    best = {}
    for x in pts:
        k = x["cap_pct"]
        if k not in best or (x["leverage"] or 0) > (best[k]["leverage"] or 0):
            best[k] = x
    return [(x["power_shed"], x["throughput_lost"], f"measured @ cap {c}%, batch {x['batch']}")
            for c, x in sorted(best.items(), key=lambda kv: -kv[0])]


if __name__ == "__main__":
    import sys
    if "--measured" in sys.argv:
        m = load_measured(sys.argv[sys.argv.index("--measured") + 1]
                          if len(sys.argv) > sys.argv.index("--measured") + 1 else "frontier_7b.json")
        if m:
            FRONTIER = [(0.0, 0.0, "baseline")] + m + [(1.0, 1.0, "full curtailment (the switch)")]
            print("USING MEASURED FRONTIER\n")
        else:
            print("no measured frontier found - falling back to literature anchors\n")
    print(f"revenue at risk: ${REV_MWH:,.0f}/MWh   4CP rate: ${CP_RATE:,.0f}/MW-yr\n")
    print("THE FRONTIER - how much flexibility costs, per unit delivered\n")
    print(f"  {'operating point':<30}{'power shed':>11}{'rev lost':>10}{'leverage':>10}{'$/MW-shed-hr':>14}")
    for shed, rl, lbl in FRONTIER:
        if shed == 0: continue
        lev = shed / rl if rl else float("inf")
        cost_per = REV_MWH * rl / shed
        print(f"  {lbl:<30}{shed:>10.0%}{rl:>10.1%}{lev:>9.1f}x{cost_per:>14,.0f}")

    print(f"\n\nBREAK-EVEN CURTAILMENT HOURS (how many false alarms you can afford)\n")
    print(f"  {'operating point':<30}{'break-even hrs/yr':>20}")
    for shed, rl, lbl in FRONTIER:
        if shed == 0: continue
        print(f"  {lbl:<30}{breakeven_hours(shed,rl):>20.0f}")
    # Derived, not asserted. The literature-anchored version of this file
    # claimed the dial beat the switch; the measured frontier says otherwise,
    # and a hardcoded conclusion that contradicts its own data is worse than
    # no conclusion at all.
    sw = breakeven_hours(1.0, 1.0)
    best_lbl, best_be = None, -1
    for shed, rl, lbl in FRONTIER:
        if shed in (0.0, 1.0) or rl <= 0:
            continue
        be = breakeven_hours(shed, rl)
        if be > best_be:
            best_lbl, best_be = lbl, be
    print(f"\n  The best no-hindsight rule in strike.py needs ~115 hrs/yr to catch all four peaks.")
    print(f"  Full curtailment breaks even at {sw:.0f} hrs -> it loses.")
    if best_lbl is None:
        print("  No partial operating point measured.")
    elif best_be > sw:
        print(f"  Best dial setting breaks even at {best_be:.0f} hrs ({best_lbl})")
        print(f"  -> the dial buys {best_be - sw:.0f} hrs more headroom than the switch.")
    else:
        print(f"  Best dial setting breaks even at {best_be:.0f} hrs ({best_lbl})")
        print(f"  -> WORSE than the switch. Every measured operating point sheds")
        print(f"     power more expensively than simply turning the machine off.\n")

    print("NET $/MW-yr, forecaster quality x dim depth\n")
    hrs = [115, 80, 60, 40, 30, 20]
    print(f"  {'operating point':<30}" + "".join(f"{h:>9}h" for h in hrs))
    print("  " + "-" * (30 + 10 * len(hrs)))
    for shed, rl, lbl in FRONTIER:
        if shed == 0: continue
        print(f"  {lbl:<30}" + "".join(f"{net(shed,rl,h):>10,.0f}" for h in hrs))
    lev = [(shed / rl, lbl) for shed, rl, lbl in FRONTIER if rl > 0 and shed not in (0.0,)]
    if lev and max(lev)[0] <= 1.0 + 1e-9:
        print(f"\n  Every measured leverage is <= 1.0x, and the switch is 1.0x by")
        print(f"  definition - so no dial setting beats turning it off. On this")
        print(f"  hardware and workload the dial does not exist.")
    else:
        print(f"\n  (bottom row is full curtailment - the strategy everyone assumes)")
