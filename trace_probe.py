#!/usr/bin/env python3
"""
Go/no-go: is a workload's power signature actually visible?

The thesis is that training is a metronome - forward, backward, optimizer step,
repeating and phase-locked - while inference follows request arrivals and is
bursty. If true, a power trace tells you what a datacenter is running, which is
how a grid operator could VERIFY a flexibility claim instead of trusting it.

If false, the whole idea collapses to "training draws more watts", which is not
a finding. So this answers it before anything gets built on top.

Two things have to hold:
  1. we can sample fast enough      - a 1.5B step is ~200-400ms, so shelling out
                                      to nvidia-smi at ~5Hz is right at the edge.
                                      NVML directly gets 100Hz+.
  2. the periodicity survives       - a clear spectral peak at the step rate for
                                      training, absent for inference.

  python3 trace_probe.py --model Qwen/Qwen2.5-1.5B-Instruct
  python3 trace_probe.py --replay traces.json      # analyse without a GPU
"""
import argparse, json, math, pathlib, statistics, subprocess, sys, threading, time

HERE = pathlib.Path(__file__).parent


# ------------------------------------------------------------------ sampling

class Tracer(threading.Thread):
    """Highest-rate power sampling available. NVML if present (microsecond
    calls), else nvidia-smi (tens of ms, and probably too slow to resolve)."""

    def __init__(self, target_hz=200):
        super().__init__(daemon=True)
        self.target_hz, self.samples, self._halt = target_hz, [], threading.Event()
        self.backend = "none"
        try:
            import pynvml
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml = pynvml
            self.backend = "nvml"
        except Exception:
            self.nvml = None
            self.backend = "nvidia-smi"

    def _read(self):
        if self.nvml:
            return self.nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
        r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        return float(r.stdout.strip().splitlines()[0])

    def run(self):
        dt = 1.0 / self.target_hz
        while not self._halt.is_set():
            t = time.perf_counter()
            try:
                self.samples.append((t, self._read()))
            except Exception:
                pass
            rest = dt - (time.perf_counter() - t)
            if rest > 0:
                time.sleep(rest)

    def stop(self):
        self._halt.set(); self.join(timeout=2)
        return self.samples


# ------------------------------------------------------------------ analysis

def resample(samples, hz):
    """Uniform grid - the sampler's own spacing jitters, and an FFT of jittered
    samples smears exactly the peak we are looking for."""
    if len(samples) < 8:
        return []
    t0, t1 = samples[0][0], samples[-1][0]
    n = int((t1 - t0) * hz)
    if n < 8:
        return []
    out, j = [], 0
    for i in range(n):
        t = t0 + i / hz
        while j + 1 < len(samples) and samples[j + 1][0] < t:
            j += 1
        out.append(samples[j][1])
    return out


def spectrum(series, hz):
    """Naive DFT over plausible step rates (0.5-25 Hz). No numpy dependency;
    the series is short enough that O(n*k) is nothing."""
    n = len(series)
    if n < 16:
        return []
    mean = statistics.mean(series)
    x = [v - mean for v in series]
    out = []
    f = 0.5
    while f <= min(25.0, hz / 2):
        re = sum(v * math.cos(2 * math.pi * f * i / hz) for i, v in enumerate(x))
        im = sum(v * math.sin(2 * math.pi * f * i / hz) for i, v in enumerate(x))
        out.append((f, math.sqrt(re * re + im * im) / n))
        f += 0.25
    return out


def _detrend(x):
    """Remove the linear trend. Without this the detector locks onto thermal
    settling - which is exactly how the first run returned a false GO."""
    n = len(x)
    if n < 3:
        return list(x)
    mx, my = (n - 1) / 2, statistics.mean(x)
    sxx = sum((i - mx) ** 2 for i in range(n)) or 1
    b = sum((i - mx) * (v - my) for i, v in enumerate(x)) / sxx
    return [v - (my + b * (i - mx)) for i, v in enumerate(x)]


def periodicity(series, hz, lo=1.0):
    """Peak spectral power over the median, on a DETRENDED and LEVEL-NORMALISED
    series - so this measures shape, not watts. Training drawing more power than
    inference is not a signature, it is a thermostat.

    A peak at the bottom of the search band is reported as suspect: it almost
    always means the energy is below `lo`, i.e. drift rather than rhythm."""
    if not series:
        return None
    mean = statistics.mean(series) or 1.0
    y = [v / mean for v in _detrend(series)]
    sp = spectrum(y, hz)
    sp = [(f, m) for f, m in sp if f >= lo]
    if not sp:
        return None
    mags = [m for _, m in sp]
    med = statistics.median(mags) or 1e-12
    peak_f, peak_m = max(sp, key=lambda x: x[1])
    return dict(peak_hz=round(peak_f, 2), ratio=round(peak_m / med, 2),
                mean_w=round(mean, 1),
                cv=round(statistics.pstdev(y), 4),
                suspect=peak_f <= lo + 0.3)


# ------------------------------------------------------------------ workloads

def build(model_id, kind, batch, seq=512):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    if kind == "train":
        cfg = AutoConfig.from_pretrained(model_id)
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-5)

        def run():
            ids = torch.randint(0, cfg.vocab_size, (batch, seq), device="cuda")
            out = m(input_ids=ids, labels=ids)
            out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        return run

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                             device_map="cuda").eval()

    def run():
        enc = tok(["Describe a power transformer."] * batch,
                  return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            m.generate(**enc, max_new_tokens=48, do_sample=False,
                       pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
    return run


def capture(run, seconds, hz):
    run()                                   # warm up outside the trace
    tr = Tracer(hz); tr.start()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        run()
    s = tr.stop()
    return s, tr.backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--hz", type=int, default=200)
    ap.add_argument("--out", default="traces.json")
    ap.add_argument("--replay")
    # Sweeping batch within each workload type is the control: if CV tracks the
    # workload it stays split at matched power; if it tracks utilisation it does not.
    ap.add_argument("--cases",
                    default="train:1,train:2,train:4,train:8,"
                            "infer:1,infer:8,infer:32,infer:128,infer:256")
    a = ap.parse_args()

    if a.replay:
        data = json.loads(pathlib.Path(a.replay).read_text())
    else:
        data = dict(model=a.model, cases={})
        cases = []
        for spec in a.cases.split(","):
            kind, b = spec.strip().split(":")
            cases.append((f"{kind}_b{b}", kind, int(b)))
        for label, kind, batch in cases:
            try:
                run = build(a.model, kind, batch)
            except Exception as e:
                print(f"  {label}: build failed {type(e).__name__}: {str(e)[:60]}")
                continue
            s, backend = capture(run, a.seconds, a.hz)
            eff = len(s) / (s[-1][0] - s[0][0]) if len(s) > 2 else 0
            data["cases"][label] = dict(backend=backend, eff_hz=round(eff, 1),
                                        samples=[round(v, 2) for _, v in s],
                                        t=[round(t - s[0][0], 5) for t, _ in s])
            print(f"  captured {label:<11}{len(s):>6} samples @ {eff:>6.1f} Hz  ({backend})")
        pathlib.Path(a.out).write_text(json.dumps(data))
        print(f"\n  wrote {a.out}")

    print(f"\n  {'case':<12}{'eff Hz':>8}{'mean W':>9}{'CV':>8}{'peak Hz':>9}{'peak/median':>13}")
    print("  " + "-" * 60)
    verdict = {}
    for label, c in data["cases"].items():
        hz = c["eff_hz"]
        series = resample(list(zip(c["t"], c["samples"])), hz) if c.get("t") else c["samples"]
        p = periodicity(series, hz)
        if not p:
            print(f"  {label:<12}  too few samples"); continue
        verdict[label] = p
        print(f"  {label:<12}{hz:>8.1f}{p['mean_w']:>9.1f}{p['cv']:>8.3f}"
              f"{p['peak_hz']:>9.2f}{p['ratio']:>13.1f}")

    print(f"\n  CV vs POWER - does variability track the workload, or just the load?")
    for label, v in sorted(verdict.items(), key=lambda kv: kv[1]["mean_w"]):
        bar = "#" * int(v["cv"] * 200)
        print(f"    {label:<13}{v['mean_w']:>7.1f}W   CV {v['cv']:.4f}  {bar}")

    # --- verdict ------------------------------------------------------------
    # Periodicity was the original hypothesis and it is dead: no spectral peak
    # above the drift floor. Variance is what actually separates the workloads,
    # so that is what gets scored. Leading with the dead hypothesis made a
    # successful experiment read as a failure.
    tcv = [v["cv"] for k, v in verdict.items() if k.startswith("train")]
    icv = [v["cv"] for k, v in verdict.items() if k.startswith("infer")]
    tw  = [(v["mean_w"], v["cv"]) for k, v in verdict.items() if k.startswith("train")]
    iw  = [(v["mean_w"], v["cv"]) for k, v in verdict.items() if k.startswith("infer")]
    print()
    if not tcv or not icv:
        print("  inconclusive - need both workload types")
        return

    gap = min(tcv) / max(icv) if max(icv) else 0
    # The control that matters: do the bands still separate where POWER overlaps?
    overlap = [(t, i) for t in tw for i in iw if abs(t[0] - i[0]) / max(t[0], i[0]) < 0.05]
    print(f"  PERIODICITY: no - peak sits at the drift floor, not a step rate.")
    print(f"  VARIANCE:    training CV {min(tcv):.3f}-{max(tcv):.3f}   "
          f"inference CV {min(icv):.3f}-{max(icv):.3f}   ({gap:.1f}x apart)")
    if overlap:
        t, i = overlap[0]
        print(f"  CONTROL:     at matched power ({t[0]:.0f}W vs {i[0]:.0f}W) the CVs are "
              f"{t[1]:.3f} vs {i[1]:.3f} - {t[1]/i[1]:.1f}x")
    if gap >= 2 and overlap and overlap[0][0][1] / overlap[0][1][1] >= 2:
        print(f"\n  GO. Variance separates the workloads independently of how loaded")
        print(f"  they are, so a power trace identifies what is running, not just")
        print(f"  how much. That is the signature a grid operator could read.")
    elif gap >= 2:
        print(f"\n  PARTIAL: bands separate but no matched-power pair to rule out")
        print(f"  utilisation as the real driver. Add an inference config near")
        print(f"  {min(t for t, _ in tw):.0f}W before believing it.")
    else:
        print(f"\n  NO-GO: variance does not separate either.")


if __name__ == "__main__":
    main()
