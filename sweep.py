#!/usr/bin/env python3
"""
Measure the GPU power/performance frontier - the artifact DIMMER rests on.

The claim is that an AI datacenter is a dial, not a switch: you can shed a large
fraction of power for a small fraction of throughput. The literature says an A100
capped 250W -> 175W loses 6.7% throughput for 30% power. There is also a credible
counter-result ("The Illusion of Power Capping in LLM Decode") arguing the lever
is weak in decode-bound serving, because the GPU already draws under the cap.

Both cannot be true for the same workload. This settles it on real hardware.

  python3 sweep.py --model <hf-id> --caps 100,70,55,40 --batches 1,8,32,128
  python3 sweep.py --replay results.json          # re-analyse without a GPU

Outputs one row per operating point: power cap, batch, measured watts, tokens/sec,
and joules/token. Feeds straight into dimmer.py as measured points rather than
literature estimates.

Runs anywhere with nvidia-smi and torch. Power capping needs privileges; without
them the sweep falls back to SM clock locking, and if that also fails it still
measures batch-size effects at default power. It says which mode it used.
"""
import argparse, json, statistics, subprocess, sys, threading, time, pathlib


# ------------------------------------------------------------------ GPU control

def smi(*args, check=False):
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
        return r.returncode == 0, (r.stdout or r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        if check:
            sys.exit(f"nvidia-smi unavailable: {e}")
        return False, str(e)


def gpu_info():
    ok, out = smi("--query-gpu=name,power.max_limit,power.default_limit,power.min_limit",
                  "--format=csv,noheader,nounits", check=True)
    name, pmax, pdef, pmin = [x.strip() for x in out.splitlines()[0].split(",")]
    return dict(name=name, max_w=float(pmax), default_w=float(pdef), min_w=float(pmin))


def set_power(watts):
    ok, msg = smi("-pl", str(int(watts)))
    return ok, msg


def lock_clock(mhz):
    ok, msg = smi("-lgc", f"0,{int(mhz)}")
    return ok, msg


def reset_limits(default_w):
    smi("-pl", str(int(default_w)))
    smi("-rgc")


class PowerSampler(threading.Thread):
    """Sample power.draw continuously - the *measured* draw is the number that
    matters, not the cap. A cap you never reach sheds nothing."""
    def __init__(self, hz=5):
        super().__init__(daemon=True)
        # NB: not `self._stop` - threading.Thread already defines _stop() and
        # join() calls it internally. Shadowing it raises
        # "TypeError: 'Event' object is not callable" from inside join().
        self.hz, self.samples, self._halt = hz, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            ok, out = smi("--query-gpu=power.draw", "--format=csv,noheader,nounits")
            if ok:
                try:
                    self.samples.append(float(out.splitlines()[0]))
                except (ValueError, IndexError):
                    pass
            time.sleep(1.0 / self.hz)

    def stop(self):
        self._halt.set(); self.join(timeout=2)
        s = self.samples
        return dict(mean=statistics.mean(s) if s else 0.0,
                    peak=max(s) if s else 0.0, n=len(s))


# ------------------------------------------------------------------ workload

def make_workload(model_id, dtype, max_new):
    """Returns fn(batch) -> tokens generated.

    `dtype` is a WORKLOAD lever, not a hardware one, and that turns out to be the
    point: RunPod containers cannot run `nvidia-smi -pl` or `-lgc` (both return
    Insufficient Permissions), and neither can most colo tenants. The levers an
    operator actually controls are precision, batch size and which models run -
    so those are what the sweep varies, and power is the measured OUTCOME.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map="cuda")
    if dtype in ("int8", "int4"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=(dtype == "int8"), load_in_4bit=(dtype == "int4"),
            bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["torch_dtype"] = dict(fp16=torch.float16, bf16=torch.bfloat16,
                                 fp32=torch.float32)[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    prompt = "Summarise the operating characteristics of a large power transformer."

    def run(batch):
        enc = tok([prompt] * batch, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return int((out.shape[1] - enc["input_ids"].shape[1]) * batch)

    return run


def make_training(model_id, dtype, seq_len=1024):
    """Sustained training: forward + backward + optimizer step, repeatedly.

    This is the load factor the whole nameplate argument turns on. Training is
    compute-bound on large matmuls and runs synchronously across the cluster,
    which is why it is assumed to sit near nameplate - but "assumed" is the
    problem, so we measure it. Returns fn(batch) -> tokens processed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    torch_dtype = dict(fp16=torch.float16, bf16=torch.bfloat16,
                       fp32=torch.float32).get(dtype, torch.bfloat16)
    cfg = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).cuda()
    model.gradient_checkpointing_disable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    vocab = cfg.vocab_size

    def run(batch):
        ids = torch.randint(0, vocab, (batch, seq_len), device="cuda")
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return batch * seq_len

    return run


# ------------------------------------------------------------------ sweep

def measure(run, batch, warmup=1, reps=3):
    for _ in range(warmup):
        run(batch)
    sampler = PowerSampler(); sampler.start()
    t0 = time.perf_counter(); toks = 0
    for _ in range(reps):
        toks += run(batch)
    dt = time.perf_counter() - t0
    p = sampler.stop()
    return dict(batch=batch, seconds=round(dt, 3), tokens=toks,
                tok_per_s=round(toks / dt, 2),
                watts_mean=round(p["mean"], 1), watts_peak=round(p["peak"], 1),
                joules_per_token=round(p["mean"] * dt / toks, 4) if toks else None,
                power_samples=p["n"])


def sweep(args):
    info = gpu_info()
    print(f"GPU: {info['name']}   default {info['default_w']:.0f}W   "
          f"range {info['min_w']:.0f}-{info['max_w']:.0f}W\n")
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]

    mode = "power-cap"
    ok, msg = set_power(info["default_w"])
    if not ok:
        print(f"  power capping unavailable ({msg.splitlines()[0][:70]})")
        ok2, _ = lock_clock(2000)
        mode = "clock-lock" if ok2 else "default-only"
        print(f"  falling back to: {mode}\n")

    rows, failures = [], []
    save_to = getattr(args, "out", None)
    caps = [c for c in args.caps if c] or [100]
    if mode == "default-only":
        caps = [100]
        print("  no hardware power control - sweeping WORKLOAD levers only\n")
    try:
      for dt in dtypes:
        print(f"  --- precision {dt} ---")
        try:
            run = (make_training(args.model, dt, args.seq_len)
                   if args.workload == "train"
                   else make_workload(args.model, dt, args.max_new))
        except Exception as e:
            print(f"    load failed ({type(e).__name__}: {str(e)[:60]}) - skipping")
            continue
        for pct in caps:
            target = info["default_w"] * pct / 100.0
            if mode == "power-cap":
                good, m = set_power(max(target, info["min_w"]))
                if not good:
                    print(f"  skip {pct}% ({m.splitlines()[0][:50]})"); continue
            elif mode == "clock-lock":
                lock_clock(int(2000 * pct / 100.0))
            elif pct != 100:
                continue                       # nothing to vary
            time.sleep(2)                      # let the limit settle
            for b in args.batches:
                # One bad configuration must not cost the whole sweep. int8 on
                # H100/SM90 raises "cublasLt ran into an error!" in bnb 0.43.3,
                # and letting that propagate discarded every fp16 measurement
                # already taken.
                try:
                    r = measure(run, b, reps=args.reps)
                except Exception as e:
                    failures.append(dict(dtype=dt, batch=b, cap_pct=pct,
                                         error=f"{type(e).__name__}: {str(e)[:90]}"))
                    print(f"    {dt:<5} cap {pct:>3}%  batch {b:>4}  FAILED "
                          f"{type(e).__name__}: {str(e)[:50]}")
                    continue
                r.update(cap_pct=pct, cap_w=round(target, 1), mode=mode, dtype=dt)
                rows.append(r)
                print(f"    {dt:<5} cap {pct:>3}%  batch {b:>4}  "
                      f"{r['watts_mean']:>6.1f}W  {r['tok_per_s']:>8.1f} tok/s  "
                      f"{r['joules_per_token']} J/tok")
                # save after every point, so a crash later in the sweep keeps
                # the measurements already taken
                if save_to:
                    pathlib.Path(save_to).write_text(json.dumps(
                        dict(gpu=info, model=args.model, dtypes=dtypes, mode=mode,
                             rows=rows, failures=failures), indent=1))
    finally:
        reset_limits(info["default_w"])
    return dict(gpu=info, model=args.model, dtypes=dtypes, mode=mode,
                workload=getattr(args, "workload", "infer"),
                rows=rows, failures=failures)


# ------------------------------------------------------------------ analysis

def frontier(data):
    """Turn measurements into the (power shed, throughput lost) pairs DIMMER needs."""
    rows = data["rows"]
    if not rows:
        return []
    # The reference point is the configuration drawing the most power - that is
    # the datacenter running flat out. Every other point is a dial setting.
    base = max(rows, key=lambda r: r["watts_mean"])
    out = []
    for r in rows:
        shed = 1 - r["watts_mean"] / base["watts_mean"] if base["watts_mean"] else 0
        lost = 1 - r["tok_per_s"] / base["tok_per_s"] if base["tok_per_s"] else 0
        out.append(dict(batch=r["batch"], dtype=r.get("dtype", "?"), cap_pct=r["cap_pct"],
                        power_shed=round(shed, 4), throughput_lost=round(lost, 4),
                        leverage=round(shed / lost, 2) if lost > 1e-6 else None))
    return sorted(out, key=lambda x: x["power_shed"])


def report(data):
    print(f"\nMEASURED FRONTIER   ({data['gpu']['name']}, {data['mode']})\n")
    print(f"  {'dtype':>7}{'batch':>7}{'cap':>6}{'power shed':>12}{'tput lost':>11}{'leverage':>10}")
    for f in frontier(data):
        lev = f"{f['leverage']:.1f}x" if f["leverage"] else "-"
        print(f"  {f['dtype']:>7}{f['batch']:>7}{f['cap_pct']:>5}%{f['power_shed']:>11.1%}"
              f"{f['throughput_lost']:>11.1%}{lev:>10}")
    for f in data.get("failures", []):
        print(f"  ! {f['dtype']} batch {f['batch']}: {f['error']}")
    if data.get("failures"):
        print()
    if not data.get("rows"):
        print("  NO MEASUREMENTS - every configuration failed to run.")
        print("  This is a broken sweep, not a null result. Check the job log:")
        print("  a load failure and a genuinely flat frontier look nothing alike.")
        return
    best = [f for f in frontier(data) if f["leverage"]]
    if best:
        top = max(best, key=lambda f: f["leverage"])
        print(f"\n  best leverage: {top['leverage']:.1f}x at batch {top['batch']}, "
              f"cap {top['cap_pct']}% - sheds {top['power_shed']:.0%} of power "
              f"for {top['throughput_lost']:.0%} of throughput")
        print(f"  (the switch is 1.0x by definition: 100% power for 100% throughput)")
    else:
        print("\n  no leverage measured - if power never dropped, the cap was never")
        print("  binding, which is the counter-paper's decode-bound result.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtypes", default="fp16",
                    help="workload lever: comma list of fp16,bf16,fp32,int8,int4")
    ap.add_argument("--caps", type=lambda s: [int(x) for x in s.split(",")],
                    default=[100, 85, 70, 55, 40])
    ap.add_argument("--batches", type=lambda s: [int(x) for x in s.split(",")],
                    default=[1, 8, 32])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--workload", default="infer", choices=["infer", "train"])
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="frontier.json")
    ap.add_argument("--replay", help="analyse a previous run instead of measuring")
    a = ap.parse_args()

    if a.replay:
        data = json.loads(pathlib.Path(a.replay).read_text())
    else:
        data = sweep(a)
        pathlib.Path(a.out).write_text(json.dumps(data, indent=1))
        print(f"\n  wrote {a.out}")
    report(data)


if __name__ == "__main__":
    main()
