"""Side study B: milestone 1 (Muon, float) and milestone 3 (4-bit, no copy) at k times the CPU preset's batch, with
the same number of characters: batch 12k, and steps, warm-up and eval interval divided by k.

Side study B2: the same, with every learning rate (Muon, AdamW and its floor) multiplied by m.

Runs the committed milestone scripts unchanged; only the preset is changed from outside.
Run:  python batch_runs.py m3 4 1                  one job (arm, k, seed)
      python batch_runs.py m3 4 1 2                 the same with the learning rates times 2 (study B2)
      python batch_runs.py --list jobs.txt -j 3     a file of "arm k seed [m]" lines, j at a time
Output: runs/batch/<arm>_x<k>/, or runs/batch_lr/<arm>_x<k>_lr<m>/ with m  (the scripts' usual .json and .jsonl)
"""
import argparse
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {"m1": "m1_muon", "m3": "m3_nomaster_sr"}


def run_one(arm, k, seed, mult=None):
    spec = importlib.util.spec_from_file_location(SCRIPTS[arm], os.path.join(HERE, SCRIPTS[arm] + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.PRESETS["cpu"]
    p.update(batch_size=p["batch_size"] * k, max_iters=p["max_iters"] // k, warmup_iters=100 // k, eval_interval=250 // k)
    out = os.path.join(HERE, "runs", "batch", f"{arm}_x{k}")
    if mult is not None:
        p.update(lr=1e-3 * mult, min_lr=1e-4 * mult)
        out = os.path.join(HERE, "runs", "batch_lr", f"{arm}_x{k}_lr{mult:g}")
    muon_lr = 0.01 * (1 if mult is None else mult)
    sys.argv = [SCRIPTS[arm] + ".py", "--seed", str(seed), "--out", out, "--device", "auto", "--muon-lr", str(muon_lr)]
    mod.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", nargs="*", help="arm k seed [m]")
    ap.add_argument("--list", help="file with one 'arm k seed [m]' per line")
    ap.add_argument("-j", type=int, default=1, help="jobs at once (with --list)")
    args = ap.parse_args()
    if args.list is None:
        arm, k, seed, *mult = args.job
        run_one(arm, int(k), int(seed), float(mult[0]) if mult else None)
        return
    jobs = [l.split() for l in open(args.list) if l.strip() and not l.startswith("#")]
    os.makedirs(os.path.join(HERE, "runs", "batch", "logs"), exist_ok=True)
    running = []
    for arm, k, seed, *mult in jobs:
        while len(running) >= args.j:
            running = [p for p in running if p.poll() is None]
            if len(running) >= args.j:
                running[0].wait()
        log = open(os.path.join(HERE, "runs", "batch", "logs", f"{arm}_x{k}_s{seed}{'_lr' + mult[0] if mult else ''}.txt"), "w")
        running.append(subprocess.Popen([sys.executable, __file__, arm, k, seed, *mult], stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, OMP_NUM_THREADS="2")))
    for p in running:
        p.wait()


if __name__ == "__main__":
    main()
