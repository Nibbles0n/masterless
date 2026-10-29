"""Side study: the milestone arms trained 5x longer (10,000 steps, CPU-preset model, same data and windows).

Runs the committed milestone scripts unchanged; only the step count and the learning rate are set from outside.
  adamw     m0_baby_gpt.py              lr = AdamW peak learning rate (min lr = lr / 10)
  muon      m1_muon.py                  lr = Muon peak learning rate (AdamW parts stay at 1e-3)
  m3        m3_nomaster_sr.py           lr = Muon peak learning rate
  eco_adam  m4_eco.py --arm eco_adam    lr = Adam peak learning rate (min lr = lr / 10)

Run:  python long_runs.py adamw 0.001 1            one job
      python long_runs.py --list jobs.txt -j 2     a file of "arm lr seed" lines, j at a time
Output: runs/long/<arm>_lr<lr>/  (the scripts' usual .json and .jsonl)
"""
import argparse
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {"adamw": "m0_baby_gpt", "muon": "m1_muon", "m3": "m3_nomaster_sr", "eco_adam": "m4_eco"}


def run_one(arm, lr, seed, iters, device):
    spec = importlib.util.spec_from_file_location(SCRIPTS[arm], os.path.join(HERE, SCRIPTS[arm] + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = os.path.join(HERE, "runs", "long", f"{arm}_lr{lr:g}")
    argv = ["--seed", str(seed), "--max-iters", str(iters), "--out", out, "--device", device]
    if arm in ("adamw", "eco_adam"):
        mod.PRESETS["cpu"].update(lr=lr, min_lr=lr / 10)
    else:
        argv += ["--muon-lr", str(lr)]
    if arm == "eco_adam":
        argv += ["--arm", "eco_adam"]
    sys.argv = [SCRIPTS[arm] + ".py"] + argv
    mod.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", nargs="*", help="arm lr seed")
    ap.add_argument("--list", help="file with one 'arm lr seed' per line")
    ap.add_argument("-j", type=int, default=1, help="jobs at once (with --list)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.list is None:
        arm, lr, seed = args.job
        run_one(arm, float(lr), int(seed), args.iters, args.device)
        return
    jobs = [l.split() for l in open(args.list) if l.strip() and not l.startswith("#")]
    os.makedirs(os.path.join(HERE, "runs", "long", "logs"), exist_ok=True)
    running = []
    for arm, lr, seed in jobs:
        while len(running) >= args.j:
            running = [p for p in running if p.poll() is None]
            if len(running) >= args.j:
                running[0].wait()
        log = open(os.path.join(HERE, "runs", "long", "logs", f"{arm}_lr{lr}_s{seed}.txt"), "w")
        cmd = [sys.executable, __file__, arm, lr, seed, "--iters", str(args.iters), "--device", args.device]
        running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
    for p in running:
        p.wait()


if __name__ == "__main__":
    main()
