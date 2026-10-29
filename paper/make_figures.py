"""Every figure of the paper, from runs/paper/*.json only. Figures whose data do not exist yet are drawn as a labelled
placeholder, so main.tex always compiles.

Run:  python paper/make_figures.py      ->  paper/figures/*.pdf
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "runs", "paper")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "figure.dpi": 150})
# one fixed colour per method, used in every figure
C = {"muon": "#c2410c", "muon_random": "#f59e0b", "sgd": "#2563eb", "adam_eco": "#059669", "adam_plain": "#7c3aed",
     "grey": "#6b7280"}


def placeholder(name, what):
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    ax.axis("off")
    ax.text(0.5, 0.5, f"{what}\n(data not available yet)", ha="center", va="center", color=C["grey"],
            transform=ax.transAxes)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print(f"{name}: placeholder")


def load(pattern):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RUNS, pattern)))]


# ---------------------------------------------------------------- M: error fed into the momentum

def fig_m():
    path = os.path.join(RUNS, "M_ns_survival.json")
    if not os.path.exists(path):
        return placeholder("m_ns_survival.pdf", "M: survival of injected error")
    d = json.load(open(path))
    last = str(d["setup"]["steps"])
    sm = d["summary"][last]
    sizes = d["setup"]["sizes"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    def series(tag, stat):
        m = [sm[f"{tag}_s{s:g}"][stat]["mean"] for s in sizes]
        e = [sm[f"{tag}_s{s:g}"][stat]["se"] for s in sizes]
        return m, e

    lab = {"muon": "Muon (through Newton–Schulz)", "sgd": "SGD-momentum; Adam with ECO's rule",
           "adam_eco": "Adam, ECO's per-weight rule", "adam_plain": "Adam, plain injection"}
    ax = axs[0]
    for tag in ("sgd", "adam_plain", "muon"):
        m, e = series(tag, "cos")
        ax.errorbar(sizes, m, yerr=e, color=C[tag], marker="o", ms=3, lw=1.2, capsize=1.5, label=lab[tag])
    rnd = [k for k in sm if k.startswith("muon_random_s")]
    if rnd:
        ax.plot([sm[k]["size"] for k in rnd], [sm[k]["cos"]["mean"] for k in rnd], ls="none", marker="x",
                color=C["muon_random"], label="Muon, random direction")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("error size / momentum size")
    ax.set_ylabel("cosine(change in update, error)")
    ax.set_title("(a) direction kept")
    ax.legend(frameon=False, loc="lower left")

    ax = axs[1]
    for tag in ("sgd", "adam_plain", "muon"):
        m, e = series(tag, "along")
        ax.errorbar(sizes, m, yerr=e, color=C[tag], marker="o", ms=3, lw=1.2, capsize=1.5, label=lab[tag])
    eco = sm.get("eco_natural_size", {})
    if eco:
        lo, hi = min(v["mean"] for v in eco.values()), max(v["mean"] for v in eco.values())
        ax.axvspan(lo, hi, color=C["grey"], alpha=0.12, lw=0)
        ax.text((lo * hi) ** 0.5, 1.3, "ECO's\nsize,\nlr 3e-3\nto 1e-4", ha="center", va="bottom", fontsize=6,
                color=C["grey"])
    if "muon_eco_lr0.001" in sm:
        ax.plot([eco["0.001"]["mean"]], [sm["muon_eco_lr0.001"]["along"]["mean"]], marker="*", ms=7, ls="none",
                color=C["muon"], label="Muon at ECO's size, lr 1e-3")
    ax.axhline(1, color=C["grey"], lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(right=max(sizes + ([hi] if eco else [])) * 1.5)
    ax.set_xlabel("error size / momentum size")
    ax.set_ylabel("share of intended correction\narriving along the error")
    ax.set_title("(b) size kept")
    star = [h for h, t in zip(*ax.get_legend_handles_labels()) if t.startswith("Muon at")]
    if star:
        ax.legend(star, ["Muon at ECO's size, lr 1e-3"], frameon=False, loc="lower left", fontsize=6)

    ax = axs[2]
    f = d["fixed_step_raw"]
    xs = [x for row in f["rms_u"] for x in row]
    ys = [y for row in f["sv_rms_ns"] for y in row]
    gs = [x for row in f["rms_g"] for x in row]
    ax.scatter(xs, ys, s=4, color=C["muon"], alpha=0.6, lw=0)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("RMS of momentum U")
    ax.set_ylabel("RMS singular value of NS(U)")
    ax.set_title("(c) fixed step size")
    ax.text(0.03, 0.05, f"{len(xs)} matrix-steps\nmomentum RMS spans {max(xs) / min(xs):.0f}x\n"
            f"gradient RMS spans {max(gs) / min(gs):.0f}x", transform=ax.transAxes, fontsize=6, color=C["grey"])
    fig.savefig(os.path.join(OUT, "m_ns_survival.pdf"))
    if os.environ.get("PNG"):
        fig.savefig(os.environ["PNG"], dpi=130)
    plt.close(fig)
    print("m_ns_survival.pdf: from", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- P1: GPT-2 small, continued training

P1_ARMS = [("muon_float", "float Muon"), ("q4_float_copy", "4-bit + float copy"), ("q4_t8_m8", "ours (ticker)"),
           ("q4_kahan_m8", "Kahan (fp32 carry)"), ("bf16_sr_m8", "bf16-SR"), ("q4_sr_m8", "naive"),
           ("eco_muon_q4", "ECO-Muon"), ("q4_float_copy_adamw", "4-bit + float copy, AdamW"),
           ("eco_adamw_q4", "ECO-AdamW")]


def fig_p1():
    path = os.path.join(RUNS, "P1_summary.json")
    if not os.path.exists(path):
        return placeholder("p1_curves.pdf", "P1: GPT-2 124M validation loss")
    curves = json.load(open(path))["curves"]
    key = {"q4_float_copy_adamw": "q4_float_copy_adamw_lr0.0001_s1", "eco_muon_q4": "eco_muon_q4_lr0.001_s1_nsfp32",
           "eco_adamw_q4": "eco_adamw_q4_lr0.0001_s1"}
    fig, ax = plt.subplots(figsize=(3.3, 2.3), constrained_layout=True)
    for i, (arm, name) in enumerate(P1_ARMS):
        ev = curves.get(key.get(arm, f"{arm}_lr0.001_s1"))
        if not ev:
            continue
        ax.plot([e[0] for e in ev], [e[1] for e in ev], lw=1.1, label=name, color=plt.cm.tab10(i))
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss (nats per token)")
    ax.legend(frameon=False, fontsize=6)
    fig.savefig(os.path.join(OUT, "p1_curves.pdf"))
    plt.close(fig)
    print("p1_curves.pdf: from", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- P2: from scratch, 45M

def fig_p2():
    path = os.path.join(RUNS, "P2_summary.json")
    if not os.path.exists(path):
        return placeholder("p2_curves.pdf", "P2: 45M from scratch")
    d = json.load(open(path))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True)
    col = {"ours": C["muon"], "copy": C["sgd"], "bf16-SR": C["grey"]}
    ax = axs[0]
    for name, curve in d["curves"].items():
        pts = [(e["step"], e["val_ce"]) for e in curve if e["step"] >= 2000]
        ax.plot(*zip(*pts), lw=1.1, color=col[name.split(" ")[0]],
                label={"ours": "ours (4-bit, 8-bit carry)", "copy": "4-bit + float copy",
                       "bf16-SR": "bf16-SR, in bf16"}[name.split(" ")[0]])
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss (nats per token)")
    ax.set_title("(a) seed 1, from step 2,000")
    ax.legend(frameon=False)
    ax = axs[1]
    gaps = [s["diff"] for s in d["seeds"]]
    ax.errorbar([1, 2, 3], gaps, yerr=[s["half"] for s in d["seeds"]], ls="none", marker="o", ms=4, capsize=2, color=C["muon"])
    ax.axhline(0.03, color=C["grey"], ls="--", lw=0.8)
    ax.text(0.55, 0.0305, "P2a line (+0.03)", va="bottom", fontsize=6, color=C["grey"])
    ax.axhline(0, color=C["grey"], lw=0.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.5, 3.9)
    ax.set_ylim(-0.005, 0.035)
    ax.set_xlabel("seed")
    ax.set_ylabel("ours $-$ float copy (final loss)")
    ax.set_title("(b) gap per seed, 95% paired interval")
    fig.savefig(os.path.join(OUT, "p2_curves.pdf"))
    plt.close(fig)
    print("p2_curves.pdf: from", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- P3: the noise model

def fig_p3():
    path = os.path.join(RUNS, "P3_summary.json")
    if not os.path.exists(path):
        return placeholder("p3_noise_model.pdf", "P3: noise ratio rho vs measured gap")
    d = json.load(open(path))
    fig, ax = plt.subplots(figsize=(3.4, 2.4), constrained_layout=True)
    mk = {0.0001: "o", 0.0003: "s", 0.001: "^", 0.003: "D"}
    bc = {4: C["muon"], 6: C["muon_random"], 8: C["sgd"], 12: C["adam_eco"]}
    for c in d["cells"]:
        ax.errorbar(c["rho"], c["gap"], yerr=2 * c["se"], marker=mk[c["lr"]], color=bc[c["bits"]], ms=4,
                    capsize=1.5, ls="none", mfc="white" if c["tied"] else bc[c["bits"]])
    ax.axvline(d["P3b"]["rho_star"], color=C["grey"], ls="--", lw=0.8)
    ax.axhline(0, color=C["grey"], lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.003)
    ax.set_ylim(-0.006, 0.1)
    ax.set_xlabel("noise ratio $\\rho$")
    ax.set_ylabel("ours $-$ float copy (loss)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=bc[b], marker="o", ls="none", ms=4, label=f"{b}-bit carry") for b in bc]
    h += [Line2D([], [], color=C["grey"], marker=mk[lr], ls="none", ms=4, mfc="white", label=f"lr {lr:g}")
          for lr in mk]
    ax.legend(handles=h, frameon=False, fontsize=5.5, ncol=2, loc="upper left")
    fig.savefig(os.path.join(OUT, "p3_noise_model.pdf"))
    plt.close(fig)
    print("p3_noise_model.pdf: from", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- P5: RL

def fig_p5():
    path = os.path.join(RUNS, "P5main_summary.json")
    if not os.path.exists(path):
        return placeholder("p5_rl.pdf", "P5: GSM8K accuracy during RL")
    d = json.load(open(path))
    col = {"a": C["sgd"], "c": C["muon"], "b": C["adam_eco"]}
    lab = {"a": "(a) bf16 + float copy", "c": "(c) ours, 4-bit", "b": "(b) 4-bit frozen + LoRA"}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), constrained_layout=True)
    ax = axs[0]
    seen = set()
    for r in d["runs"]:
        rw = r["reward"]
        k = 20
        sm = [sum(rw[i - k + 1:i + 1]) / k for i in range(k - 1, len(rw))]
        ax.plot(range(k, len(rw) + 1), sm, color=col[r["arm"]], lw=1.0, alpha=0.9,
                label=None if r["arm"] in seen else lab[r["arm"]])
        seen.add(r["arm"])
    ax.set_xlabel("RL step")
    ax.set_ylabel("training reward (20-step mean)")
    ax.set_title("(a) practice answers marked right")
    ax.legend(frameon=False)
    ax = axs[1]
    order = ["a", "b", "c"]
    rounded_seen = False
    for i, arm in enumerate(order):
        rs = [r for r in d["runs"] if r["arm"] == arm]
        for j, r in enumerate(rs):
            x = i + (j - (len(rs) - 1) / 2) * 0.12
            ax.plot([x, x], [100 * r["acc_start"], 100 * r["acc_end"]], color=col[arm], lw=1.2)
            ax.plot([x], [100 * r["acc_start"]], marker="o", mfc="white", color=col[arm], ms=4)
            ax.plot([x], [100 * r["acc_end"]], marker="o", color=col[arm], ms=4)
            if r.get("lora_rounded"):  # the same model, add-on merged and rounded to 4 bits
                ax.plot([x], [100 * r["lora_rounded"]["acc"]], marker="x", color="black", ms=4, mew=1.0)
                rounded_seen = True
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab[a] for a in order], fontsize=6)
    ax.set_ylabel("GSM8K test accuracy (%)")
    ax.set_title("(b) start (open) to end (filled), one line per seed"
                 + ("\n\u00d7: (b) with its add-on merged and rounded to 4 bits" if rounded_seen else ""))
    fig.savefig(os.path.join(OUT, "p5_rl.pdf"))
    plt.close(fig)
    print("p5_rl.pdf: from", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- P6: memory

def fig_p6():
    path = os.path.join(RUNS, "P6_memory.json")
    if not os.path.exists(path):
        return placeholder("p6_memory.pdf", "P6: peak training memory")
    d = json.load(open(path))
    rows = [("a", "bf16 + AdamW,\nfp32 copy"), ("b", "float Muon"), ("c", "bf16-SR"),
            ("c_lw", "bf16-SR,\nlayer by layer"), ("d", "ours")]
    fig, ax = plt.subplots(figsize=(3.4, 2.3), constrained_layout=True)
    for i, (k, name) in enumerate(rows):
        r = d["rows"][f"Qwen/Qwen3.5-0.8B|{k}"]
        c = C["muon"] if k == "d" else C["grey"]
        ax.bar(i, r["peak_GiB"], color=c, alpha=0.35, width=0.7)
        ax.bar(i, r["steady_state_GiB"], color=c, width=0.7)
        ax.text(i, r["peak_GiB"] + 0.2, f"{r['peak_GiB']:.2f}", ha="center", fontsize=6)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([n for _, n in rows], fontsize=6)
    ax.set_ylabel("GiB")
    ax.set_title("Qwen3.5-0.8B, one step on a T4: peak (light), weights + state (dark)", fontsize=6)
    fig.savefig(os.path.join(OUT, "p6_memory.pdf"))
    plt.close(fig)
    print("p6_memory.pdf: from", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    fig_m()
    fig_p1()
    fig_p2()
    fig_p3()
    fig_p5()
    fig_p6()
