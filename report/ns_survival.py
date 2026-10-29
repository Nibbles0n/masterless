"""Step M of PAPER_PLAN.md: how much of an error fed into the momentum reaches the update? No training.

Pretrained GPT-2 small, its real gradients on FineWeb windows, for the 48 block matrices that Muon trains (attention
c_attn and c_proj, MLP c_fc and c_proj; about 85M weights). Over STEPS steps (WIN windows of 1024 tokens each) we build:
  Muon:          buf <- lerp(buf, g, 1 - 0.95),  u = lerp(g, buf, 0.95)   (Nesterov, as in side_gpt2.py)
  SGD-momentum:  the same u, applied as it is (update = u)
  Adam:          m = Muon's buf (beta1 0.95), v <- lerp(v, g^2, 0.05), update = m_hat / (sqrt(v_hat) + 1e-8)

Then a known error is added to the momentum. Its direction is each matrix's actual 4-bit rounding leftover
e = w - quantize_rtn(w) (E2M1, block-16 absmax scales); its size is swept relative to the momentum,
s = |delta| / |u|. ECO (arXiv 2601.22101) adds alpha * e to the momentum, alpha = (1/eta)(1 - 1/beta) for SGD-momentum
and the same times Adam's per-weight factor (sqrt(v_hat) + eps) for Adam; with a linear optimizer this puts exactly
(1 - beta) * e into the next step and e into the sum of all later steps. The "intended correction" in the update is
therefore the injected delta passed through linearly, at the optimizer's own gain kappa = |update| / |momentum|.
For each optimizer and size we measure, per matrix:
  cos    cosine between (update with error - update without) and delta
  ratio  |update with error - update without| / (kappa |delta|)            (1 = the full size arrives)
  along  the part of the change along delta, as a share of the intended correction (= cos * ratio)
  keep   (Muon) cosine between the updates with and without the error: how much of the gradient's direction is left
Adam is measured twice: with ECO's per-weight rule (delta = c (sqrt(v_hat)+eps) * e, which Adam divides out again) and
with the plain SGD rule (delta = c e, which Adam rescales weight by weight).

Muon's fixed step size: at every step, the RMS of u and of NS(u) for each matrix. Newton-Schulz first divides by the
norm, so a gradient 1000x bigger gives the same update; the RMS singular value of NS(u) shows how fixed it is.

Also, at the last step (one SVD per matrix, u = P S Q^T): the share of delta's energy in the part of P^T delta Q that is
symmetric. To first order that part only changes the singular values, and an orthogonaliser maps every singular value
to about 1, so this share is thrown away whatever its size.

CPU (2 threads, about 20-30 minutes, about 4 GB of memory) or a GPU (a few minutes).  Run:  python report/ns_survival.py
Writes runs/paper/M_ns_survival.json.
"""
import importlib.util
import json
import math
import os
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
torch.set_num_threads(2)
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # same maths either way; fp32 throughout
spec = importlib.util.spec_from_file_location("side_gpt2", os.path.join(ROOT, "side_gpt2.py"))
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)

STEPS, WIN, BETA = int(os.environ.get("M_STEPS", 10)), 2, 0.95  # Muon / SGD-momentum
B1, B2, EPS = BETA, 0.95, 1e-8  # Adam (first moment shared with Muon's buffer to save memory)
SIZES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
INJECT_AT = (STEPS // 2, STEPS)  # steps at which the error is injected
LRS = (1e-4, 3e-4, 1e-3, 3e-3)  # Muon learning rates of P3, for the size ECO's rule would give
OUT = os.environ.get("M_OUT", os.path.join(ROOT, "runs", "paper", "M_ns_survival.json"))
t0 = time.time()

model = g2.load_pretrained(g2.GPT()).train(False).to(DEV)
for p in model.parameters():
    p.requires_grad_(False)
mats = {f"h.{i}.{k.removesuffix('.weight')}": p for i in range(g2.N_LAYER)
        for k, p in model.h[i].named_parameters() if k.endswith(g2.MUON_NAMES)}
for p in mats.values():
    p.requires_grad_(True)
if os.environ.get("M_MATS"):  # smoke test: the first few matrices only
    mats = dict(list(mats.items())[:int(os.environ["M_MATS"])])
names = list(mats)
kinds = [n.split(".", 2)[2] for n in names]
xs, ys = (t.to(DEV) for t in g2.fixed_windows(g2.load_shard("fineweb_val_000000.bin"), STEPS * WIN, 1024))
err_of = lambda n: mats[n].detach() - g2.quantize_rtn(mats[n].detach())
shape_scale = {n: max(1, p.size(0) / p.size(1)) ** 0.5 for n, p in mats.items()}

rms = lambda a: a.pow(2).mean().sqrt().item()
cos = lambda a, b: (a.flatten() @ b.flatten() / (a.norm() * b.norm())).item()


def survival(change, delta, kappa, base=None):
    """base: the update without error; then also the cosine between the updates with and without error (how much
    of the gradient's own direction is left)."""
    c, r = cos(change, delta), (change.norm() / (kappa * delta.norm())).item()
    out = dict(cos=c, ratio=r, along=c * r)
    if base is not None:
        out["keep"] = cos(base + change, base)
    return out


buf = {n: torch.zeros_like(p) for n, p in mats.items()}  # Muon's momentum; also Adam's first moment (beta1 = 0.95)
av = {n: torch.zeros_like(p) for n, p in mats.items()}  # Adam's second moment
fixed = dict(rms_g=[], rms_u=[], rms_ns=[], rms_update=[], sv_rms_ns=[])  # [step][matrix]
inject = {s: {} for s in INJECT_AT}
gen = torch.Generator(device=DEV).manual_seed(1)


def put(step, key, size, item):
    inject[step].setdefault(key, dict(size=size, per_matrix=[]))["per_matrix"].append(item)


def measure(step, n, u, base):
    """All injection measurements for one matrix (one matrix at a time keeps the memory small)."""
    e = err_of(n)
    kappa = (base.norm() / u.norm()).item()
    unorm = u.norm()
    for s in SIZES:
        d = e * (s * unorm / e.norm())
        put(step, f"muon_s{s:g}", s, survival(g2.newton_schulz(u + d) - base, d, kappa, base))
        put(step, f"sgd_s{s:g}", s, survival((u + d) - u, d, 1.0))  # SGD-momentum applies u as it is
    # Adam: linear in its first moment, so one size is as good as any; the sweep is kept for the table
    m, vhat = buf[n], (av[n] / (1 - B2 ** step)).sqrt() + EPS
    mhat_scale = 1 / (1 - B1 ** step)
    up = m * mhat_scale / vhat
    kappa_a = (up.norm() / m.norm()).item()
    for s in SIZES:
        d = e * (s * m.norm() / e.norm())
        put(step, f"adam_plain_s{s:g}", s, survival((m + d) * mhat_scale / vhat - up, d, kappa_a))
        c = s * m.norm() / (e * vhat).norm()
        d = c * e * vhat  # ECO's rule: the per-weight factor (sqrt(v_hat) + eps) is undone by Adam
        put(step, f"adam_eco_s{s:g}", s, survival((m + d) * mhat_scale / vhat - up, e, (c * mhat_scale).item()))
    del vhat, up
    if step != STEPS:
        return
    for s in (0.1, 1.0):  # control: a random direction instead of the rounding leftover
        r = torch.randn(u.shape, generator=gen, device=DEV)
        d = r * (s * unorm / r.norm())
        put(step, f"muon_random_s{s:g}", s, survival(g2.newton_schulz(u + d) - base, d, kappa, base))
    # the size ECO's rule gives: delta in u = beta^2 alpha e, alpha = (1/eta_eff)(1 - 1/beta), eta_eff = eta kappa
    # (Muon's step per unit of momentum, milestone 4's adaptation), so |delta| / |u| = (1 - beta) |e| / (eta |step|),
    # with the shape scale included in the step
    eco = inject[step].setdefault("eco_natural_size", {f"{lr:g}": [] for lr in LRS})
    for lr in LRS:
        eco[f"{lr:g}"].append(((1 - BETA) * e.norm() / (lr * shape_scale[n] * base.norm())).item())
    s = eco["0.001"][-1]
    d = e * (s * unorm / e.norm())
    put(step, "muon_eco_lr0.001", "ECO rule at lr 1e-3", survival(g2.newton_schulz(u + d) - base, d, kappa, base))
    # symmetric (singular-value-only) share of the leftover, from one SVD
    un, en = (u.T, e.T) if u.size(0) > u.size(1) else (u, e)
    P, S, Qh = torch.linalg.svd(un, full_matrices=False)
    dt = P.T @ en @ Qh.T
    inject[step].setdefault("symmetric_share", []).append(((dt + dt.T) / 2).pow(2).sum().item() / en.pow(2).sum().item())
    inject[step].setdefault("in_row_space_share", []).append(dt.pow(2).sum().item() / en.pow(2).sum().item())


u0 = None
for step in range(1, STEPS + 1):
    grads = {n: torch.zeros_like(p) for n, p in mats.items()}
    loss_sum = 0.0
    for j in range(WIN):
        k = (step - 1) * WIN + j
        loss = F.cross_entropy(model(xs[k:k + 1])[0], ys[k]) / WIN
        for n, gk in zip(names, torch.autograd.grad(loss, list(mats.values()))):
            grads[n] += gk
        del gk
        loss_sum += loss.item()
    row = {a: [] for a in fixed}
    for i, n in enumerate(names):
        g = grads.pop(n)
        buf[n].lerp_(g, 1 - BETA)
        u = g.lerp(buf[n], BETA)
        av[n].lerp_(g * g, 1 - B2)
        ns = g2.newton_schulz(u)
        row["rms_g"].append(rms(g))
        row["rms_u"].append(rms(u))
        row["rms_ns"].append(rms(ns))
        row["rms_update"].append(shape_scale[n] * rms(ns))
        row["sv_rms_ns"].append(rms(ns) * math.sqrt(max(ns.shape)))  # RMS singular value of NS(u)
        if step in INJECT_AT:
            measure(step, n, u, ns)
            if i % 12 == 11:
                print(f"  injected into {i + 1} matrices  {time.time() - t0:.0f}s", flush=True)
        if step == STEPS and i == 0:
            u0 = u.clone()
        del g, u, ns
    for a in fixed:
        fixed[a].append(row[a])
    print(f"step {step}  loss {loss_sum:.3f}  {time.time() - t0:.0f}s", flush=True)
    del grads

# scale invariance of Newton-Schulz, on one matrix: a 1000x bigger or smaller momentum gives the same update
scale_check = {f"{c:g}": rms(g2.newton_schulz(u0 * c)) / rms(g2.newton_schulz(u0)) for c in (1e-3, 1e3)}


def stats(v):
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return dict(mean=m, se=sd / math.sqrt(len(v)), sd=sd, min=min(v), max=max(v))


summary = {}
for step, res in inject.items():
    sm = {}
    for key, r in res.items():
        if isinstance(r, dict) and "per_matrix" in r:
            sm[key] = {a: stats([o[a] for o in r["per_matrix"]]) for a in ("cos", "ratio", "along", "keep")
                       if a in r["per_matrix"][0]}
            sm[key]["size"] = r["size"]
    if "symmetric_share" in res:
        sm["symmetric_share"] = stats(res["symmetric_share"])
        sm["in_row_space_share"] = stats(res["in_row_space_share"])
        sm["eco_natural_size"] = {lr: stats(v) for lr, v in res["eco_natural_size"].items()}
    summary[step] = sm

flat = lambda a: [x for row in fixed[a] for x in row]
cv = lambda v: stats(v)["sd"] / stats(v)["mean"]
fixed_summary = dict(
    rms_u=stats(flat("rms_u")), rms_ns=stats(flat("rms_ns")), rms_update=stats(flat("rms_update")),
    sv_rms_ns=stats(flat("sv_rms_ns")), rms_g=stats(flat("rms_g")),
    max_over_min=dict(rms_g=max(flat("rms_g")) / min(flat("rms_g")), rms_u=max(flat("rms_u")) / min(flat("rms_u")),
                      rms_update=max(flat("rms_update")) / min(flat("rms_update")),
                      sv_rms_ns=max(flat("sv_rms_ns")) / min(flat("sv_rms_ns"))),
    cv=dict(rms_u=cv(flat("rms_u")), rms_update=cv(flat("rms_update")), sv_rms_ns=cv(flat("sv_rms_ns"))),
    ns_scale_invariance=scale_check)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(setup=dict(model="GPT-2 small (124M), pretrained, float32", matrices=names, kinds=kinds,
                          steps=STEPS, windows_per_step=WIN, tokens_per_window=1024, muon_beta=BETA, nesterov=True,
                          ns_steps=5, adam=dict(beta1=B1, beta2=B2, eps=EPS), sizes=list(SIZES),
                          inject_at=list(INJECT_AT), error="w - quantize_rtn(w), E2M1 block-16 absmax/6",
                          data="fineweb_val_000000.bin, fixed evenly spaced windows", device=DEV,
                          gpu=torch.cuda.get_device_name() if DEV == "cuda" else None, matmul_tf32=False,
                          seconds=time.time() - t0),
               summary=summary, fixed_step=fixed_summary, fixed_step_raw=fixed,
               per_matrix={str(k): v for k, v in inject.items()}), open(OUT, "w"), indent=1)
print(json.dumps(dict(summary=summary[STEPS], fixed=fixed_summary), indent=1))
print(f"wrote {OUT}  {time.time() - t0:.0f}s")
