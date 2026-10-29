"""How much learning signal sits in the tokens a model already predicts well? Pretrained GPT-2 small on fixed FineWeb
validation windows, no training. Not a milestone.

Per token: loss, chance given to the right token, and the size of the loss gradient at the output (|softmax - onehot|).
Then, per window of 1024 tokens, the exact weight gradient of every sampled matrix is taken with a loss mask:
  all            every token
  drop20/50/80   leave out the easiest 20/50/80% of the window's tokens (lowest loss)
  imp50/imp20    keep each token with chance q = min(1, c |g|), c set so the expected kept share is 50/20%, and weight the
                 kept ones by 1/q (unbiased: the expected gradient is the full one)
  rand50/rand20  keep a random 50/20%, weight 2/5 (unbiased baseline)
For each subset: how well the gradient summed over all windows points the same way as the full one (cosine, plain and
after Muon's orthogonalisation), and how noisy it is from window to window.

Run:  python report/gpt2_easy_tokens.py  ->  report/data/gpt2_easy_tokens.json
"""
import importlib.util
import json
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
torch.set_num_threads(2)
spec = importlib.util.spec_from_file_location("side_gpt2", os.path.join(ROOT, "side_gpt2.py"))
g2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)

N_WIN, LAYERS = 24, (0, 3, 6, 9, 11)
gen = torch.Generator().manual_seed(0)
model = g2.load_pretrained(g2.GPT()).train(False)
for p in model.parameters():
    p.requires_grad_(False)
mats = {f"h.{i}.{k}": p for i in LAYERS for k, p in model.h[i].named_parameters() if k.endswith(g2.MUON_NAMES)}
for p in mats.values():
    p.requires_grad_(True)
xs, ys = g2.fixed_windows(g2.load_shard("fineweb_val_000000.bin"), N_WIN, 1024)

SUBSETS = ("all", "drop20", "drop50", "drop80", "imp50", "imp20", "rand50", "rand20")


def importance(gn, share):
    """q = min(1, c gn) with mean(q) = share, by bisection on c."""
    lo, hi = 0.0, 1e6
    for _ in range(60):
        c = (lo + hi) / 2
        lo, hi = (c, hi) if torch.clamp(c * gn, max=1).mean() < share else (lo, c)
    return torch.clamp(c * gn, max=1)


def weights(loss, gn):
    n, order = loss.numel(), loss.argsort()  # easiest first
    w = {"all": torch.ones(n)}
    for f in (20, 50, 80):
        m = torch.ones(n)
        m[order[:n * f // 100]] = 0
        w[f"drop{f}"] = m
    for f in (50, 20):
        q = importance(gn, f / 100)
        keep = torch.rand(n, generator=gen) < q
        w[f"imp{f}"] = torch.where(keep, 1 / q.clamp_min(1e-12), torch.zeros(n))
    for f in (50, 20):
        w[f"rand{f}"] = (torch.rand(n, generator=gen) < f / 100).float() * 100 / f
    return w


tok = dict(loss=[], p=[], gn=[], hn=[])
gsum = {s: {k: torch.zeros_like(p) for k, p in mats.items()} for s in SUBSETS}
gsq = {s: {k: 0.0 for k in mats} for s in SUBSETS}
kept = {s: 0.0 for s in SUBSETS}
for j in range(N_WIN):
    logits = model(xs[j:j + 1])[0]
    loss = F.cross_entropy(logits, ys[j], reduction="none")
    with torch.no_grad():
        pr = logits.softmax(-1)
        py = pr.gather(1, ys[j][:, None])[:, 0]
        gn = (pr.pow(2).sum(1) - 2 * py + 1).clamp_min(0).sqrt()  # |softmax - onehot|
        g = pr.clone()
        g[torch.arange(1024), ys[j]] -= 1
        hn = (g @ model.wte.weight).norm(dim=1)  # gradient size at the final hidden state
    for a, v in zip(("loss", "p", "gn", "hn"), (loss.detach(), py, gn, hn)):
        tok[a].append(v)
    w = weights(loss.detach(), gn)
    for s in SUBSETS:
        kept[s] += (w[s] > 0).float().mean().item() / N_WIN
        grads = torch.autograd.grad((w[s] * loss).sum(), list(mats.values()), retain_graph=s != SUBSETS[-1])
        for (k, _), gk in zip(mats.items(), grads):
            gsum[s][k] += gk
            gsq[s][k] += gk.pow(2).sum().item()
    print(f"window {j + 1}/{N_WIN}  mean loss {loss.mean().item():.3f}", flush=True)

tok = {a: torch.cat(v) for a, v in tok.items()}
order = tok["loss"].argsort()
n = order.numel()
dist = dict(tokens=n, mean_loss=tok["loss"].mean().item(),
            loss_quantiles={q: tok["loss"].quantile(q).item() for q in (0.1, 0.2, 0.5, 0.8, 0.9)},
            share_p_over={t: (tok["p"] > t).float().mean().item() for t in (0.5, 0.9, 0.99)},
            easiest_share={})
for f in (10, 20, 30, 50, 80):
    e = order[:n * f // 100]
    dist["easiest_share"][f] = dict(loss=(tok["loss"][e].sum() / tok["loss"].sum()).item(),
                                    out_grad_sq=(tok["gn"][e].pow(2).sum() / tok["gn"].pow(2).sum()).item(),
                                    hidden_grad_sq=(tok["hn"][e].pow(2).sum() / tok["hn"].pow(2).sum()).item(),
                                    min_p=tok["p"][e].min().item())
print(json.dumps(dist, indent=1))

cos = lambda a, b: (a.flatten() @ b.flatten() / (a.norm() * b.norm())).item()
res = {}
for s in SUBSETS:
    r = dict(kept=kept[s], cos=[], cos_muon=[], norm_ratio=[], noise_to_signal=[])
    for k in mats:
        mean, full = gsum[s][k] / N_WIN, gsum["all"][k] / N_WIN
        r["cos"].append(cos(mean, full))
        r["cos_muon"].append(cos(g2.newton_schulz(mean), g2.newton_schulz(full)))
        r["norm_ratio"].append((mean.norm() / full.norm()).item())
        var = gsq[s][k] / N_WIN - mean.pow(2).sum().item()  # window-to-window variance (total over entries)
        sig = mean.pow(2).sum().item() - var / (N_WIN - 1)  # unbiased |true mean|^2
        r["noise_to_signal"].append(var / max(sig, 1e-30))
    res[s] = {a: (sum(v) / len(v) if isinstance(v, list) else v) for a, v in r.items()}
    print(f"{s:7s} kept {res[s]['kept']:.2f}  cos {res[s]['cos']:.3f}  cos after Muon {res[s]['cos_muon']:.3f}  "
          f"norm {res[s]['norm_ratio']:.3f}  noise/signal (one window) {res[s]['noise_to_signal']:.1f}")

json.dump(dict(distribution=dist, subsets=res, layers=LAYERS, windows=N_WIN),
          open(os.path.join(HERE, "data", "gpt2_easy_tokens.json"), "w"), indent=1)
