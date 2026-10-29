"""Tiny test, not a milestone: round a whole network's weights to 4 bits once, in several ways, and see how close
its output stays to the full-precision network.

Each weight sits between two grid values; every method picks one of the two for each weight.
  nearest        the closer one
  random         at random, without bias (milestone 3's rounding), mean of 3 draws
  row            choose so each row's running sum of errors stays near zero (sigma-delta along the row)
  row+col        choose so both the row's and the column's running error sums stay near zero (greedy)
  gptq           GPTQ: round column by column, spreading each column's error over the later columns, weighted by
                 the input covariance H = mean(x x^T) measured on sample inputs
Scales (with nearest rounding): one per matrix, one per row, one per 64 weights, one per 16 (what the milestones
use), and one per row with the scale chosen to minimise the squared error instead of absmax / 6.

Two networks:
  random   a residual MLP, 4 layers, width 256 (hidden 1024), Gaussian weights; score = cosine similarity and
           relative error of the final vector against the full-precision network, over 512 random inputs
  trained  milestone 1's model (Muon, CPU preset, seed 1), trained here and saved to report/data/m1_s1_model.pt;
           its 16 big matrices are rounded; score = validation loss on the 256 fixed windows and cosine similarity
           of the output scores (logits)

Run:  python report/round_test.py   ->  report/data/round_test.json
"""
import importlib.util
import json
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("m1", os.path.join(ROOT, "m1_muon.py"))
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


# ---------------------------------------------------------------- scales and the two candidates per weight

def scales(T, kind):
    """Scale per weight (same shape as T), so that T / scale lies in [-6, 6]."""
    a = T.abs()
    if kind == "tensor":
        s = a.max().expand_as(T)
    elif kind == "row":
        s = a.amax(1, keepdim=True).expand_as(T)
    else:
        b = int(kind.split("_")[1])
        s = a.view(T.size(0), -1, b).amax(-1, keepdim=True).expand(-1, -1, b).reshape_as(T)
    return s.clamp_min(1e-12) / 6


def neighbours(T, s):
    """The two grid values around each weight, in weight units: (lower in size, higher in size)."""
    v = (T / s).clamp(-6, 6)
    a = v.abs()
    lo_i = (torch.bucketize(a, E2M1, right=True) - 1).clamp(max=len(E2M1) - 2)
    return E2M1[lo_i] * v.sign() * s, E2M1[lo_i + 1] * v.sign() * s, a, E2M1[lo_i], E2M1[lo_i + 1]


def q_nearest(T, s):
    lo, hi, a, glo, ghi = neighbours(T, s)
    return torch.where(a - glo > ghi - a, hi, lo)


def q_random(T, s, gen):
    lo, hi, a, glo, ghi = neighbours(T, s)
    return torch.where(torch.rand(a.shape, generator=gen, dtype=a.dtype) < (a - glo) / (ghi - glo), hi, lo)


def q_row(T, s):
    lo, hi, *_ = neighbours(T, s)
    Q, r = torch.empty_like(T), torch.zeros(T.size(0), dtype=T.dtype)
    for j in range(T.size(1)):
        e1, e2 = lo[:, j] - T[:, j], hi[:, j] - T[:, j]
        pick = (r + e1).abs() <= (r + e2).abs()
        Q[:, j] = torch.where(pick, lo[:, j], hi[:, j])
        r += torch.where(pick, e1, e2)
    return Q


def q_rowcol(T, s):
    lo, hi, *_ = neighbours(T, s)
    E1, E2 = (lo - T).tolist(), (hi - T).tolist()
    out, n = T.shape
    r, c, pick = [0.0] * out, [0.0] * n, [[0] * n for _ in range(out)]
    for i in range(out):
        Ei1, Ei2, pi = E1[i], E2[i], pick[i]
        ri = r[i]
        for j in range(n):
            e1, e2, cj = Ei1[j], Ei2[j], c[j]
            if (ri + e1) ** 2 + (cj + e1) ** 2 <= (ri + e2) ** 2 + (cj + e2) ** 2:
                ri += e1; c[j] = cj + e1
            else:
                ri += e2; c[j] = cj + e2; pi[j] = 1
        r[i] = ri
    return torch.where(torch.tensor(pick, dtype=torch.bool), hi, lo)


def q_gptq(T, s, H):
    W, Q = T.clone(), torch.empty_like(T)
    n = T.size(1)
    Hd = H + 0.01 * H.diagonal().mean() * torch.eye(n, dtype=H.dtype)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
    for j in range(n):
        q = q_nearest(W[:, j:j + 1], s[:, j:j + 1])[:, 0]
        Q[:, j] = q
        W[:, j + 1:] -= ((W[:, j] - q) / U[j, j])[:, None] * U[j, j + 1:][None, :]
    return Q


def best_row_scale(T):
    """Per row, try absmax/6 times 0.5..1.0 and keep the factor with the smallest squared rounding error."""
    base = scales(T, "row")
    best, err = base, ((q_nearest(T, base) - T) ** 2).sum(1)
    for f in torch.linspace(0.5, 0.98, 25):
        s = base * f
        e = ((q_nearest(T, s) - T) ** 2).sum(1)
        better = e < err
        best = torch.where(better[:, None], s, best)
        err = torch.minimum(e, err)
    return best


METHODS = ("nearest", "random", "row", "row+col", "gptq")
SCALES = ("tensor", "row", "row_opt", "block_64", "block_16")


def quantise(T, method, scale, H=None, gen=None):
    s = best_row_scale(T) if scale == "row_opt" else scales(T, scale)
    if method == "nearest":
        return q_nearest(T, s)
    if method == "random":
        return q_random(T, s, gen)
    if method == "row":
        return q_row(T, s)
    if method == "row+col":
        return q_rowcol(T, s)
    return q_gptq(T, s, H)


def cos(a, b):
    return F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).mean().item()


# ---------------------------------------------------------------- random network

def random_net():
    g = torch.Generator().manual_seed(0)
    layers = [(torch.randn(1024, 256, generator=g, dtype=torch.float64) / 16,
               torch.randn(256, 1024, generator=g, dtype=torch.float64) / 32) for _ in range(4)]
    X = torch.randn(512, 256, generator=g, dtype=torch.float64)
    Xcal = torch.randn(2048, 256, generator=g, dtype=torch.float64)

    def run(ws, x, keep=None):
        for k, (a, b) in enumerate(ws):
            h = F.layer_norm(x, (256,))
            if keep is not None:
                keep.append(h)
            z = F.gelu(h @ a.T)
            if keep is not None:
                keep.append(z)
            x = x + z @ b.T
        return x
    ins = []
    run(layers, Xcal, ins)
    Hs = [x.T @ x / x.size(0) for x in ins]
    ref = run(layers, X)
    res = {}
    for scale in SCALES:
        for method in METHODS:
            if scale != "block_16" and method != "nearest":
                continue
            draws = 3 if method == "random" else 1
            cs, rel = [], []
            for d in range(draws):
                gen = torch.Generator().manual_seed(d)
                qs = [quantise(w, method, scale, Hs[2 * k + i], gen) for k, pair in enumerate(layers) for i, w in enumerate(pair)]
                out = run(list(zip(qs[0::2], qs[1::2])), X)
                cs.append(cos(out, ref))
                rel.append(((out - ref).norm() / ref.norm()).item())
            res[f"{method} / {scale}"] = dict(cos=sum(cs) / draws, rel_err=sum(rel) / draws)
            print(f"random   {method:8s} {scale:9s} cos {res[f'{method} / {scale}']['cos']:.5f}  rel err {sum(rel) / draws:.4f}", flush=True)
    return res


# ---------------------------------------------------------------- trained network

def trained_model():
    train, val, vocab = m1.load_data()
    cfg = m1.Config(**m1.PRESETS["cpu"], vocab_size=vocab)
    path = os.path.join(HERE, "data", "m1_s1_model.pt")
    torch.manual_seed(1)
    model = m1.GPT(cfg)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    else:  # milestone 1's training loop, seed 1, Muon lr 0.01
        gen = torch.Generator().manual_seed(1)
        opts = m1.build_optimizer(model, cfg, "muon", 0.01)
        for it in range(cfg.max_iters):
            for opt, peak in opts:
                for g in opt.param_groups:
                    g["lr"] = peak * m1.lr_at(it, cfg) / cfg.lr
            x, y = m1.get_batch(train, cfg, gen, "cpu")
            loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            for opt, _ in opts:
                opt.step()
        torch.save(model.state_dict(), path)
    return model, cfg, train, val


def trained_net():
    import contextlib
    model, cfg, train, val = trained_model()
    model.eval()
    windows = m1.fixed_windows(val, cfg)
    names = [n for n, _ in model.named_parameters() if n.endswith(m1.MUON_NAMES)]
    mods = {n + ".weight": m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n + ".weight" in names}
    H = {n: 0 for n in names}
    hooks = [mods[n].register_forward_hook(lambda m, i, o, n=n: H.__setitem__(n, H[n] + (lambda x: x.T @ x)(i[0].reshape(-1, i[0].size(-1)).double())))
             for n in names]
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for _ in range(16):
            model(m1.get_batch(train, cfg, gen, "cpu")[0])
    for h in hooks:
        h.remove()
    full = {n: p.detach().clone() for n, p in model.named_parameters()}
    xs = windows[0][:64]
    with torch.no_grad():
        ref_logits = model(xs)
    base_ce = m1.evaluate(model, windows, "cpu", contextlib.nullcontext)[0]
    print(f"trained  full precision val CE {base_ce:.4f}", flush=True)
    res = {"full": dict(val_ce=base_ce, cos=1.0)}
    for scale in SCALES:
        for method in METHODS:
            if scale != "block_16" and method != "nearest":
                continue
            draws = 3 if method == "random" else 1
            ces, cs = [], []
            for d in range(draws):
                g = torch.Generator().manual_seed(d)
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in names:
                            p.copy_(quantise(full[n].double(), method, scale, H[n], g).float())
                    ces.append(m1.evaluate(model, windows, "cpu", contextlib.nullcontext)[0])
                    cs.append(cos(model(xs), ref_logits))
            res[f"{method} / {scale}"] = dict(val_ce=sum(ces) / draws, cos=sum(cs) / draws)
            print(f"trained  {method:8s} {scale:9s} val CE {sum(ces) / draws:.4f} (+{sum(ces) / draws - base_ce:.4f})  logit cos {sum(cs) / draws:.5f}", flush=True)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(full[n])
    return res


if __name__ == "__main__":
    torch.set_num_threads(2)
    out = dict(random=random_net(), trained=trained_net())
    json.dump(out, open(os.path.join(HERE, "data", "round_test.json"), "w"), indent=1)
