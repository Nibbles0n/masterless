"""Tests for ticker_muon.py. Run:  python test_ticker_muon.py   (or pytest)

The main test checks the packed implementation (two 4-bit codes per byte, FP8 or fp32 block scales, int8 carry, int8
momentum) against a plain dense reference of the same maths written separately below, given the same random draws.
Random draws come from ticker_muon.uniform (a counter-based hash), which is the one thing the two share.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ticker_muon as tm  # noqa: E402

torch.set_num_threads(2)

# ---------------------------------------------------------------- dense reference (written from the description)

E = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
SIGNED = torch.cat([-E.flip(0)[:-1], E])


def ref_scales(t, fp8):
    a = t.view(t.size(0), -1, 16).abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6
    if not fp8:
        return a
    g = a.max() / 448
    return (a / g).clamp_min(2.0 ** -9).to(torch.float8_e4m3fn).float() * g


def ref_round(x):
    """Nearest signed E2M1 value by a 15-way search (clipping at +-6)."""
    i = (x.unsqueeze(-1) - SIGNED).abs().argmin(-1)
    return SIGNED[i]


def ref_gaps(x):
    i = (x.unsqueeze(-1) - SIGNED).abs().argmin(-1)
    nxt = torch.cat([SIGNED[1:], torch.tensor([8.0])])
    prv = torch.cat([torch.tensor([-8.0]), SIGNED[:-1]])
    return nxt[i] - SIGNED[i], SIGNED[i] - prv[i]


def ref_carry(r, x_new, L, u):
    up, dn = ref_gaps(x_new)
    c = 2 * L * r / torch.where(r > 0, up, dn)
    lo = c.floor()
    return (lo + (u.view_as(c) < c - lo)).clamp(-L, L)


def ref_store8(a, u):
    ab = a.reshape(-1, 128)
    s = ab.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    x = ab / s
    lo = x.floor()
    return ((lo + (u.view_as(x) < x - lo)).clamp(-127, 127) * s).reshape(-1)


def ref_ns(G):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(5):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class RefTicker:
    """Dense fp32 weights on the grid, dense float carry integers, dense float momentum."""

    def __init__(self, w, L, fp8, seed, lid, lr, beta=0.95):
        self.L, self.fp8, self.seed, self.lid, self.lr, self.beta, self.t = L, fp8, seed, lid, lr, beta, 0
        s = ref_scales(w, fp8)
        x = w.view(w.size(0), -1, 16) / s
        xn = ref_round(x)
        r = (w.view_as(x) - xn * s) / s
        self.k = ref_carry(r, xn, L, tm.uniform(tm.rng_key(seed, lid, 0, tm.STREAM_INIT), w.numel(), w.device))
        self.x, self.s = xn, s
        n = w.numel()
        self.npad = (n + 127) // 128 * 128
        self.buf = torch.zeros(self.npad)

    def weight(self):
        return (self.x * self.s).view(self.x.size(0), -1)

    def step(self, grad):
        self.t += 1
        n = grad.numel()
        g = F.pad(grad.reshape(-1).float(), (0, self.npad - n))
        self.buf.lerp_(g, 1 - self.beta)
        u = g.lerp(self.buf, self.beta)
        self.buf = ref_store8(self.buf, tm.uniform(tm.rng_key(self.seed, self.lid, self.t, tm.STREAM_MOMENTUM),
                                                   self.npad, g.device))
        u = u[:n].view_as(grad)
        upd = max(1, u.size(0) / u.size(1)) ** 0.5 * ref_ns(u)
        up, dn = ref_gaps(self.x)
        carry = self.k / (2 * self.L) * torch.where(self.k > 0, up, dn) * self.s
        w = self.x * self.s
        target = w - self.lr * upd.view_as(w) + carry
        s = ref_scales(target.view(target.size(0), -1), self.fp8)
        xn = ref_round(target / s)
        r = (target - xn * s) / s
        self.k = ref_carry(r, xn, self.L, tm.uniform(tm.rng_key(self.seed, self.lid, self.t, tm.STREAM_CARRY),
                                                     n, g.device))
        self.x, self.s = xn, s
        return r  # leftover before clamping, in scale units


def on_grid(w, s):
    x = w.view(w.size(0), -1, 16) / s
    return bool(((x.unsqueeze(-1) - SIGNED).abs().amin(-1) < 1e-5).all())


# ---------------------------------------------------------------- tests

def test_pack_roundtrip():
    g = torch.Generator().manual_seed(0)
    c = torch.randint(0, 16, (37, 64), generator=g, dtype=torch.uint8)
    p = tm.pack(c)
    assert p.shape == (37, 32) and p.dtype == torch.uint8
    assert torch.equal(tm.unpack(p), c)
    v = SIGNED.repeat(3)
    assert torch.equal(tm.decode(tm.encode(v)), v)
    assert torch.equal(tm.decode(tm.encode(-0.0 * torch.ones(4))), torch.zeros(4))


def test_uniform_chunk_independent():
    k = tm.rng_key(1, 2, 3, 4)
    a = tm.uniform(k, 5000, torch.device("cpu"))
    b = torch.cat([tm.uniform(k, 1234, torch.device("cpu")), tm.uniform(k, 5000 - 1234, torch.device("cpu"), 1234)])
    assert torch.equal(a, b) and 0 <= a.min() and a.max() < 1 and abs(a.mean() - 0.5) < 0.02
    assert not torch.equal(a, tm.uniform(tm.rng_key(1, 2, 3, 5), 5000, torch.device("cpu")))


def _compare(shape, fp8, bits, steps=25, seed=3):
    torch.manual_seed(seed)
    w0 = torch.randn(shape) * 0.02
    lin = nn.Linear(shape[1], shape[0], bias=False)
    with torch.no_grad():
        lin.weight.copy_(w0)
    L = 2 ** (bits - 1) - 1
    mod = tm.TickerLinear.from_linear(lin, scales="fp8" if fp8 else "fp32", carry_bits=bits, layer_id=5, seed=seed)
    lr = 3e-3
    opt = tm.TickerMuon([mod], lr=lr, ns_dtype=torch.float32)
    ref = RefTicker(w0, L, fp8, seed, 5, lr)
    assert torch.equal(mod.dense_weight(), ref.weight())
    assert torch.equal(mod.carry.float(), ref.k.view(shape))
    max_r = 0.0
    for t in range(steps):
        g = torch.randn(shape) * (1 + t % 3)
        opt.update_layer(mod, g.clone())
        r = ref.step(g)
        up, dn = ref_gaps(ref.x)
        if fp8:  # with FP8 scales the block maximum can clip above 6; check the others
            r = torch.where(ref.x.abs() < 6, r, torch.zeros_like(r))
        max_r = max(max_r, float((r.abs() / torch.where(r > 0, up, dn)).max()))
        assert torch.equal(mod.dense_weight(), ref.weight()), f"weights differ at step {t + 1}"
        assert torch.equal(mod.carry.float(), ref.k.view(shape)), f"carry differs at step {t + 1}"
        assert torch.equal(tm.load_int8(mod.mom_q, mod.mom_s), ref.buf), f"momentum differs at step {t + 1}"
        assert on_grid(mod.dense_weight(), mod.full_scales())
    assert max_r <= 0.5 + 1e-6, max_r  # the leftover is at most half a gap on its side
    assert int(mod.carry.abs().max()) <= L


def test_packed_matches_reference():
    for shape in ((48, 64), (40, 80), (16, 256), (96, 32)):
        for fp8, bits in ((True, 8), (False, 8), (True, 4), (False, 12)):
            _compare(shape, fp8, bits)


def test_simulated_matches_reference():
    for fp8 in (False, True):
        torch.manual_seed(0)
        w0 = torch.randn(32, 64) * 0.05
        p = nn.Parameter(w0.clone())
        opt = tm.TickerMuon([p], lr=1e-2, scales="fp8" if fp8 else "fp32", carry_bits=8, seed=7,
                            ns_dtype=torch.float32)
        ref = RefTicker(w0, 127, fp8, 7, 0, 1e-2)
        assert torch.equal(p.detach(), ref.weight())
        for _ in range(20):
            g = torch.randn(32, 64)
            p.grad = g.clone()
            opt.step()
            ref.step(g)
            assert torch.equal(p.detach(), ref.weight())
            assert torch.equal(opt.state[p]["carry"].float(), ref.k.view(32, 64))


def test_layerwise_backward():
    """Training through loss.backward(): each layer must get the true gradient of the model as it was before the step
    (its input gradient uses the pre-update weight), with or without activation checkpointing."""
    for use_ckpt in (False, True):
        torch.manual_seed(1)
        lins = [nn.Linear(64, 96, bias=True), nn.Linear(96, 32, bias=False)]
        mods = [tm.TickerLinear.from_linear(l, scales="fp8", layer_id=i, seed=2) for i, l in enumerate(lins)]
        refs = [RefTicker(l.weight.detach().clone(), 127, True, 2, i, 1e-2) for i, l in enumerate(lins)]
        dense = [r.weight().clone().requires_grad_(True) for r in refs]
        b0 = mods[0].bias.detach().clone().requires_grad_(True)
        opt = tm.TickerMuon(mods, lr=1e-2, ns_dtype=torch.float32)
        seen = {}
        orig = opt.update_layer

        def spy(mod, grad):
            seen[mod.layer_id] = grad.clone()
            orig(mod, grad)
        opt.update_layer = spy
        x = torch.randn(4, 5, 64, requires_grad=True)

        def net(x):
            return mods[1](F.gelu(mods[0](x)))
        y = checkpoint(net, x, use_reentrant=False) if use_ckpt else net(x)
        y.square().mean().backward()
        y_ref = F.linear(F.gelu(F.linear(x, dense[0], b0)), dense[1])
        x2 = x.detach().clone().requires_grad_(True)
        y_ref = F.linear(F.gelu(F.linear(x2, dense[0], b0)), dense[1])
        y_ref.square().mean().backward()
        assert torch.allclose(x.grad, x2.grad, atol=1e-6, rtol=1e-4)
        assert torch.allclose(mods[0].bias.grad, b0.grad, atol=1e-6, rtol=1e-4)
        for i in range(2):
            assert torch.allclose(seen[i], dense[i].grad, atol=1e-6, rtol=1e-4)
            refs[i].step(seen[i])
            assert torch.equal(mods[i].dense_weight(), refs[i].weight())
            assert mods[i].step == 1


def test_frozen_and_eval():
    torch.manual_seed(0)
    m = tm.TickerLinear.from_linear(nn.Linear(32, 16))
    w = m.dense_weight().clone()
    x = torch.randn(3, 32, requires_grad=True)
    m.trainable = False
    m(x).sum().backward()  # frozen: input gradient only, no update
    assert x.grad is not None and torch.equal(m.dense_weight(), w)
    with torch.no_grad():
        assert torch.allclose(m(x), F.linear(x, w, m.bias))


def test_state_dict_roundtrip():
    torch.manual_seed(0)
    m = tm.TickerLinear.from_linear(nn.Linear(32, 16, bias=False), layer_id=3, seed=9)
    opt = tm.TickerMuon([m], lr=1e-2, ns_dtype=torch.float32)
    opt.update_layer(m, torch.randn(16, 32))
    m2 = tm.TickerLinear(32, 16)
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2.dense_weight(), m.dense_weight()) and m2.step == 1 and m2.layer_id == 3 and m2.seed == 9


def test_adamw8bit():
    torch.manual_seed(0)
    target = torch.randn(1000)
    for dt in (torch.float32, torch.bfloat16):
        p = nn.Parameter(torch.zeros(1000, dtype=dt))
        opt = tm.AdamW8bit([p], lr=0.05)
        for _ in range(300):
            p.grad = (p.float() - target).to(dt)
            opt.step()
        assert float((p.detach().float() - target).abs().mean()) < 0.05, dt
    y = torch.full((200000,), 1.0 + 2 ** -10)  # halfway-ish between two bf16 values: random rounding is unbiased
    r = tm._sr_to_bf16(y, tm.rng_key(0, 0, 0, 0), 0).float()
    assert abs(float(r.mean()) - (1.0 + 2 ** -10)) < 2e-5


def test_bytes_per_weight():
    assert abs(tm.ticker_bytes_per_weight(8, "fp8") - 2.59375) < 1e-9
    m = tm.TickerLinear(1024, 256)
    nbytes = sum(b.numel() * b.element_size() for b in m.buffers())
    assert abs(nbytes / (1024 * 256) - 2.59375) < 1e-3


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
