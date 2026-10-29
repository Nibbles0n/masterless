"""TickerMuon: train 4-bit weights with Muon and no full-precision copy.

One file, PyTorch only. Copy it into your project.

The idea in one paragraph. Each weight matrix is stored as 4-bit values: E2M1 numbers {0, .5, 1, 1.5, 2, 3, 4, 6}
with a sign, times one scale per block of 16 weights along the input dimension (largest |value| in the block / 6).
A float copy is not kept. Instead each weight has a small "carry" (the ticker): the part of past updates that has not
yet been big enough to move the weight to the next 4-bit value. Each step

    target     = stored 4-bit weight + carry - lr * update        (update = Muon's orthogonalised momentum)
    new weight = nearest 4-bit value of the target                (block scales recomputed from the target)
    new carry  = target - new weight

The carry is stored as an integer k in -L..L (L = 2^(b-1) - 1, b = 8 by default) meaning k / (2L) of the distance to
the next 4-bit value on its own side, rounded at random so it is right on average. Momentum is stored in 8 bits
(random rounding, one scale per 128 weights).

Two ways to use it:

1. Simulated (easy): keep ordinary dense nn.Linear weights; `TickerMuon(params)` puts them on the 4-bit grid at
   construction and keeps them there. Memory is not saved (the weights are dense floats), but the maths is identical.

       opt = TickerMuon([p for n, p in model.named_parameters() if p.ndim == 2 and "layers." in n], lr=1e-3)
       loss.backward(); opt.step(); opt.zero_grad()

2. Packed (real memory savings): `convert_linears(model)` replaces nn.Linear layers by `TickerLinear`, which stores
   two 4-bit codes per byte, FP8 (E4M3) block scales under one fp32 scale per matrix (or fp32 block scales), an int8
   carry and int8 momentum with one fp32 scale per 128 weights: 2.59 bytes per weight in all (2.78 with fp32 scales).
   The forward pass rebuilds each layer's weight on the fly. In the backward pass each layer computes its weight
   gradient, applies the TickerMuon update to itself at once and frees the gradient, so no full gradient buffer ever
   exists. `optimizer.step()` is then a no-op for those layers; gradient accumulation is not possible in this mode.

       layers = convert_linears(model)
       opt = TickerMuon(layers, lr=1e-3)
       rest = AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-5)
       loss.backward()          # the 4-bit layers update themselves here
       rest.step(); rest.zero_grad()

Random draws come from a counter-based hash of (seed, layer, step, stream, element index), so they do not depend on
the device, the order of operations or chunking, and a reference implementation can reproduce them exactly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TickerMuon", "TickerLinear", "AdamW8bit", "convert_linears", "build_optimizers", "newton_schulz",
           "uniform", "ticker_bytes_per_weight"]

BLOCK = 16  # weights per 4-bit block scale, along the input dimension
MBLOCK = 128  # momentum values per int8 scale (over the flattened matrix)
FP8_MAX = 448.0  # largest E4M3 value
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
STREAM_CARRY, STREAM_MOMENTUM, STREAM_INIT, STREAM_ADAM_M, STREAM_ADAM_W = 1, 2, 3, 4, 5

# ---------------------------------------------------------------- counter-based random numbers

_M32 = 0xFFFFFFFF
_CHUNK = 1 << 22


def _mix_int(x):
    """A 32-bit integer hash (Chris Wellons' lowbias32) on a Python int."""
    x &= _M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & _M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & _M32
    return x ^ (x >> 16)


def _mix(x):
    """The same hash on an int64 tensor holding 32-bit values (low bits survive the int64 wrap-around)."""
    x = x & _M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & _M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & _M32
    return x ^ (x >> 16)


def rng_key(seed, layer, step, stream):
    h = _mix_int(_mix_int(_mix_int(_mix_int(seed) ^ layer) ^ step) ^ stream)
    return h, _mix_int(h ^ 0x9E3779B9)


def _hash_bits(key, start, end, device):
    i = torch.arange(start, end, dtype=torch.int64, device=device)
    return _mix(_mix(i ^ key[0]) ^ key[1])


def uniform(key, n, device, start=0):
    """n uniform numbers in [0, 1) (multiples of 2^-24) for elements start..start+n-1 of the stream `key`."""
    out = torch.empty(n, dtype=torch.float32, device=device)
    for a in range(0, n, _CHUNK):
        b = min(n, a + _CHUNK)
        out[a:b] = (_hash_bits(key, start + a, start + b, device) >> 8).float() * 2.0 ** -24
    return out


# ---------------------------------------------------------------- the 4-bit grid

_TABLES = {}


def _tables(device):
    t = _TABLES.get(device)
    if t is None:
        e = torch.tensor(E2M1, dtype=torch.float32)
        grid = torch.cat([-e.flip(0)[:-1], e])  # the 15 signed values -6..6
        t = dict(
            e2m1_mid=((e[1:] + e[:-1]) / 2).to(device),
            e2m1=e.to(device),
            grid_mid=((grid[1:] + grid[:-1]) / 2).to(device),
            gap_up=torch.cat([grid[1:] - grid[:-1], torch.tensor([2.0])]).to(device),  # 2 above +6
            gap_dn=torch.cat([torch.tensor([2.0]), grid[1:] - grid[:-1]]).to(device),  # 2 below -6
            lut=torch.cat([e, -e]).to(device),  # 4-bit code -> value: bit 3 is the sign, bits 0-2 index E2M1
        )
        _TABLES[device] = t
    return t


def side_gaps(x):
    """For values x on the signed grid: the distances to the next grid value up and down."""
    t = _tables(x.device)
    i = torch.bucketize(x, t["grid_mid"])
    return t["gap_up"][i], t["gap_dn"][i]


def block_scales(t, fp8):
    """Block scales of t (out, in): returns (full fp32 scales (out, in/16, 1), stored scales (out, in/16), global
    fp32 scale (0-d tensor)). fp32 mode: absmax / 6, exactly. fp8 mode (as NVFP4): absmax / 6 divided by a global
    scale that maps the largest block scale to 448, rounded to nearest E4M3 (never below its smallest step 2^-9)."""
    exact = t.view(t.size(0), -1, BLOCK).abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6
    if not fp8:
        return exact, exact.squeeze(-1), torch.ones((), device=t.device)
    g = exact.max() / FP8_MAX
    s8 = (exact / g).clamp_min(2.0 ** -9).to(torch.float8_e4m3fn)
    return s8.float() * g, s8.squeeze(-1), g


def round_to_grid(x):
    """x in scale units (any shape) -> nearest signed E2M1 value (values above 6 clip to 6)."""
    t = _tables(x.device)
    return t["e2m1"][torch.bucketize(x.abs(), t["e2m1_mid"])] * x.sign()


def encode(v):
    """Signed grid values -> 4-bit codes (uint8, 0..15). -0 is stored as +0."""
    t = _tables(v.device)
    return (torch.bucketize(v.abs(), t["e2m1_mid"]) + 8 * (v < 0)).to(torch.uint8)


def decode(c):
    return _tables(c.device)["lut"][c.int()]


def pack(c):
    """(out, in) codes 0..15 -> (out, in/2) bytes: element 2j in the low half, 2j+1 in the high half."""
    return c[:, 0::2] | (c[:, 1::2] << 4)


def unpack(p):
    return torch.stack([p & 15, p >> 4], -1).view(p.size(0), -1)


def carry_code(r, x_new, L, u):
    """The leftover r (scale units) as integers -L..L meaning k / (2L) of the gap on its own side, rounded at random
    with the uniform numbers u."""
    up, dn = side_gaps(x_new)
    c = 2 * L * r / torch.where(r > 0, up, dn)
    lo = c.floor()
    return (lo + (u.view_as(c) < c - lo)).clamp(-L, L)


def store_int8(a, u):
    """a (any shape, numel a multiple of 128) at random to int8 (-127..127) times one fp32 scale per 128 values."""
    ab = a.reshape(-1, MBLOCK)
    s = ab.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    x = ab / s
    lo = x.floor()
    q = (lo + (u.view_as(x) < x - lo)).clamp(-127, 127)
    return q.to(torch.int8).view(-1), s.squeeze(-1)


def load_int8(q, s):
    return (q.view(-1, MBLOCK).float() * s.unsqueeze(-1)).view(-1)


def newton_schulz(G, steps=5, dtype=torch.float32):
    """Approximately orthogonalise G (push its singular values towards 1), as in Keller Jordan's Muon. The size is
    normalised in fp32, the iterations run in `dtype`."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    X = (X / (X.norm() + 1e-7)).to(dtype)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


def _pad128(n):
    return (n + MBLOCK - 1) // MBLOCK * MBLOCK


def ticker_bytes_per_weight(carry_bits=8, scales="fp8"):
    """Stored bytes per weight: codes, block scales, carry, momentum and its scales."""
    return 0.5 + (1 if scales == "fp8" else 4) / BLOCK + (1 if carry_bits <= 8 else 2) + 1 + 4 / MBLOCK


# ---------------------------------------------------------------- the update, shared by both modes

def _momentum(g, q, s, beta, nesterov, key):
    """8-bit Muon momentum: returns the vector to orthogonalise and the new (q, s)."""
    n = g.numel()
    buf = load_int8(q, s)
    gf = g.reshape(-1).float()
    if buf.numel() != n:
        gf = F.pad(gf, (0, buf.numel() - n))
    buf.lerp_(gf, 1 - beta)
    u = gf.lerp(buf, beta) if nesterov else buf.clone()
    q, s = store_int8(buf, uniform(key, buf.numel(), g.device))
    return u[:n].view_as(g), q, s


def _ticker_round(x_old, s_old, k, upd, lr, wd, L, fp8, u):
    """One ticker step. x_old: grid values (out, in/16, 16); s_old: scales (out, in/16, 1); k: carry integers (out, in);
    upd: the Muon update (out, in). Returns (new grid values, full scales, stored scales, global scale, new k)."""
    up, dn = side_gaps(x_old)
    kf = k.view_as(x_old).float()
    carry = kf / (2 * L) * torch.where(kf > 0, up, dn) * s_old
    del up, dn, kf
    w = x_old * s_old
    target = w - lr * upd.view_as(w) + carry
    if wd:
        target = target - lr * wd * w
    del carry, w
    s_full, s_store, g = block_scales(target.view(target.size(0), -1), fp8)
    x_new = round_to_grid(target / s_full)
    r = (target - x_new * s_full) / s_full
    del target
    k_new = carry_code(r, x_new, L, u)
    return x_new, s_full, s_store, g, k_new


def _orthogonal_update(u, ns_steps, ns_dtype):
    return max(1, u.size(0) / u.size(1)) ** 0.5 * newton_schulz(u, ns_steps, ns_dtype).float()


def _default_ns_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


# ---------------------------------------------------------------- packed layer

class _TickerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias, trigger, mod):
        cdt = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        ctx.x_dtype, ctx.cdt, ctx.mod, ctx.has_bias = x.dtype, cdt, mod, bias is not None
        xc = x.to(cdt)
        w = mod.dense_weight(cdt)
        y = F.linear(xc, w, None if bias is None else bias.to(cdt))
        ctx.save_for_backward(xc)
        return y

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        mod = ctx.mod
        gy = gy.to(ctx.cdt)
        w = mod.dense_weight(ctx.cdt)  # the weight before this step's update: rebuilt, not saved
        gx = (gy @ w).to(ctx.x_dtype)
        del w
        gb = gy.reshape(-1, gy.size(-1)).sum(0) if ctx.has_bias else None
        if mod.trainable and mod.optimizer is not None:
            gw = gy.reshape(-1, gy.size(-1)).T @ x.reshape(-1, x.size(-1))
            mod.optimizer.update_layer(mod, gw)
        return gx, gb, None, None


class TickerLinear(nn.Module):
    """nn.Linear replacement holding packed 4-bit weights, the carry and the 8-bit momentum. No dense weight is kept;
    the forward pass rebuilds it per call, and when a TickerMuon optimizer is attached the backward pass updates the
    layer in place. Call each TickerLinear at most once per forward pass while training."""

    def __init__(self, in_features, out_features, bias=False, scales="fp8", carry_bits=8, layer_id=0, seed=0,
                 device=None):
        super().__init__()
        assert in_features % BLOCK == 0, "in_features must be a multiple of 16"
        assert scales in ("fp8", "fp32") and 2 <= carry_bits <= 16
        self.in_features, self.out_features = in_features, out_features
        self.scales, self.carry_bits, self.layer_id, self.seed, self.step = scales, carry_bits, layer_id, seed, 0
        self.trainable, self.optimizer = True, None
        n, nb, npad = out_features * in_features, in_features // BLOCK, _pad128(out_features * in_features)
        self.register_buffer("codes", torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=device))
        sdt = torch.float8_e4m3fn if scales == "fp8" else torch.float32
        self.register_buffer("scale", torch.ones(out_features, nb, dtype=sdt, device=device))
        self.register_buffer("scale_g", torch.ones((), dtype=torch.float32, device=device))
        cdt = torch.int8 if carry_bits <= 8 else torch.int16
        self.register_buffer("carry", torch.zeros(out_features, in_features, dtype=cdt, device=device))
        self.register_buffer("mom_q", torch.zeros(npad, dtype=torch.int8, device=device))
        self.register_buffer("mom_s", torch.zeros(npad // MBLOCK, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device)) if bias else None
        self._trigger = torch.empty(0, requires_grad=True)  # makes autograd call backward even if x needs no grad
        del n

    @property
    def L(self):
        return 2 ** (self.carry_bits - 1) - 1

    def get_extra_state(self):
        return dict(step=self.step, layer_id=self.layer_id, seed=self.seed)

    def set_extra_state(self, state):
        self.step, self.layer_id, self.seed = state["step"], state["layer_id"], state["seed"]

    def full_scales(self):
        return (self.scale.float() * self.scale_g).unsqueeze(-1)

    def grid_values(self):
        """The stored 4-bit values in scale units, (out, in/16, 16) fp32."""
        return decode(unpack(self.codes)).view(self.out_features, -1, BLOCK)

    def dense_weight(self, dtype=torch.float32):
        return (self.grid_values() * self.full_scales()).view(self.out_features, self.in_features).to(dtype)

    @torch.no_grad()
    def set_from_dense(self, w):
        """Round a dense weight (out, in) to nearest on the grid; the carry starts as the leftover."""
        w = w.to(self.codes.device, torch.float32)
        s_full, s_store, g = block_scales(w, self.scales == "fp8")
        x = w.view(self.out_features, -1, BLOCK) / s_full
        x_new = round_to_grid(x)
        r = (w.view_as(x) - x_new * s_full) / s_full
        u = uniform(rng_key(self.seed, self.layer_id, 0, STREAM_INIT), w.numel(), w.device)
        self.carry.copy_(carry_code(r, x_new, self.L, u).view_as(self.carry))
        self.codes.copy_(pack(encode(x_new).view(self.out_features, -1)))
        self.scale.copy_(s_store)
        self.scale_g.copy_(g)
        self.mom_q.zero_()
        self.mom_s.zero_()
        self.step = 0

    @classmethod
    def from_linear(cls, lin, scales="fp8", carry_bits=8, layer_id=0, seed=0):
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None, scales=scales, carry_bits=carry_bits,
                layer_id=layer_id, seed=seed, device=lin.weight.device)
        m.set_from_dense(lin.weight.detach())
        if lin.bias is not None:
            with torch.no_grad():
                m.bias = nn.Parameter(lin.bias.detach().clone())
        return m

    def forward(self, x):
        if torch.is_grad_enabled() and self.trainable:
            if self._trigger.device != x.device:
                self._trigger = torch.empty(0, requires_grad=True, device=x.device)
            return _TickerFn.apply(x, self.bias, self._trigger, self)
        if torch.is_grad_enabled():  # frozen: input gradients only
            return _TickerFn.apply(x, self.bias, self._trigger.detach(), self)
        cdt = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        return F.linear(x.to(cdt), self.dense_weight(cdt), None if self.bias is None else self.bias.to(cdt))

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"scales={self.scales}, carry_bits={self.carry_bits}")


# ---------------------------------------------------------------- the optimizer

class TickerMuon(torch.optim.Optimizer):
    """Muon on 4-bit weights with a quantized carry and 8-bit momentum, no full-precision copy.

    params: 2-D nn.Parameters (simulated mode: the dense weights are put on the grid now and kept there), and/or
    TickerLinear modules (packed mode: the update runs inside the backward pass; their storage format is fixed by the
    module). Param-group dicts work as usual.
    carry_bits, scales: storage of simulated weights ("fp8" or "fp32" block scales). seed: for the random rounding.
    ns_dtype: Newton-Schulz precision (default fp32 on CPU, bf16 on Ampere+, fp16 on older GPUs).
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, carry_bits=8,
                 scales="fp32", seed=0, ns_dtype=None):
        assert scales in ("fp8", "fp32") and 2 <= carry_bits <= 16
        groups = list(params)
        if groups and not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        self._modules_by_id = {}
        for gr in groups:
            ps = []
            for p in (gr["params"] if isinstance(gr["params"], (list, tuple)) else list(gr["params"])):
                if isinstance(p, TickerLinear):
                    self._modules_by_id[id(p.codes)] = p
                    ps.append(p.codes)
                else:
                    ps.append(p)
            gr["params"] = ps
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(groups, defaults)
        self.carry_bits, self.scales, self.seed, self.ns_dtype = carry_bits, scales, seed, ns_dtype
        self.update_hook = None  # optional callable(module_or_param, seconds_event_pair) for profiling
        self._group_of = {}
        lid = 0
        for gr in self.param_groups:
            for p in gr["params"]:
                self._group_of[id(p)] = gr
                m = self._modules_by_id.get(id(p))
                if m is not None:
                    m.optimizer = self
                else:
                    assert p.ndim == 2 and p.size(1) % BLOCK == 0, "simulated mode needs 2-D weights, in % 16 == 0"
                    self._init_simulated(p, lid)
                    lid += 1

    @property
    def L(self):
        return 2 ** (self.carry_bits - 1) - 1

    @torch.no_grad()
    def _init_simulated(self, p, lid):
        w = p.detach().float()
        s_full, _, _ = block_scales(w, self.scales == "fp8")
        x = w.view(p.size(0), -1, BLOCK) / s_full
        x_new = round_to_grid(x)
        r = (w.view_as(x) - x_new * s_full) / s_full
        u = uniform(rng_key(self.seed, lid, 0, STREAM_INIT), w.numel(), w.device)
        st = self.state[p]
        st["layer_id"], st["step"] = lid, 0
        st["carry"] = carry_code(r, x_new, self.L, u).to(torch.int8 if self.carry_bits <= 8 else torch.int16)
        st["carry"] = st["carry"].view_as(p)
        st["scale"] = s_full
        npad = _pad128(p.numel())
        st["mom_q"] = torch.zeros(npad, dtype=torch.int8, device=p.device)
        st["mom_s"] = torch.zeros(npad // MBLOCK, dtype=torch.float32, device=p.device)
        p.copy_((x_new * s_full).view_as(p))

    def _ns_dtype(self, device):
        return self.ns_dtype or _default_ns_dtype(device)

    @torch.no_grad()
    def update_layer(self, mod, grad):
        """Packed mode: called from TickerLinear's backward with the layer's weight gradient (out, in)."""
        gr = self._group_of[id(mod.codes)]
        if self.update_hook is not None:
            self.update_hook("start", mod)
        mod.step += 1
        u, q, s = _momentum(grad, mod.mom_q, mod.mom_s, gr["momentum"], gr["nesterov"],
                            rng_key(mod.seed, mod.layer_id, mod.step, STREAM_MOMENTUM))
        del grad
        mod.mom_q.copy_(q)
        mod.mom_s.copy_(s)
        del q, s
        upd = _orthogonal_update(u, gr["ns_steps"], self._ns_dtype(u.device))
        del u
        x_old = mod.grid_values()
        uc = uniform(rng_key(mod.seed, mod.layer_id, mod.step, STREAM_CARRY), upd.numel(), upd.device)
        x_new, _, s_store, g, k = _ticker_round(x_old, mod.full_scales(), mod.carry, upd, gr["lr"],
                                                gr["weight_decay"], mod.L, mod.scales == "fp8", uc)
        del upd, uc, x_old
        mod.codes.copy_(pack(encode(x_new).view(mod.out_features, -1)))
        mod.scale.copy_(s_store)
        mod.scale_g.copy_(g)
        mod.carry.copy_(k.view_as(mod.carry))
        if self.update_hook is not None:
            self.update_hook("end", mod)

    @torch.no_grad()
    def step(self, closure=None):
        """Simulated parameters are updated here; packed layers already updated themselves during backward."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for gr in self.param_groups:
            for p in gr["params"]:
                if id(p) in self._modules_by_id or p.grad is None:
                    continue
                st = self.state[p]
                st["step"] += 1
                lid = st["layer_id"]
                u, q, s = _momentum(p.grad, st["mom_q"], st["mom_s"], gr["momentum"], gr["nesterov"],
                                    rng_key(self.seed, lid, st["step"], STREAM_MOMENTUM))
                st["mom_q"], st["mom_s"] = q, s
                upd = _orthogonal_update(u, gr["ns_steps"], self._ns_dtype(p.device))
                s_old = st["scale"]
                x_old = (2 * p.float().view(p.size(0), -1, BLOCK) / s_old).round() / 2  # exact grid values
                uc = uniform(rng_key(self.seed, lid, st["step"], STREAM_CARRY), p.numel(), p.device)
                x_new, s_full, _, _, k = _ticker_round(x_old, s_old, st["carry"], upd, gr["lr"], gr["weight_decay"],
                                                       self.L, self.scales == "fp8", uc)
                st["carry"] = k.to(st["carry"].dtype).view_as(p)
                st["scale"] = s_full
                p.copy_((x_new * s_full).view_as(p))
        return loss


# ---------------------------------------------------------------- 8-bit AdamW for the other parameters

def _sr_to_bf16(y, key, start):
    """fp32 -> bf16 with random rounding (unbiased): add 16 random low bits to the magnitude, then truncate."""
    r = _hash_bits(key, start, start + y.numel(), y.device).view_as(y) & 0xFFFF
    bits = (y.view(torch.int32).to(torch.int64) & _M32) + r
    bits = bits & 0xFFFF0000
    bits = torch.where(bits >= 2 ** 31, bits - 2 ** 32, bits).to(torch.int32)
    return bits.view(torch.float32).to(torch.bfloat16)


class AdamW8bit(torch.optim.Optimizer):
    """AdamW with 8-bit states, for embeddings, norms and other non-matrix parameters.

    First moment: int8 with random rounding, one fp32 scale per 128 values. Second moment: its square root in FP8
    (E4M3), one fp32 scale per 128 values. bf16 parameters are updated with random rounding (no float copy); fp32
    parameters normally. Work is done in chunks of 2^24 values so no full-size fp32 temporaries are made.
    A simple reference implementation; bitsandbytes' AdamW8bit is a faster alternative."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, seed=0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.seed = seed
        lid = 0
        for gr in self.param_groups:
            for p in gr["params"]:
                st = self.state[p]
                npad = _pad128(p.numel())
                st.update(step=0, pid=lid, m_q=torch.zeros(npad, dtype=torch.int8, device=p.device),
                          m_s=torch.zeros(npad // MBLOCK, device=p.device),
                          v_q=torch.zeros(npad, dtype=torch.float8_e4m3fn, device=p.device),
                          v_s=torch.zeros(npad // MBLOCK, device=p.device))
                lid += 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        CH = 1 << 24
        for gr in self.param_groups:
            b1, b2 = gr["betas"]
            for p in gr["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                st["step"] += 1
                t = st["step"]
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                pf, gf = p.view(-1), p.grad.view(-1)
                n = pf.numel()
                km = rng_key(self.seed, st["pid"], t, STREAM_ADAM_M)
                kw = rng_key(self.seed, st["pid"], t, STREAM_ADAM_W)
                for a in range(0, n, CH):
                    b = min(n, a + CH)
                    bp = _pad128(b - a)
                    g = F.pad(gf[a:b].float(), (0, bp - (b - a)))
                    sl = slice(a // MBLOCK, a // MBLOCK + bp // MBLOCK)
                    m = load_int8(st["m_q"][a:a + bp], st["m_s"][sl])
                    rv = (st["v_q"][a:a + bp].view(-1, MBLOCK).float() * st["v_s"][sl].unsqueeze(-1)).view(-1)
                    m.lerp_(g, 1 - b1)
                    v = (rv * rv).lerp_(g * g, 1 - b2)
                    del g, rv
                    rv = v.sqrt_()
                    rs = rv.view(-1, MBLOCK).amax(-1).clamp_min(1e-30) / FP8_MAX
                    st["v_q"][a:a + bp].copy_((rv.view(-1, MBLOCK) / rs.unsqueeze(-1)).view(-1).to(torch.float8_e4m3fn))
                    st["v_s"][sl].copy_(rs)
                    # use the stored (rounded) second moment, so the step matches what is kept
                    rv = (st["v_q"][a:a + bp].view(-1, MBLOCK).float() * rs.unsqueeze(-1)).view(-1)
                    upd = (m / bc1) / (rv / math.sqrt(bc2) + gr["eps"])
                    q, s = store_int8(m, uniform(km, bp, p.device, start=a))
                    st["m_q"][a:a + bp].copy_(q)
                    st["m_s"][sl].copy_(s)
                    del m, rv, q, s
                    w = pf[a:b].float()
                    w = w * (1 - gr["lr"] * gr["weight_decay"]) - gr["lr"] * upd[: b - a]
                    del upd
                    if p.dtype == torch.bfloat16:
                        pf[a:b].copy_(_sr_to_bf16(w, kw, a))
                    else:
                        pf[a:b].copy_(w)
                    del w
        return loss


# ---------------------------------------------------------------- Hugging Face helpers

def _default_include(name, mod):
    low = name.lower()
    return ("layers." in low and not any(s in low for s in ("visual", "vision", "mtp", "lm_head", "embed"))
            and mod.in_features % BLOCK == 0)


@torch.no_grad()
def convert_linears(model, include=_default_include, scales="fp8", carry_bits=8, seed=0):
    """Replace the nn.Linear layers that `include(name, module)` accepts (default: decoder-layer linears of a
    Hugging Face model, not embeddings, head, vision tower or multi-token-prediction layers) by TickerLinear, one at a
    time, freeing each dense weight. Returns the new modules in order."""
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and include(n, m)]
    out = []
    for i, (name, lin) in enumerate(targets):
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        tl = TickerLinear.from_linear(lin, scales=scales, carry_bits=carry_bits, layer_id=i, seed=seed)
        setattr(parent, attr, tl)
        del lin
        out.append(tl)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def build_optimizers(model, ticker_layers, lr=1e-3, rest_lr=1e-5, rest="adamw8bit", weight_decay=0.0, **kw):
    """TickerMuon for the TickerLinear layers and AdamW (rest="adamw", torch's) or AdamW8bit (rest="adamw8bit") for
    every other trainable parameter. Returns (ticker_optimizer, rest_optimizer)."""
    tick = TickerMuon(ticker_layers, lr=lr, weight_decay=weight_decay, **kw)
    others = [p for p in model.parameters() if p.requires_grad]
    if rest == "adamw":
        ro = torch.optim.AdamW(others, lr=rest_lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    else:
        ro = AdamW8bit(others, lr=rest_lr, weight_decay=weight_decay)
    return tick, ro
