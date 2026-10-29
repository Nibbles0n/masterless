# TickerMuon

Train 4-bit weights with the Muon optimizer, **without a full-precision copy of the weights**.

The usual way to train a 4-bit model keeps a 16- or 32-bit "master" copy of every weight. The optimizer updates the
copy, and the 4-bit weights are re-made from it each step. TickerMuon keeps only the 4-bit weights plus a small
per-weight *carry* (the "ticker"): the part of past updates that has not yet been big enough to move the weight to the
next 4-bit value. The weights you train are the 4-bit weights you run.

One file, `ticker_muon.py`, PyTorch 2.1 or newer, nothing else to install.

## How it works

Every weight matrix is stored as 4-bit numbers. Each number is one of 0, 0.5, 1, 1.5, 2, 3, 4 or 6 (the E2M1 format),
with a sign. Each block of 16 neighbouring weights along a row shares one scale (largest value in the block / 6).
Each step, for each weight:

```
target     = stored 4-bit weight + carry - learning rate * update     (update = Muon's orthogonalised momentum)
new weight = nearest 4-bit value to the target                        (block scales re-made from the target)
new carry  = target - new weight
```

The carry is stored as a whole number k from -127 to 127 (8 bits) that means k/254 of the distance to the next 4-bit
value on its own side. It is rounded at random, so it is right on average. Muon's momentum is stored in 8 bits too
(random rounding, one scale per 128 values). Muon itself is standard: Nesterov momentum 0.95, five Newton-Schulz steps
with coefficients (3.4445, -4.7750, 2.0315), update scaled by max(1, rows/columns)^0.5.

Memory per weight of the converted matrices:

| part | bytes |
|---|---|
| 4-bit codes (two per byte) | 0.5 |
| block scales, FP8 (E4M3) with one fp32 scale per matrix (or fp32: 0.25) | 0.0625 |
| carry, int8 | 1 |
| momentum, int8 + one fp32 scale per 128 | 1.03 |
| **total** | **2.59** (2.78 with fp32 scales) |

For comparison, bf16 weights with an fp32 master copy and AdamW need 14 bytes per weight (plus 2 for gradients), and
bf16 weights with random rounding and 8-bit Muon momentum need 3.03.

Measured on one T4 (`p6_memory.py`, `runs/paper/P6_RESULT.md`): a full-parameter training step of Qwen3.5-0.8B
peaks at 3.74 GiB in the packed mode, against 11.74 GiB for bf16 + AdamW with an fp32 master copy; Qwen3.5-2B fits
in 8.34 GiB.

## Use it in ten minutes

Copy `ticker_muon.py` next to your training script.

### Option 1: simulated (easiest; same maths, no memory saving)

Your model keeps ordinary `nn.Linear` layers. TickerMuon rounds the chosen weights onto the 4-bit grid when it is
created and keeps them there. Use any optimizer for the other parameters.

```python
import torch
from ticker_muon import TickerMuon

matrices = [p for n, p in model.named_parameters() if p.ndim == 2 and "layers." in n]
others = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "layers." in n)]
opt = TickerMuon(matrices, lr=1e-3)               # puts the matrices on the 4-bit grid now
opt_rest = torch.optim.AdamW(others, lr=1e-5)

for batch in data:
    loss = model(**batch).loss
    loss.backward()
    opt.step(); opt_rest.step()
    opt.zero_grad(); opt_rest.zero_grad()
```

### Option 2: packed (real memory saving)

`convert_linears` swaps the decoder-layer `nn.Linear`s of a Hugging Face model for `TickerLinear`, which stores packed
4-bit codes and no dense weight. In the backward pass each layer computes its own weight gradient, updates itself at
once and throws the gradient away, so there is never a full gradient buffer. Use activation checkpointing to keep
activations small too.

```python
import torch
from transformers import AutoModelForCausalLM
from ticker_muon import convert_linears, build_optimizers

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
layers = convert_linears(model)                   # decoder linears -> 4-bit TickerLinear (FP8 scales, 8-bit carry)
opt, opt_rest = build_optimizers(model, layers, lr=1e-4, rest_lr=1e-5, rest="adamw8bit")   # or rest="adamw"

for batch in data:
    loss = model(**batch).loss
    loss.backward()            # the 4-bit layers update themselves here, one by one
    opt_rest.step(); opt_rest.zero_grad()   # embeddings, norms and other small parameters
    opt.step()                 # no-op for packed layers; keeps the loop shape (and LR schedulers work on opt)
```

Things to know in packed mode:
- **No gradient accumulation**: every backward pass is an update. Use a bigger batch instead.
- Call each `TickerLinear` at most once per forward pass while training (true for normal transformers).
- Set `layer.trainable = False` to freeze a layer (it still passes gradients to earlier layers).
- `model.state_dict()` holds everything: codes, scales, carry, momentum, step counts. `layer.dense_weight()` gives the
  current weight as a dense tensor, e.g. to export.
- `AdamW8bit` (for the other parameters) is a simple, chunked 8-bit AdamW that updates bf16 weights with random
  rounding. bitsandbytes' 8-bit AdamW is a faster alternative.

### Options

`TickerMuon(params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0, carry_bits=8,
scales="fp32", seed=0, ns_dtype=None)`. `carry_bits` and `scales` apply to simulated parameters; packed layers take
theirs from `convert_linears(model, scales="fp8", carry_bits=8)`. Carries above 8 bits are stored as int16.
`ns_dtype` defaults to fp32 on CPU, bf16 on Ampere or newer GPUs and fp16 on older ones.

## Random numbers

Random rounding uses a counter-based hash of (seed, layer, step, purpose, element index) instead of a random
generator's stream. The draws therefore do not depend on the device, on chunking or on the order layers are updated,
and runs are reproducible.

## Tests

```
python test_ticker_muon.py
```

They check that the packed implementation gives exactly the same weights, carry and momentum as a plain dense
reference implementation of the same maths (fed the same random draws) over 25 steps, for several shapes, FP8 and fp32
scales and 4-, 8- and 12-bit carries; that weights stay on the 4-bit grid and the leftover never exceeds half a gap;
that pack/unpack round-trips; that the update inside the backward pass uses the true gradient of the model before the
step, with and without activation checkpointing; plus the simulated mode, the 8-bit AdamW and saving/loading.

## Limits

- The forward pass rebuilds each weight from its 4-bit codes in plain PyTorch; there is no fused 4-bit matrix-multiply
  kernel, so it is slower than a dense bf16 layer.
- Weight rows must have a length that is a multiple of 16.
- Tested on CPU and an NVIDIA T4. Results on training quality are in the paper; this file is the method, not the
  evidence.
