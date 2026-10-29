"""P5 of PAPER_PLAN.md: full-parameter RL (GRPO on GSM8K) with 4-bit weights and no float copy, on the Qwen3.5 small
series (hybrid Gated DeltaNet : gated attention at 3 : 1; text only).

Arms (--arm):
  float   full-parameter RL with an fp32 copy of every weight, bf16 autocast (the usual way; arm (a))
  lora4   every 2D linear weight inside the decoder layers frozen at 4-bit round-to-nearest, LoRA rank 32 trained on
          them (QeRL-style; arm (b)); everything else frozen too; always AdamW
  ticker  full-parameter on 4-bit weights with no float copy (ours; arm (c)): each step
            target = stored 4-bit weight + carry - lr * update
            new weight = nearest 4-bit value of the target (block scales recomputed from the target)
            new carry  = target - new weight, an integer k in -L..L standing for k/(2L) of the gap on its own side,
                         rounded at random, L = 2^(b-1) - 1 (b = --carry-bits; b=8: k/254)
The 4-bit format is side_gpt2.py's: E2M1 values {0,.5,1,1.5,2,3,4,6} with sign, times an fp32 scale per block of 16
along the input dimension (absmax / 6). The 4-bit / ticker treatment covers every 2D linear weight inside the decoder
layers (attention q/k/v/o, Gated DeltaNet in_proj_qkv/z/a/b and out_proj, MLP gate/up/down). The embedding (tied to the
output head), norms, conv1d, A_log, dt_bias use float AdamW at --lr.

Optimizer (--opt) for the decoder matrices:
  muon   Nesterov 0.95, 5 Newton-Schulz steps (3.4445, -4.7750, 2.0315), update scaled by max(1, rows/cols)**0.5;
         float arm: fp32 momentum; ticker: 8-bit momentum (random rounding, fp32 scale per 128 weights)
  adamw  float arm: AdamW with fp32 moments; ticker: AdamW direction m^/(sqrt(v^)+eps) with 8-bit moments
         (m: 8-bit integers, random rounding, fp32 scale per 128 weights; sqrt(v): FP8 E4M3, round to nearest, fp32
         scale per 128 weights, because a linear 8-bit grid flushes small second moments to zero)
  The ticker carries whichever update is used.

GRPO: --answers answers (8) to each of --questions questions (64) of the GSM8K train set per step, sampled at
temperature 1; reward 1 if the final number matches the '####' answer; advantage = (reward - group mean) / (group std
+ 1e-4); one on-policy update per step (so the clipped ratio is 1 and the loss is the policy gradient), token-mean over
the step, no KL term. Prompt: the model's chat template in non-thinking mode with one fixed instruction.

Generation (--gen): hf = the trainer model's own generate with KV cache; vllm = vLLM in the same process, with the
trained weights copied in (bf16) after every optimizer step. With vllm the gap between the probabilities the generator
and the trainer give each sampled token is logged.

Eval: GSM8K test (1,319 questions), greedy, at step 0 and at the end (and every --eval-every steps); per-question
correctness is saved so paired confidence intervals can be computed later.

Environment (tested on Colab A100-40GB and T4, 2026-10-28): pip install vllm==0.32.0 transformers==5.19.0
flash-linear-attention pyarrow (vLLM 0.32 wants transformers < 5.20; Qwen3.5 needs >= 5.2). If the conv1d of the Gated
DeltaNet fails with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH, two cuDNN wheels share one folder: pip uninstall
nvidia-cudnn-cu12, then pip install --force-reinstall --no-deps nvidia-cudnn-cu13==<the version torch wants>.
causal-conv1d is not needed (torch's conv1d is used). On a T4 (no bf16) everything runs in fp16 with loss scaling, but
the full-parameter arms of the 0.8B model do not fit in its 15 GB; the pilot settings are in runs/paper/P5_PILOT.md.

Run:  python p5_rl.py --arm ticker --opt adamw --gen vllm --questions 32
      python p5_rl.py --arm float --steps 0 --gen vllm   (just the float GSM8K test accuracy)
Writes runs/paper/p5_<arm>_<opt>[_c<bits>]_<gen>_s<seed><tag>.jsonl (one line per step) and .json (summary).
"""
import argparse
import json
import math
import os
import re
import time
import types
import urllib.request

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch touches the GPU
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # vLLM in this process, so weights can be copied in
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
GSM_URL = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/{}-00000-of-00001.parquet"
INSTRUCTION = "Solve the problem step by step. At the end, write the final answer as a single number after '####'."


# ---------------------------------------------------------------- data and reward

def fetch(url, name):
    path = os.path.join(HERE, "data", name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    return path


def load_gsm8k(split):
    import pyarrow.parquet as pq
    rows = pq.read_table(fetch(GSM_URL.format(split), f"gsm8k_{split}.parquet")).to_pylist()
    return [dict(q=r["question"], gold=parse_number(r["answer"].split("####")[-1])) for r in rows]


NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def parse_number(s):
    nums = NUM.findall(s)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").replace("$", ""))
    except ValueError:
        return None


def extract_answer(text):
    """The final number: inside the last \\boxed{}, else after the last '####', else the last number in the text."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        return parse_number(boxed[-1])
    if "####" in text:
        v = parse_number(text.rsplit("####", 1)[-1])
        if v is not None:
            return v
    return parse_number(text)


def reward(text, gold):
    a = extract_answer(text)
    return float(a is not None and gold is not None and abs(a - gold) <= 1e-6 * max(1.0, abs(gold)))


def build_prompt(tok, question):
    msgs = [{"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------- 4-bit grid (side_gpt2.py)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BLOCK = 16
MBLOCK = 128
GRID = torch.cat([-E2M1.flip(0)[:-1], E2M1])  # the 15 signed 4-bit values, -6..6
GRID_MID = (GRID[1:] + GRID[:-1]) / 2
GAP_UP = torch.cat([GRID[1:] - GRID[:-1], torch.tensor([2.0])])  # gap to the next value up (2 above +6)
GAP_DN = torch.cat([torch.tensor([2.0]), GRID[1:] - GRID[:-1]])  # gap to the next value down (2 below -6)


def block_scale(w):
    """One fp32 scale per block of 16 weights along the input dimension: absmax / 6. Shape (out, in / 16, 1)."""
    return w.view(w.size(0), -1, BLOCK).abs().amax(-1, keepdim=True).clamp_min(1e-12) / 6


def quantize_rtn(w):
    """Round w (out, in) to nearest on the scaled E2M1 grid."""
    grid = E2M1.to(w.device)
    wb, s = w.view(w.size(0), -1, BLOCK), block_scale(w)
    x = wb / s
    idx = torch.bucketize(x.abs(), (grid[1:] + grid[:-1]) / 2)
    return (grid[idx] * x.sign() * s).view_as(w)


def on_grid(w):
    v = (w.view(w.size(0), -1, BLOCK) / block_scale(w)).abs()
    return bool(((v.unsqueeze(-1) - E2M1.to(w.device)).abs().amin(-1) < 1e-4).all())


def side_gaps(x):
    """For codes x already on the signed grid: the gaps to the next value up and down."""
    i = torch.bucketize(x, GRID_MID.to(x.device))
    return GAP_UP.to(x.device)[i], GAP_DN.to(x.device)[i]


def carry_code(r, x_new, L, gen):
    """The leftover r (in code units of the new block scale) as integers -L..L standing for k/(2L) of the gap on its
    own side, rounded at random."""
    up, dn = side_gaps(x_new)
    c = 2 * L * r / torch.where(r > 0, up, dn)
    lo = c.floor()
    k = (lo + (torch.rand(c.shape, generator=gen, device=c.device) < c - lo)).clamp(-L, L)
    return k.to(torch.int8 if L <= 127 else torch.int16)


def store_int8(a, gen):
    """a (out, in) rounded at random to 8-bit integers (-127..127) times one fp32 scale per 128 weights of a row;
    returned as the float values the integers stand for."""
    ab = a.view(a.size(0), -1, math.gcd(a.size(1), MBLOCK))
    s = ab.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
    x = ab / s
    lo = x.floor()
    q = lo + (torch.rand(x.shape, generator=gen, device=a.device) < x - lo)
    return (q * s).view_as(a)


def store_fp8(a):
    """a (out, in), not negative, stored as FP8 E4M3 (round to nearest) times one fp32 scale per 128 weights of a row
    (absmax / 448); returned as the float values the codes stand for."""
    ab = a.view(a.size(0), -1, math.gcd(a.size(1), MBLOCK))
    s = ab.amax(-1, keepdim=True).clamp_min(1e-30) / 448
    return ((ab / s).to(torch.float8_e4m3fn).float() * s).view_as(a)


def newton_schulz(G, steps=5):
    """Approximately orthogonalise G (push its singular values towards 1), as in Keller Jordan's Muon."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


# ---------------------------------------------------------------- optimizer for the decoder matrices

class MatrixOpt(torch.optim.Optimizer):
    """Muon or AdamW on the decoder matrices, stored in float (fp32 weights and state) or as the ticker (4-bit weights,
    b-bit carry, 8-bit optimizer state)."""

    def __init__(self, params, lr, gen, opt, ticker, carry_bits=8, momentum=0.95, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, betas=betas, eps=eps, weight_decay=weight_decay))
        self.gen, self.opt, self.ticker = gen, opt, ticker
        self.L = 2 ** (carry_bits - 1) - 1
        self.changed, self.total = 0, 0  # 4-bit codes changed, summed over steps since the last read
        self.upd_sq, self.upd_n = 0.0, 0  # squared size of lr * update, for matching Muon's lr to AdamW's

    @torch.no_grad()
    def init_storage(self):
        """Round the pretrained weights to nearest 4-bit; the carry starts as the leftover (as side_gpt2.py)."""
        if not self.ticker:
            return
        for group in self.param_groups:
            for p in group["params"]:
                w_hat = quantize_rtn(p)
                s = block_scale(p)
                x_new = w_hat.view(p.size(0), -1, BLOCK) / s
                self.state[p]["carry"] = carry_code((p - w_hat).view_as(x_new) / s, x_new, self.L, self.gen)
                p.copy_(w_hat)

    @torch.no_grad()
    def direction(self, p, group):
        """The update before the learning rate (Muon: orthogonalised Nesterov momentum; AdamW: m^/(sqrt(v^)+eps))."""
        st = self.state[p]
        g = p.grad.float()
        if self.opt == "muon":
            beta = group["momentum"]
            buf = st.setdefault("momentum", torch.zeros_like(p))
            buf.lerp_(g, 1 - beta)
            u = g.lerp(buf, beta)  # Nesterov
            if self.ticker:
                buf.copy_(store_int8(buf, self.gen))
            return max(1, p.size(0) / p.size(1)) ** 0.5 * newton_schulz(u).to(p.dtype)
        b1, b2 = group["betas"]
        st["step"] = st.get("step", 0) + 1
        m = st.setdefault("m", torch.zeros_like(p))
        r = st.setdefault("r", torch.zeros_like(p))  # sqrt of the second moment
        m.lerp_(g, 1 - b1)
        v = (r.square() * b2).addcmul_(g, g, value=1 - b2)
        if self.ticker:
            m.copy_(store_int8(m, self.gen))
            r.copy_(store_fp8(v.sqrt()))
        else:
            r.copy_(v.sqrt())
        t = st["step"]
        return (m / (1 - b1 ** t)) / (r / math.sqrt(1 - b2 ** t) + group["eps"])

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                delta = -lr * self.direction(p, group)
                if wd:
                    delta.add_(p, alpha=-lr * wd)
                self.upd_sq += delta.square().sum().item()
                self.upd_n += delta.numel()
                if not self.ticker:
                    p.add_(delta)
                    continue
                st = self.state[p]
                s_old = block_scale(p)
                x = p.view(p.size(0), -1, BLOCK) / s_old
                k = st["carry"].float()
                up, dn = side_gaps(x)
                carry = (k / (2 * self.L) * torch.where(k > 0, up, dn) * s_old).view_as(p)
                w_tilde = p + delta + carry
                w_hat = quantize_rtn(w_tilde)
                s_new = block_scale(w_tilde)
                x_new = w_hat.view_as(x) / s_new
                st["carry"] = carry_code((w_tilde - w_hat).view_as(x) / s_new, x_new, self.L, self.gen)
                self.changed += ((2 * x_new).round() != (2 * x).round()).sum().item()
                self.total += p.numel()
                p.copy_(w_hat)

    def read(self):
        """Share of 4-bit codes changed per step (ticker only) and the root-mean-square of lr * update, since the
        last read."""
        out = (self.changed / self.total if self.total else None,
               math.sqrt(self.upd_sq / self.upd_n) if self.upd_n else None)
        self.changed, self.total, self.upd_sq, self.upd_n = 0, 0, 0.0, 0
        return out


def state_bytes_per_weight(arm, opt, carry_bits):
    """Training-state bytes per decoder-matrix weight (weights and optimizer state; gradients not counted)."""
    if arm == "lora4":
        return 0.5 + 4 / 16  # frozen 4-bit weight + scale (the LoRA factors are counted separately)
    state = {"muon": 4, "adamw": 8}[opt]
    if arm == "float":
        return 4 + state
    carry = 1 if carry_bits <= 8 else 2
    return 0.5 + 4 / 16 + carry + {"muon": 1, "adamw": 2}[opt] + (4 / MBLOCK) * {"muon": 1, "adamw": 2}[opt]


# ---------------------------------------------------------------- model

def is_decoder_matrix(name, p):
    return ".layers." in name and name.endswith(".weight") and p.dim() == 2 and "mtp" not in name


def load_model(repo, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).to(device)
    model.config.use_cache = False
    return tok, model


def add_lora(model, rank, alpha, gen):
    """Freeze every parameter; put rank-r LoRA factors on each decoder matrix (frozen at 4-bit round-to-nearest, held
    as fp32 values on the grid like the ticker's). Names of the base weights do not change, so the merged weights can
    be copied into vLLM under the checkpoint names."""
    for p in model.parameters():
        p.requires_grad_(False)
    lora = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".layers." in name and "mtp" not in name:
            w = mod.weight
            mod.weight = nn.Parameter(quantize_rtn(w.data), requires_grad=False)
            mod.lora_A = nn.Parameter(torch.empty(rank, w.size(1), device=w.device))
            nn.init.kaiming_uniform_(mod.lora_A, a=math.sqrt(5), generator=gen)
            mod.lora_B = nn.Parameter(torch.zeros(w.size(0), rank, device=w.device))
            mod.lora_scale = alpha / rank

            def forward(self, x):
                return F.linear(x, self.weight, self.bias) + \
                    F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scale

            mod.forward = types.MethodType(forward, mod)
            lora += [mod.lora_A, mod.lora_B]
    return lora


@torch.no_grad()
def merged_lora(mod):
    return mod.weight + mod.lora_scale * mod.lora_B @ mod.lora_A


def ckpt_name(name):
    """Trainer parameter name -> checkpoint name, as vLLM's loader expects (Qwen3.5 checkpoints are multimodal)."""
    if name.startswith("model.language_model.") or name.startswith("model.visual."):
        return name
    if name.startswith("model."):
        return "model.language_model." + name[len("model."):]
    return name


# ---------------------------------------------------------------- log-probabilities (trainer)

def _chunk_logprob(h, W, t):
    logits = (h @ W.T).float()
    return logits.gather(-1, t[:, None])[:, 0] - torch.logsumexp(logits, -1)


def token_logprobs(model, ids, comp_mask, amp, chunk=1024):
    """Log-probabilities of the completion tokens of right-padded sequences ids (B, T); comp_mask marks completion
    tokens. Logits are made a chunk at a time and not kept for the backward pass (checkpointed), since the vocabulary
    has 248k entries. Returns a flat tensor in row-major order of comp_mask[:, 1:]."""
    with amp():
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
        sel = comp_mask[:, 1:].bool()
        hs, ts = h[sel], ids[:, 1:][sel]
        W = model.get_output_embeddings().weight
        out = [checkpoint(_chunk_logprob, hs[i:i + chunk], W, ts[i:i + chunk], use_reentrant=False)
               for i in range(0, len(hs), chunk)]
    return torch.cat(out)


# ---------------------------------------------------------------- generators

class HFGen:
    """The trainer model's own generate (KV cache), left-padded batches."""

    def __init__(self, model, tok, amp, eos_ids, batch):
        self.model, self.tok, self.amp, self.eos, self.batch = model, tok, amp, eos_ids, batch

    @torch.no_grad()
    def generate(self, prompts, n, temperature, max_tokens):
        """prompts: lists of token ids. Returns per prompt a list of n (completion ids, generator logprobs or None,
        truncated)."""
        model, dev = self.model, next(self.model.parameters()).device
        model.eval()
        model.config.use_cache = True
        flat = [p for p in prompts for _ in range(n)]
        res = []
        pad = self.tok.pad_token_id
        for i in range(0, len(flat), self.batch):
            chunk = flat[i:i + self.batch]
            T = max(map(len, chunk))
            ids = torch.tensor([[pad] * (T - len(c)) + c for c in chunk], device=dev)
            att = torch.tensor([[0] * (T - len(c)) + [1] * len(c) for c in chunk], device=dev)
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else \
                dict(do_sample=False)
            with self.amp():
                out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_tokens, pad_token_id=pad,
                                     eos_token_id=self.eos, repetition_penalty=1.0, **kw)
            for row in out[:, T:].tolist():
                comp = []
                for t in row:
                    comp.append(t)
                    if t in self.eos:
                        break
                truncated = comp[-1] not in self.eos if comp else True
                res.append((comp, None, truncated))
        model.config.use_cache = False
        model.train()
        return [res[j * n:(j + 1) * n] for j in range(len(prompts))]

    def load(self, named):
        pass  # the generator is the trainer


class VLLMGen:
    """vLLM in this process; the trainer's weights are copied in (bf16) after every optimizer step."""

    def __init__(self, repo, args, eos_ids):
        from vllm import LLM
        self.eos = eos_ids
        kw = dict(model=repo, dtype=args.vllm_dtype, gpu_memory_utilization=args.vllm_mem,
                  max_model_len=args.max_model_len, max_num_seqs=args.vllm_max_seqs, seed=args.seed,
                  enable_prefix_caching=False, generation_config="vllm", enforce_eager=args.vllm_eager)
        try:
            self.llm = LLM(**kw, language_model_only=True)
        except TypeError:
            self.llm = LLM(**kw, limit_mm_per_prompt={"image": 0, "video": 0})
        self._model = None
        for path in ("llm_engine.model_executor.driver_worker.worker.model_runner.model",
                     "llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model",
                     "llm_engine.model_executor.driver_worker.model_runner.model"):
            obj = self.llm
            try:
                for a in path.split("."):
                    obj = getattr(obj, a)
                self._model = obj
                break
            except AttributeError:
                continue

    def load(self, named):
        named = list(named)
        if self._model is not None:
            self._model.load_weights(named)
        else:
            self.llm.apply_model(lambda m: m.load_weights(named))

    def generate(self, prompts, n, temperature, max_tokens):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_tokens,
                            logprobs=0 if temperature > 0 else None, stop_token_ids=self.eos, detokenize=False)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
        res = []
        for o in outs:
            row = []
            for c in o.outputs:
                ids = list(c.token_ids)
                lps = None
                if c.logprobs is not None:
                    lps = [d[t].logprob for d, t in zip(c.logprobs, ids)]
                row.append((ids, lps, not (ids and ids[-1] in self.eos)))
            res.append(row)
        return res


# ---------------------------------------------------------------- eval

def evaluate(gen, tok, test_prompts, test, max_tokens):
    """Greedy accuracy on the test questions; per-question correctness as a string of 0/1."""
    t0 = time.time()
    outs = gen.generate(test_prompts, 1, 0.0, max_tokens)
    correct, lens, trunc = [], [], 0
    for (ids, _, tr), ex in zip((o[0] for o in outs), test):
        correct.append(int(reward(tok.decode(ids, skip_special_tokens=True), ex["gold"])))
        lens.append(len(ids))
        trunc += tr
    n = len(correct)
    acc = sum(correct) / n
    return dict(acc=round(acc, 5), se=round(math.sqrt(acc * (1 - acc) / n), 5), n=n,
                mean_len=round(sum(lens) / n, 1), truncated=round(trunc / n, 4), eval_s=round(time.time() - t0, 1),
                correct="".join(map(str, correct)))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("float", "lora4", "ticker"), required=True)
    ap.add_argument("--opt", choices=("adamw", "muon"), default="adamw")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--gen", choices=("hf", "vllm"), default="hf")
    ap.add_argument("--carry-bits", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6, help="AdamW lr (matrices with --opt adamw, and the rest)")
    ap.add_argument("--muon-lr", type=float, default=2e-5)
    ap.add_argument("--lora-lr", type=float, default=1e-5)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=float, default=32)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--questions", type=int, default=64, help="questions per step")
    ap.add_argument("--answers", type=int, default=8, help="answers per question")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--eval-max-new-tokens", type=int, default=1024)
    ap.add_argument("--eval-n", type=int, default=0, help="test questions to score (0: all 1,319)")
    ap.add_argument("--eval-every", type=int, default=0, help="also evaluate every this many steps (0: start, end)")
    ap.add_argument("--no-eval", action="store_true", help="skip the evals (timing runs)")
    ap.add_argument("--seed", type=int, default=1, help="seeds the question order, sampling and random rounding")
    ap.add_argument("--micro-tokens", type=int, default=16384, help="tokens per training forward pass")
    ap.add_argument("--gen-batch", type=int, default=256, help="sequences per HF generate call")
    ap.add_argument("--vllm-mem", type=float, default=0.35, help="share of GPU memory vLLM may take")
    ap.add_argument("--vllm-max-seqs", type=int, default=256)
    ap.add_argument("--vllm-eager", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto",
                    help="autocast dtype (auto: bf16 on GPUs that have it, else fp16)")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0: torch's default)")
    ap.add_argument("--time-limit", type=float, default=0, help="stop training after this many minutes (0: none)")
    ap.add_argument("--fake-reward", action="store_true", help="smoke tests only: reward is a coin flip")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "paper"))
    args = ap.parse_args()
    if args.arm == "lora4":
        args.opt = "adamw"  # QeRL-style LoRA trains its factors with AdamW

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False  # as p1_paper_gpt2.py: TF32 changed results on an A100
    if args.dtype == "auto":
        args.dtype = "bf16" if device == "cpu" or torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    adt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.vllm_dtype = "bfloat16" if args.dtype == "bf16" else "float16"
    amp = (lambda: torch.autocast(device, dtype=adt))
    gen_rng = torch.Generator(device).manual_seed(1000 + args.seed)  # random rounding and LoRA init
    data_rng = torch.Generator().manual_seed(args.seed)

    train, test = load_gsm8k("train"), load_gsm8k("test")
    if args.eval_n:
        test = test[:args.eval_n]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    eos_ids = sorted({tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None})
    enc = lambda q: tok(build_prompt(tok, q), add_special_tokens=False)["input_ids"]  # noqa: E731
    test_prompts = [enc(ex["q"]) for ex in test]

    vgen = VLLMGen(args.model, args, eos_ids) if args.gen == "vllm" else None  # before the trainer takes memory
    tok, model = load_model(args.model, device)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    names = {p: n for n, p in model.named_parameters()}
    mats = [p for n, p in model.named_parameters() if is_decoder_matrix(n, p)]
    lang = [p for n, p in model.named_parameters() if not n.startswith("model.visual") and not n.startswith("mtp")]
    rest = [p for p in lang if not any(p is m for m in mats)]
    for n, p in model.named_parameters():
        if n.startswith("model.visual") or n.startswith("mtp"):
            p.requires_grad_(False)

    mopt, opts, lora_mods = None, [], []
    if args.arm == "lora4":
        lora = add_lora(model, args.lora_rank, args.lora_alpha, gen_rng)
        lora_mods = [m for m in model.modules() if hasattr(m, "lora_A")]
        names = {p: n for n, p in model.named_parameters()}  # the frozen 4-bit weights are new Parameters
        opts.append(torch.optim.AdamW(lora, lr=args.lora_lr, weight_decay=args.weight_decay))
        trainable = lora
    else:
        mopt = MatrixOpt(mats, lr=args.muon_lr if args.opt == "muon" else args.lr, gen=gen_rng, opt=args.opt,
                         ticker=args.arm == "ticker", carry_bits=args.carry_bits, weight_decay=args.weight_decay)
        mopt.init_storage()
        opts += [mopt, torch.optim.AdamW(rest, lr=args.lr, weight_decay=args.weight_decay)]
        trainable = lang
    # fp16 only (GPUs without bf16): loss scaling, starting low because the token-mean loss is small
    scaler = torch.amp.GradScaler(device, init_scale=2.0 ** 10, enabled=args.dtype == "fp16")

    def synced():
        """The weights the generator needs, under checkpoint names, in bf16 (fp16 on GPUs without bf16)."""
        if args.arm == "lora4":
            return [(ckpt_name(names[m.weight]), merged_lora(m).to(adt)) for m in lora_mods]
        return [(ckpt_name(names[p]), p.detach().to(adt)) for p in lang]

    gen = vgen if vgen is not None else HFGen(model, tok, amp, eos_ids, args.gen_batch)
    if vgen is not None and args.arm != "float":
        vgen.load(synced())  # the generator starts from the arm's own 4-bit weights

    os.makedirs(args.out, exist_ok=True)
    cb = f"_c{args.carry_bits}" if args.arm == "ticker" else ""
    stem = os.path.join(args.out, f"p5_{args.arm}_{args.opt}{cb}_{args.gen}_s{args.seed}{args.tag}")
    log = open(stem + ".jsonl", "w")
    import transformers
    header = dict(study="P5", arm=args.arm, opt=args.opt, model=args.model, gen=args.gen,
                  carry_bits=args.carry_bits if args.arm == "ticker" else None, lr=args.lr, muon_lr=args.muon_lr,
                  lora_lr=args.lora_lr if args.arm == "lora4" else None, lora_rank=args.lora_rank,
                  lora_alpha=args.lora_alpha, weight_decay=args.weight_decay, clip=args.clip, seed=args.seed,
                  steps=args.steps, questions=args.questions, answers=args.answers, temperature=args.temperature,
                  max_new_tokens=args.max_new_tokens, eval_max_new_tokens=args.eval_max_new_tokens,
                  eval_n=len(test), instruction=INSTRUCTION, dtype=args.dtype, device=device,
                  gpu=torch.cuda.get_device_name() if device == "cuda" else None, torch=torch.__version__,
                  transformers=transformers.__version__,
                  vllm=__import__("vllm").__version__ if vgen is not None else None,
                  decoder_matrix_weights=sum(p.numel() for p in mats),
                  trainable_params=sum(p.numel() for p in trainable),
                  state_bytes_per_weight=state_bytes_per_weight(args.arm, args.opt, args.carry_bits))
    try:
        import fla  # noqa: F401
        header["fla"] = getattr(fla, "__version__", "yes")
    except ImportError:
        header["fla"] = None
    print(json.dumps(header), flush=True)

    def write(row):
        print(json.dumps({k: v for k, v in row.items() if k != "correct"}), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    evals, rows = [], []
    t_start = time.time()
    if not args.no_eval:
        e = dict(step=0, **evaluate(gen, tok, test_prompts, test, args.eval_max_new_tokens))
        evals.append(e)
        write(dict(e, kind="eval"))
    order = torch.randperm(len(train), generator=data_rng).tolist()
    for it in range(args.steps):
        t0 = time.time()
        qi = [order[(it * args.questions + j) % len(order)] for j in range(args.questions)]
        prompts = [enc(train[i]["q"]) for i in qi]
        outs = gen.generate(prompts, args.answers, args.temperature, args.max_new_tokens)
        if device == "cuda":
            torch.cuda.synchronize()
        t_gen = time.time() - t0

        # rewards and group-normalised advantages
        seqs, rewards, n_tok, n_trunc, lens = [], [], 0, 0, []
        for p, i, group in zip(prompts, qi, outs):
            r = torch.tensor([reward(tok.decode(c, skip_special_tokens=True), train[i]["gold"]) for c, _, _ in group])
            if args.fake_reward:
                r = (torch.rand(len(group), generator=data_rng) < 0.5).float()
            rewards.append(r)
            adv = (r - r.mean()) / (r.std(unbiased=True) + 1e-4) if len(r) > 1 else r * 0
            for (c, lp, tr), a in zip(group, adv.tolist()):
                lens.append(len(c))
                n_trunc += tr
                n_tok += len(c)
                if a != 0 and len(c):
                    seqs.append((p, c, a, lp))
        R = torch.stack(rewards)
        # policy gradient, token mean over all completion tokens of the step; sequences with zero advantage add no
        # gradient and are skipped
        t1 = time.time()
        seqs.sort(key=lambda s: len(s[0]) + len(s[1]))
        loss_sum, gaps = 0.0, []
        i = 0
        model.train()
        while i < len(seqs):
            j = i + 1
            while j < len(seqs) and (j - i + 1) * (len(seqs[j][0]) + len(seqs[j][1])) <= args.micro_tokens:
                j += 1
            mb = seqs[i:j]
            T = max(len(p) + len(c) for p, c, _, _ in mb)
            ids = torch.full((len(mb), T), tok.pad_token_id, dtype=torch.long)
            cm = torch.zeros((len(mb), T), dtype=torch.long)
            for k, (p, c, _, _) in enumerate(mb):
                ids[k, :len(p) + len(c)] = torch.tensor(p + c)
                cm[k, len(p):len(p) + len(c)] = 1
            ids, cm = ids.to(device), cm.to(device)
            lp = token_logprobs(model, ids, cm, amp)
            adv = torch.cat([torch.full((len(c),), a) for _, c, a, _ in mb]).to(device)
            loss = -(adv * lp).sum() / max(1, n_tok)
            scaler.scale(loss).backward()
            loss_sum += loss.item()
            if mb[0][3] is not None:  # generator log-probabilities (vLLM): the generator-trainer gap
                lg = torch.tensor([x for _, _, _, l in mb for x in l], device=device)
                lt = lp.detach().float()
                gaps.append(torch.stack([(lg - lt).sum(), (lg.exp() - lt.exp()).abs().sum(),
                                         (lg - lt).abs().max(), torch.tensor(float(len(lt)), device=device)]))
            i = j
        if device == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t1

        t2 = time.time()
        gn = 0.0
        if seqs:  # else every group had equal rewards: no gradient, no step
            for o in opts:
                scaler.unscale_(o)
            gn = torch.nn.utils.clip_grad_norm_([p for p in trainable if p.grad is not None], args.clip).item()
            for o in opts:
                scaler.step(o)
            scaler.update()
            for o in opts:
                o.zero_grad(set_to_none=True)
            if vgen is not None:
                vgen.load(synced())
        changed, upd_rms = mopt.read() if mopt is not None else (None, None)
        if device == "cuda":
            torch.cuda.synchronize()
        t_opt = time.time() - t2

        row = dict(step=it + 1, reward=round(R.mean().item(), 4),
                   solved_any=round((R.max(1).values > 0).float().mean().item(), 4),
                   zero_var=round((R.std(1) == 0).float().mean().item(), 4), loss=round(loss_sum, 6),
                   grad_norm=round(gn, 4), mean_len=round(sum(lens) / len(lens), 1),
                   truncated=round(n_trunc / len(lens), 4), trained_seqs=len(seqs), changed=changed,
                   update_rms=upd_rms, gen_s=round(t_gen, 2), train_s=round(t_train, 2), opt_sync_s=round(t_opt, 2),
                   step_s=round(time.time() - t0, 2),
                   mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device == "cuda" else None)
        if gaps:
            g = torch.stack(gaps).sum(0)
            row.update(gen_minus_train_logp=round((g[0] / g[3]).item(), 5),
                       prob_gap_mean=round((g[1] / g[3]).item(), 5), logp_gap_max=round(g[2].item(), 4))
        rows.append(row)
        write(row)
        if not math.isfinite(loss_sum):
            break
        if args.time_limit and time.time() - t_start > 60 * args.time_limit:
            break
        if not args.no_eval and args.eval_every and (it + 1) % args.eval_every == 0 and it + 1 < args.steps:
            e = dict(step=it + 1, **evaluate(gen, tok, test_prompts, test, args.eval_max_new_tokens))
            evals.append(e)
            write(dict(e, kind="eval"))

    if args.arm == "ticker":
        assert all(on_grid(p) for p in mats), "a matrix left the 4-bit grid"
    if args.arm == "lora4":
        assert all(on_grid(m.weight) for m in lora_mods), "a frozen matrix left the 4-bit grid"
    if not args.no_eval and args.steps:
        e = dict(step=len(rows), **evaluate(gen, tok, test_prompts, test, args.eval_max_new_tokens))
        evals.append(e)
        write(dict(e, kind="eval"))
    lora_rounded = None
    if args.arm == "lora4" and not args.no_eval and args.steps:
        # LoRA as a pure 4-bit model: merge the add-on into the 4-bit weights, round to nearest on the same grid (block
        # scales recomputed, as the ticker does) and take the exam again
        changed, total, d_sq, rel_sum = 0, 0, 0.0, 0.0
        rounded = []
        with torch.no_grad():
            for m in lora_mods:
                w4, merged = m.weight, merged_lora(m)
                r = quantize_rtn(merged)
                s = block_scale(w4)
                up, dn = side_gaps(w4.view(w4.size(0), -1, BLOCK) / s)
                gap = ((up + dn) / 2 * s).view_as(w4)
                d = merged - w4
                codes = lambda w: (2 * w.view(w.size(0), -1, BLOCK) / block_scale(w)).round()  # noqa: E731
                changed += (codes(r) != codes(w4)).sum().item()
                total += w4.numel()
                d_sq += d.square().sum().item()
                rel_sum += (d.abs() / gap).sum().item()
                rounded.append((m, r))
        if vgen is not None:
            vgen.load([(ckpt_name(names[m.weight]), r.to(adt)) for m, r in rounded])
            e = evaluate(gen, tok, test_prompts, test, args.eval_max_new_tokens)
            vgen.load(synced())
        else:  # the trainer generates: swap the rounded weights in with the add-on off, then put both back
            saved = [(m, m.weight.data.clone(), m.lora_scale) for m, _ in rounded]
            for m, r in rounded:
                m.weight.data.copy_(r)
                m.lora_scale = 0.0
            e = evaluate(gen, tok, test_prompts, test, args.eval_max_new_tokens)
            for m, w, sc in saved:
                m.weight.data.copy_(w)
                m.lora_scale = sc
        lora_rounded = dict(e, changed=round(changed / total, 6), addon_rms=math.sqrt(d_sq / total),
                            addon_over_gap=round(rel_sum / total, 6))
        write(dict(lora_rounded, kind="eval_lora_rounded"))
    timed = rows[1:] if len(rows) > 1 else rows  # the first step includes warm-up (compilation, CUDA graphs)
    mean = lambda k: round(sum(r[k] for r in timed) / len(timed), 2) if timed else None  # noqa: E731
    summary = dict(header, on_grid=args.arm in ("ticker", "lora4"), wall_s=round(time.time() - t_start, 1),
                   steps_done=len(rows), step_s=mean("step_s"), gen_s=mean("gen_s"), train_s=mean("train_s"),
                   opt_sync_s=mean("opt_sync_s"),
                   peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device == "cuda" else None,
                   acc_start=evals[0]["acc"] if evals else None, acc_end=evals[-1]["acc"] if evals else None,
                   lora_rounded=lora_rounded,
                   evals=evals, train=rows)
    json.dump(summary, open(stem + ".json", "w"))
    if args.save:
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, stem + ".pt")
    print(f"done: acc {summary['acc_start']} -> {summary['acc_end']}, {summary['step_s']} s/step, "
          f"{summary['wall_s']} s total", flush=True)


if __name__ == "__main__":
    main()
