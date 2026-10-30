"""GPTQ start for P5 (PAPER_PLAN.md, seventh change-log entry): the GSM8K test accuracy of Qwen3.5-0.8B with its decoder
matrices rounded to our 4-bit format by GPTQ (Frantar et al., 2023), against round to nearest (the 4-bit start every P5
4-bit run used) and the float model. A measurement only: no training.

Same format, same matrices and same evaluation as p5_rl.py: E2M1 values with one fp32 scale per 16 inputs (absmax / 6)
on every 2D linear weight inside the decoder layers; greedy answers to all 1,319 test questions from vLLM in bf16, with
the rounded weights copied in as the RL runs did.

GPTQ: layer by layer; each matrix's input statistics come from calibration text run through the model whose earlier
layers are already rounded (the matrices of one layer all see the same, not yet rounded, layer). Calibration: --calib
GSM8K train questions with their reference solutions, in the chat format the RL prompts use, which is GPTQ's best case
for this test. 1% damping, columns in their natural order, blocks of 128 columns, as gptq_paper.py.

Run:  python gptq_qwen.py                       (on the P5 environment, see the top of p5_rl.py)
      python gptq_qwen.py --no-eval --calib 2 --layers 1 --threads 2   (CPU smoke test of the GPTQ part)
Writes runs/paper/GPTQ/qwen_start.json.
"""
import argparse
import json
import math
import os
import re
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I leaves the script's folder off the path
import p5_rl as p5  # noqa: E402  (sets the environment variables vLLM needs before torch loads)
import torch  # noqa: E402

GROUP = p5.BLOCK


def round_codes(x):
    """x already divided by its scale: the nearest signed E2M1 value, clipped to +-6."""
    grid = p5.E2M1.to(x.device)
    idx = torch.bucketize(x.abs().clamp(max=6.0), (grid[1:] + grid[:-1]) / 2)
    return grid[idx] * x.sign()


@torch.no_grad()
def gptq(W, H, damp=0.01, blocksize=128):
    """W (out, in), H (in, in) = mean of x x^T over the calibration tokens. Returns the rounded W and its scales (as
    gptq_paper.py). A scale is set from the block's weights when GPTQ reaches the block, so it can differ from the
    rounded block's absmax / 6; the stored form is still a 4-bit value per weight and one fp32 scale per 16."""
    W = W.clone().float()
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += torch.eye(H.shape[0], device=H.device) * damp * torch.diag(H).mean()
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    scales = torch.zeros(W.shape[0], W.shape[1] // GROUP, device=W.device)
    for i1 in range(0, W.shape[1], blocksize):
        i2 = min(i1 + blocksize, W.shape[1])
        W1, Hinv1 = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        Err1 = torch.zeros_like(W1)
        for i in range(i2 - i1):
            if (i1 + i) % GROUP == 0:  # the block's scale, from the error-updated weights of its 16 columns
                s = W1[:, i:i + GROUP].abs().amax(1).clamp_min(1e-12) / 6
                scales[:, (i1 + i) // GROUP] = s
            w, d = W1[:, i], Hinv1[i, i]
            q = round_codes(w / s) * s
            Q[:, i1 + i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, scales


def check_grid(Q, scales):
    codes = Q.reshape(Q.shape[0], -1, GROUP) / scales.unsqueeze(-1)
    return bool(((codes.abs().unsqueeze(-1) - p5.E2M1.to(Q.device)).abs().amin(-1) < 1e-4).all())


class Stop(Exception):
    pass


@torch.no_grad()
def gptq_model(model, mats, names, calib, amp, n_layers):
    """Round the decoder matrices of the first n_layers layers in place, in layer order."""
    layer_of = {}
    for p, n in names.items():
        m = re.search(r"\.layers\.(\d+)\.", n)
        layer_of.setdefault(int(m.group(1)), []).append(p)
    linears = {m.weight: (name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    assert all(p in linears for p in mats), "every decoder matrix should be the weight of an nn.Linear"
    blocks = {int(m.group(1)): mod for name, mod in model.named_modules()
              if (m := re.fullmatch(r".*\.layers\.(\d+)", name)) and int(m.group(1)) in layer_of}
    on_grid, err = True, {}
    for li in sorted(layer_of)[:n_layers]:
        stats = {p: [None, 0] for p in layer_of[li]}

        def make_hook(p):
            def hook(mod, inp):
                a = inp[0].reshape(-1, inp[0].shape[-1]).float()
                st = stats[p]
                with torch.autocast(a.device.type, enabled=False):  # the hook runs inside the bf16 autocast
                    st[0] = a.T @ a if st[0] is None else st[0] + a.T @ a
                st[1] += a.shape[0]
            return hook

        def stop(mod, inp, out):
            raise Stop

        hs = [linears[p][1].register_forward_pre_hook(make_hook(p)) for p in layer_of[li]]
        hs.append(blocks[li].register_forward_hook(stop))
        for ids in calib:
            try:
                with amp():
                    model(input_ids=ids.unsqueeze(0))
            except Stop:
                pass
        for h in hs:
            h.remove()
        for p in layer_of[li]:
            H, n = stats[p]
            Q, s = gptq(p.data, H / n)
            on_grid &= check_grid(Q, s)
            err[names[p]] = float((Q - p).norm() / p.norm())
            p.data.copy_(Q)
        print(f"layer {li}: {len(layer_of[li])} matrices rounded", flush=True)
    return on_grid, err


def paired(a, b):
    """mean and standard error over the test questions of a - b (per-question 0/1 strings)"""
    d = [int(x) - int(y) for x, y in zip(a, b)]
    m = sum(d) / len(d)
    return dict(mean=round(m, 5), se=round(math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)), 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--calib", type=int, default=128, help="calibration examples (GSM8K train)")
    ap.add_argument("--calib-seed", type=int, default=1234)
    ap.add_argument("--layers", type=int, default=0, help="round only the first this many layers (smoke tests)")
    ap.add_argument("--eval-max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=1536)
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--no-eval", action="store_true", help="GPTQ only, no vLLM (CPU smoke tests)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(p5.HERE, "runs", "paper", "GPTQ", "qwen_start.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    adt = torch.bfloat16
    amp = (lambda: torch.autocast(device, dtype=adt))
    t0 = time.time()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    eos_ids = sorted({tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None})
    test = p5.load_gsm8k("test")
    test_prompts = [tok(p5.build_prompt(tok, ex["q"]), add_special_tokens=False)["input_ids"] for ex in test]
    vgen = None
    if not args.no_eval:  # as p5_rl.py: vLLM before the trainer model takes memory, the same settings
        vargs = types.SimpleNamespace(vllm_dtype="bfloat16", vllm_mem=args.vllm_mem, max_model_len=args.max_model_len,
                                      vllm_max_seqs=256, seed=1, vllm_eager=False)
        vgen = p5.VLLMGen(args.model, vargs, eos_ids)
    tok, model = p5.load_model(args.model, device)
    model.eval()
    names = {p: n for n, p in model.named_parameters() if p5.is_decoder_matrix(n, p)}
    mats = list(names)

    out = dict(script="gptq_qwen.py", model=args.model, calib=args.calib, calib_seed=args.calib_seed,
               calib_source="GSM8K train question + reference solution, chat format", damp=0.01, blocksize=128,
               decoder_matrices=len(mats), decoder_matrix_weights=sum(p.numel() for p in mats),
               gpu=torch.cuda.get_device_name() if device == "cuda" else None, eval_n=len(test),
               eval_max_new_tokens=args.eval_max_new_tokens,
               logged=dict(float=0.54663, rtn=0.48143,
                           source="runs/paper/P5/p5_*_s1_lr4e-5.json, step-0 eval (A100, bf16)"))

    def score(tag, weights):
        if vgen is None:
            return
        if weights is not None:
            vgen.load([(p5.ckpt_name(names[p]), w.to(adt)) for p, w in weights])
        e = p5.evaluate(vgen, tok, test_prompts, test, args.eval_max_new_tokens)
        out[tag] = e
        print(tag, json.dumps({k: v for k, v in e.items() if k != "correct"}), flush=True)

    score("float", None)
    score("rtn", [(p, p5.quantize_rtn(p.data)) for p in mats])

    train = p5.load_gsm8k("train")
    import pyarrow.parquet as pq
    raw = pq.read_table(p5.fetch(p5.GSM_URL.format("train"), "gsm8k_train.parquet")).to_pylist()
    assert len(raw) == len(train)
    g = torch.Generator().manual_seed(args.calib_seed)
    pick = torch.randperm(len(raw), generator=g)[:args.calib].tolist()
    calib = [torch.tensor(tok(p5.build_prompt(tok, raw[i]["question"]) + raw[i]["answer"] + "<|im_end|>",
                              add_special_tokens=False)["input_ids"], device=device) for i in pick]
    out["calib_tokens"] = sum(len(c) for c in calib)
    t1 = time.time()
    on_grid, err = gptq_model(model, mats, names, calib, amp, args.layers or 10 ** 9)
    out.update(on_grid=on_grid, gptq_s=round(time.time() - t1, 1),
               rel_change_mean=round(sum(err.values()) / len(err), 5))
    print(f"GPTQ done in {out['gptq_s']} s, on grid {on_grid}, mean relative change {out['rel_change_mean']}",
          flush=True)
    score("gptq", [(p, p.data) for p in mats])

    if vgen is not None:
        out["reproduced"] = dict(float=abs(out["float"]["acc"] - out["logged"]["float"]) <= 0.005,
                                 rtn=abs(out["rtn"]["acc"] - out["logged"]["rtn"]) <= 0.005)
        out["gptq_minus_rtn"] = paired(out["gptq"]["correct"], out["rtn"]["correct"])
        out["float_minus_gptq"] = paired(out["float"]["correct"], out["gptq"]["correct"])
        out["share_of_rounding_loss_recovered"] = round(
            (out["gptq"]["acc"] - out["rtn"]["acc"]) / (out["float"]["acc"] - out["rtn"]["acc"]), 4)
    out["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("float", "rtn", "gptq")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
