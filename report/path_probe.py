"""How much do real Muon weights zig-zag? One milestone 1 run (Muon, float weights, CPU preset, seed 1), measuring
for each weight of the four hidden matrices, in units of its own 4-bit gap (the E2M1 gap at that weight, as milestone
3 would see it at the start):
  travelled  the sum over steps of |change in the weight|
  net        |final weight - first weight|
and the push per step at the peak and at the end. These set the noise of random rounding: drift ~ sqrt(travelled).

Run:  python report/path_probe.py   ->  prints the numbers, writes report/data/path_probe.json
"""
import importlib.util
import json
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m1", os.path.join(os.path.dirname(HERE), "m1_muon.py"))
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def gap_at(w):
    """The E2M1 gap around each weight (the gap above its magnitude), scale = block-of-16 absmax / 6."""
    s = w.view(w.size(0), -1, 16).abs().amax(-1, keepdim=True) / 6
    a = (w.view(w.size(0), -1, 16) / s).abs()
    i = torch.bucketize(a, E2M1, right=True).clamp(1, len(E2M1) - 1)
    return ((E2M1[i] - E2M1[i - 1]) * s).view_as(w)


def main():
    torch.manual_seed(1)
    train, val, vocab = m1.load_data()
    cfg = m1.Config(**m1.PRESETS["cpu"], vocab_size=vocab)
    gen = torch.Generator().manual_seed(1)
    model = m1.GPT(cfg)
    opts = m1.build_optimizer(model, cfg, "muon", 0.01)
    hidden = [p for n, p in model.named_parameters() if n.endswith(m1.MUON_NAMES)]
    gaps = [gap_at(p.detach()) for p in hidden]
    first = [p.detach().clone() for p in hidden]
    prev = [p.detach().clone() for p in hidden]
    travelled = [torch.zeros_like(p) for p in hidden]
    push = []
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
        with torch.no_grad():
            step = [(p - q).abs() / g for p, q, g in zip(hidden, prev, gaps)]
            for t, s in zip(travelled, step):
                t += s
            push.append(torch.cat([s.flatten() for s in step]).mean().item())
            prev = [p.detach().clone() for p in hidden]
    trav = torch.cat([t.flatten() for t in travelled])
    net = torch.cat([((p.detach() - f).abs() / g).flatten() for p, f, g in zip(hidden, first, gaps)])
    out = dict(travelled_mean=trav.mean().item(), travelled_median=trav.median().item(),
               net_mean=net.mean().item(), net_median=net.median().item(),
               push_peak=max(push), push_end=push[-1], push=push[::25])
    print({k: round(v, 4) for k, v in out.items() if k != "push"})
    print(f"travelled / net (means): {out['travelled_mean'] / out['net_mean']:.1f}; sqrt(travelled) {out['travelled_mean'] ** 0.5:.2f} gaps")
    json.dump(out, open(os.path.join(HERE, "data", "path_probe.json"), "w"))


if __name__ == "__main__":
    main()
