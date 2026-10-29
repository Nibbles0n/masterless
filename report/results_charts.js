const fmt = (v, d = 3) => (v == null || !isFinite(v)) ? "–" : (+v).toFixed(d);
const sgn = (v, d = 1) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
const M = DATA.p5main, S = DATA.p5short, P2 = DATA.p2;
const ARMC = { a: "--c-m2", b: "--c-m3", c: "--c-c" };
const ARMN = { a: "(a) normal 16-bit", b: "(b) 4-bit frozen + LoRA", c: "(c) ours, 4-bit" };

function hover(host, svg, W, H, hitAttrs, rowsFn, title, cx, cy) {
  const tip = host._tip || (host._tip = tipBox(host));
  const hit = el("svg:rect", Object.assign({ fill: "transparent", tabindex: "0" }, hitAttrs), svg);
  const on = () => { tipRows(tip, title, rowsFn()); const r = svg.getBoundingClientRect(), hr = host.getBoundingClientRect(); placeTip(host, tip, r.left - hr.left + cx * r.width / W, r.top - hr.top + cy * r.height / H); };
  hit.addEventListener("pointerenter", on); hit.addEventListener("focus", on);
  hit.addEventListener("pointerleave", () => tip.hidden = true); hit.addEventListener("blur", () => tip.hidden = true);
}

/* ---------- P5: start -> end per run (dumbbell) ---------- */
(function () {
  const host = document.getElementById("ch-p5runs");
  const rows = M.runs;
  const draw = () => {
    header(host, { title: "GSM8K exam accuracy before and after 200 steps of RL", sub: "1,319 questions, greedy; hollow dot: the start, filled dot: after 200 steps; ×: LoRA's add-on merged and rounded to 4 bits" });
    legend(host, ["a", "b", "c"].map(k => ({ label: ARMN[k], color: tok(ARMC[k]) })));
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 34, m = { l: narrow ? 74 : 150, r: narrow ? 44 : 60, t: 8, b: 34 };
    const H = m.t + rows.length * rh + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "accuracy before and after per run" }, host);
    const X = scale(45, 75, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of [45, 50, 55, 60, 65, 70, 75]) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok("--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, v + "%"); }
    rows.forEach((r, i) => {
      const y = m.t + i * rh + rh / 2, s = 100 * r.acc_start, e = 100 * r.acc_end, c = tok(ARMC[r.arm]);
      const nm = narrow ? `(${r.arm}) s${r.seed}` : `${ARMN[r.arm].split(" ").slice(1).join(" ")} s${r.seed}`;
      el("svg:text", { x: m.l - 10, y: y + 4, "text-anchor": "end" }, g, nm);
      el("svg:line", { x1: X(s), x2: X(e), y1: y, y2: y, stroke: c, "stroke-width": 2, "stroke-opacity": 0.6 }, g);
      el("svg:circle", { cx: X(s), cy: y, r: 5, fill: tok("--surface"), stroke: c, "stroke-width": 2 }, g);
      el("svg:circle", { cx: X(e), cy: y, r: 6, fill: c, stroke: tok("--surface"), "stroke-width": 2 }, g);
      el("svg:text", { x: X(e) + 10, y: y + 4, class: "lab" }, g, e.toFixed(1) + "%");
      const rd = r.lora_rounded ? 100 * r.lora_rounded.acc : null;
      if (rd != null) {  // the same model, add-on merged and rounded to 4 bits
        const q = 5, xr = X(rd);
        el("svg:path", { d: `M${xr - q},${y - q}L${xr + q},${y + q}M${xr - q},${y + q}L${xr + q},${y - q}`, stroke: tok("--ink"), "stroke-width": 2.5 }, g);
      }
      hover(host, svg, W, H, { x: 0, y: y - rh / 2, width: W, height: rh }, () => [
        { value: s.toFixed(1) + "%", label: "start", color: c },
        { value: e.toFixed(1) + "%", label: "after 200 steps", color: c },
        { value: sgn(e - s), label: "points gained" },
        ...(rd != null ? [{ value: rd.toFixed(1) + "%", label: "add-on merged, rounded to 4 bits (×)" }] : []),
        { value: fmt(r.peak_mem_gb, 1) + " GB", label: "peak GPU memory" }], `${ARMN[r.arm]}, seed ${r.seed}`, X(e), y);
    });
    numbers(host, ["run", "start", "after 200 steps", "gain (points)", "rounded to 4 bits", "peak memory"],
      rows.map(r => [`${ARMN[r.arm]} s${r.seed}`, fmt(100 * r.acc_start, 1) + "%", fmt(100 * r.acc_end, 1) + "%", sgn(100 * (r.acc_end - r.acc_start)), r.lora_rounded ? fmt(100 * r.lora_rounded.acc, 1) + "%" : "–", fmt(r.peak_mem_gb, 1) + " GB"]));
  };
  register(host, draw);
})();

/* ---------- P5: the pass lines, both designs (forest plot) ---------- */
(function () {
  const host = document.getElementById("ch-p5lines");
  const items = [
    { k: "P5a, 200 steps, seeds 1–3", n: "P5a 1–3", d: M.lines.P5a.diff, h: M.lines.P5a.half, line: -4, c: "--c-c", strong: true },
    { k: "P5a, 200 steps, seeds 1–2", n: "P5a 1–2", d: M.lines.P5a_seeds12.diff, h: M.lines.P5a_seeds12.half, line: -4, c: "--c-c" },
    { k: "P5a, 30 steps", n: "P5a 30", d: S.lines.P5a.diff, h: S.lines.P5a.half, line: -4, c: "--c-c" },
    { k: "4-bit start vs float start", n: "start gap", d: M.lines.start_gap_4bit.diff, h: M.lines.start_gap_4bit.half, c: "--c-m0" },
    { k: "(c) − (b), 200 steps", n: "c − b 200", d: M.lines.c_minus_b ? M.lines.c_minus_b.diff : null, h: M.lines.c_minus_b ? M.lines.c_minus_b.half : null, c: "--c-m3" },
    { k: "(c) − (b), 30 steps", n: "c − b 30", d: S.lines.c_minus_b.diff, h: S.lines.c_minus_b.half, c: "--c-m3" },
    { k: "(c) − (b) rounded to 4 bits", n: "c − b 4-bit", d: M.lines.c_minus_b_rounded.diff, h: M.lines.c_minus_b_rounded.half, c: "--c-m3", strong: true },
  ].filter(x => x.d != null);
  const draw = () => {
    header(host, { title: "Ours minus the other setups, in exam points", sub: "question-by-question differences, seeds pooled; bars are 95% intervals; P5a passes if the whole bar is right of −4" });
    const W = Math.max(280, host.clientWidth - 24), narrow = W < 560, rh = 34, m = { l: narrow ? 78 : 190, r: 20, t: 8, b: 34 };
    const H = m.t + items.length * rh + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "differences with intervals" }, host);
    const X = scale(-10, 25, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of [-10, -5, 0, 5, 10, 15, 20, 25]) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(v === 0 ? "--axis" : "--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v)); }
    el("svg:line", { x1: X(-4), x2: X(-4), y1: m.t, y2: H - m.b, stroke: tok("--ink-2"), "stroke-dasharray": "4 4" }, g);
    el("svg:text", { x: X(-4) - 4, y: H - m.b - 6, "text-anchor": "end", class: "lab" }, g, "−4 line");
    items.forEach((it, i) => {
      const y = m.t + i * rh + rh / 2, c = tok(it.c);
      el("svg:text", { x: m.l - 10, y: y + 4, "text-anchor": "end", class: it.strong ? "lab-strong" : null }, g, narrow ? it.n : it.k);
      el("svg:line", { x1: X(it.d - it.h), x2: X(it.d + it.h), y1: y, y2: y, stroke: c, "stroke-width": 2, "stroke-linecap": "round" }, g);
      el("svg:circle", { cx: X(it.d), cy: y, r: it.strong ? 7 : 5.5, fill: c, stroke: tok("--surface"), "stroke-width": 2 }, g);
      hover(host, svg, W, H, { x: 0, y: y - rh / 2, width: W, height: rh }, () => [
        { value: sgn(it.d) + " ± " + fmt(it.h, 1), label: "points, 95% interval", color: c },
        { value: sgn(it.d - it.h) + " to " + sgn(it.d + it.h), label: "range" }], it.k, X(it.d), y);
    });
    numbers(host, ["comparison", "points", "95% interval"], items.map(it => [it.k, sgn(it.d), sgn(it.d - it.h) + " to " + sgn(it.d + it.h)]));
  };
  register(host, draw);
})();

/* ---------- P5: training reward over 200 steps ---------- */
(function () {
  const sm = (v, k = 10) => v.map((_, i) => { const w = v.slice(Math.max(0, i - k + 1), i + 1); return [i + 1, w.reduce((s, q) => s + q, 0) / w.length]; });
  const mean = arm => { const rs = M.runs.filter(r => r.arm === arm).map(r => r.reward); return rs.length ? rs[0].map((_, i) => rs.reduce((s, r) => s + r[i], 0) / rs.length) : []; };
  lineChart(document.getElementById("ch-p5reward"), {
    title: "Share of practice answers marked right, during training",
    sub: "mean over the last 10 steps, seeds averaged; 32 questions × 8 answers per step",
    height: 300,
    x: { ticks: [0, 50, 100, 150, 200], domain: [1, 200], label: "step" },
    y: { domain: [0.25, 0.8], ticks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], fmt: v => v.toFixed(1), tip: v => fmt(v, 3) },
    series: () => [
      { label: "(a) normal, 4e-5", color: ARMC.a, points: sm(mean("a")) },
      { label: "(c) ours, 4e-5", color: ARMC.c, strong: true, points: sm(mean("c")) },
      { label: "(b) LoRA", color: ARMC.b, points: sm(mean("b")) },
      { label: "(a) normal, 1e-4", color: "--c-m0", points: sm(DATA.lr1e4.reward) },
    ],
  });
})();

/* ---------- P2: ours minus the float copy, per seed ---------- */
(function () {
  const host = document.getElementById("ch-p2gap");
  const rows = P2.seeds;
  const draw = () => {
    header(host, { title: "Ours minus the 4-bit float copy, final validation loss", sub: "per seed; bars are 95% intervals over the 256 validation windows; pass line +0.03 (dashed)" });
    const W = Math.max(280, host.clientWidth - 24), rh = 36, m = { l: 70, r: 20, t: 8, b: 34 };
    const H = m.t + rows.length * rh + m.b;
    const svg = el("svg:svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "loss gap per seed" }, host);
    const X = scale(-0.01, 0.04, m.l, W - m.r), g = el("svg:g", {}, svg);
    for (const v of [-0.01, 0, 0.01, 0.02, 0.03, 0.04]) { el("svg:line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b, stroke: tok(v === 0 ? "--axis" : "--grid") }, g); el("svg:text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, g, (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(2)); }
    el("svg:line", { x1: X(0.03), x2: X(0.03), y1: m.t, y2: H - m.b, stroke: tok("--ink-2"), "stroke-dasharray": "4 4" }, g);
    rows.forEach((r, i) => {
      const y = m.t + i * rh + rh / 2, c = tok("--c-c");
      el("svg:text", { x: m.l - 10, y: y + 4, "text-anchor": "end" }, g, "seed " + r.seed);
      el("svg:line", { x1: X(r.diff - r.half), x2: X(r.diff + r.half), y1: y, y2: y, stroke: c, "stroke-width": 2, "stroke-linecap": "round" }, g);
      el("svg:circle", { cx: X(r.diff), cy: y, r: 6, fill: c, stroke: tok("--surface"), "stroke-width": 2 }, g);
      hover(host, svg, W, H, { x: 0, y: y - rh / 2, width: W, height: rh }, () => [
        { value: fmt(r.ours, 4), label: "ours", color: c },
        { value: fmt(r.copy, 4), label: "float copy", color: tok("--c-m2") },
        { value: sgn(r.diff, 4) + " ± " + fmt(r.half, 4), label: "gap, 95% interval" }], "seed " + r.seed, X(r.diff), y);
    });
    numbers(host, ["seed", "ours", "float copy", "gap", "95% interval"], rows.map(r => ["seed " + r.seed, fmt(r.ours, 4), fmt(r.copy, 4), sgn(r.diff, 4), "± " + fmt(r.half, 4)]));
  };
  register(host, draw);
})();

/* ---------- P2: validation loss curves ---------- */
lineChart(document.getElementById("ch-p2curve"), {
  title: "Validation loss during training, seed 1",
  sub: "from step 2,000 (all three start at 10.9); evaluated every 500 steps",
  height: 300,
  x: { ticks: [5000, 10000, 15000, 20000, 25000], domain: [2000, 27466], label: "step", fmt: v => (v / 1000) + "k" },
  y: { domain: [3.85, 4.6], ticks: [3.9, 4.1, 4.3, 4.5], fmt: v => v.toFixed(1), tip: v => fmt(v, 4), tipAscending: true },
  series: () => [
    { label: "ours", color: "--c-c", strong: true, points: P2.curves["ours s1"].map(e => [e.step, e.val_ce]) },
    { label: "4-bit float copy", color: "--c-m2", points: P2.curves["copy s1"].map(e => [e.step, e.val_ce]) },
    { label: "bf16-SR (in bf16)", color: "--c-m5", points: P2.curves["bf16-SR s1"].map(e => [e.step, e.val_ce]) },
  ],
});
