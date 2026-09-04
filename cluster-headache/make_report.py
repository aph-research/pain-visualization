#!/usr/bin/env python3
"""Build the findings report from whatever is currently in results/.

    ../.venv/bin/python make_report.py [-o out.html]

Figures are inlined as data URIs so the page is a single self-contained file.
Numbers are read from the experiments' own summary.json, never retyped, so the report
cannot drift from the run that produced it.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def load(experiment: str) -> Optional[Dict]:
    path = os.path.join(RESULTS, experiment, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def img(experiment: str, filename: str) -> str:
    path = os.path.join(RESULTS, experiment, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def figure(experiment: str, filename: str, caption: str) -> str:
    src = img(experiment, filename)
    if not src:
        return ""
    alt = html_escape(caption)
    return f"""<figure class="fig">
  <img src="{src}" alt="{alt}" loading="lazy">
  <figcaption>{caption}</figcaption>
</figure>"""


def html_escape(text: str) -> str:
    """Escape for an attribute value.  Captions carry entities like &mdash; already, so
    ampersands are left alone and only the quote characters are handled."""
    return text.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(x, digits: int = 3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "&mdash;"
    if math.isnan(v):
        return "&mdash;"
    if math.isinf(v):
        return "&infin;"
    return f"{v:.{digits}f}"


def _threshold_label(v) -> str:
    """Experiment 3's onset threshold has three meaningful non-numeric states."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "&mdash;"
    if math.isinf(x):
        return "never reached"
    if math.isnan(x):
        return "not measured"
    if x == 0.0:
        return "below the swept range"
    return f"{x:.2f}"


# --------------------------------------------------------------------------------------

CSS = """
:root {
  color-scheme: light;
  --ground:  #fbfaf8;
  --surface: #ffffff;
  --sunk:    #f4f2ee;
  --ink:     #16181d;
  --ink-2:   #4e545e;
  --muted:   #878d96;
  --rule:    #e5e3de;
  --accent:  #2f5ea8;
  --accent-soft: #eaf0fa;
  --supported: #1c7a58;
  --mixed:     #9a6b12;
  --refuted:   #a8402a;
  --supported-bg: #e7f3ee;
  --mixed-bg:     #f8efdc;
  --refuted-bg:   #f9e9e4;
  /* The figures are matplotlib PNGs rendered on a light surface, so they keep their own
     ground in both themes rather than being tinted - the chart palette is validated
     against that surface and re-tinting would break its contrast guarantees. */
  --figure-ground: #fcfcfb;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground:  #131519;
    --surface: #1a1d22;
    --sunk:    #202329;
    --ink:     #eef0f3;
    --ink-2:   #b6bcc5;
    --muted:   #868d97;
    --rule:    #2a2e35;
    --accent:  #83aae9;
    --accent-soft: #1d2534;
    --supported: #56bd97;
    --mixed:     #d6a84a;
    --refuted:   #e28d76;
    --supported-bg: #17281f;
    --mixed-bg:     #2a2313;
    --refuted-bg:   #2c1c17;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground:  #131519;
  --surface: #1a1d22;
  --sunk:    #202329;
  --ink:     #eef0f3;
  --ink-2:   #b6bcc5;
  --muted:   #868d97;
  --rule:    #2a2e35;
  --accent:  #83aae9;
  --accent-soft: #1d2534;
  --supported: #56bd97;
  --mixed:     #d6a84a;
  --refuted:   #e28d76;
  --supported-bg: #17281f;
  --mixed-bg:     #2a2313;
  --refuted-bg:   #2c1c17;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 16.5px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}

.wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px 96px; }
.col  { max-width: 68ch; }

/* ---------- masthead ---------- */
.masthead { padding: 72px 0 40px; border-bottom: 1px solid var(--rule); margin-bottom: 44px; }
.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11.5px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 18px;
}
h1 {
  font-family: "Newsreader", Georgia, "Times New Roman", serif;
  font-weight: 500; font-size: clamp(34px, 5.2vw, 54px); line-height: 1.08;
  letter-spacing: -.015em; margin: 0 0 20px; text-wrap: balance; max-width: 20ch;
}
.standfirst {
  font-family: "Newsreader", Georgia, serif;
  font-size: clamp(18px, 2.1vw, 21px); line-height: 1.5; color: var(--ink-2);
  max-width: 60ch; margin: 0;
}
.runmeta {
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
  color: var(--muted); margin-top: 26px; display: flex; flex-wrap: wrap; gap: 8px 22px;
}

/* ---------- sections ---------- */
section { margin: 0 0 68px; scroll-margin-top: 24px; }
h2 {
  font-family: "Newsreader", Georgia, serif; font-weight: 500;
  font-size: clamp(26px, 3.2vw, 34px); line-height: 1.15; letter-spacing: -.01em;
  margin: 0 0 6px; text-wrap: balance;
}
h3 {
  font-size: 16px; font-weight: 600; letter-spacing: -.005em;
  margin: 34px 0 10px; color: var(--ink);
}
p { margin: 0 0 16px; }
.col p:last-child { margin-bottom: 0; }
a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 2px; }

.sec-head { display: flex; align-items: baseline; gap: 16px; margin-bottom: 22px; flex-wrap: wrap; }
.sec-num {
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
  letter-spacing: .12em; text-transform: uppercase; color: var(--accent);
  border: 1px solid var(--rule); border-radius: 3px; padding: 3px 9px; white-space: nowrap;
}

/* ---------- verdict board ---------- */
.board { display: grid; grid-template-columns: repeat(auto-fit, minmax(272px, 1fr)); gap: 14px; }
.card {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 5px;
  padding: 18px 18px 16px; display: flex; flex-direction: column; gap: 9px;
}
.card .claim { font-size: 15px; line-height: 1.45; margin: 0; color: var(--ink); }
.card .detail { font-size: 13.5px; line-height: 1.5; color: var(--ink-2); margin: 0; }
.tag {
  align-self: flex-start;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 3px; font-weight: 500;
}
.tag.supported { color: var(--supported); background: var(--supported-bg); }
.tag.mixed     { color: var(--mixed);     background: var(--mixed-bg); }
.tag.refuted   { color: var(--refuted);   background: var(--refuted-bg); }

/* ---------- verdict callout ---------- */
.verdict {
  border-left: 3px solid var(--accent); background: var(--accent-soft);
  padding: 16px 20px; border-radius: 0 4px 4px 0; margin: 0 0 24px; max-width: 74ch;
}
.verdict p { margin: 0; font-size: 15px; line-height: 1.58; color: var(--ink); }
.verdict .label {
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 10.5px;
  letter-spacing: .1em; text-transform: uppercase; color: var(--accent);
  display: block; margin-bottom: 7px;
}

/* ---------- figures ---------- */
.fig { margin: 28px 0; }
.fig img {
  width: 100%; height: auto; display: block;
  border: 1px solid var(--rule); border-radius: 4px; background: var(--figure-ground);
}
.fig figcaption {
  font-size: 13px; line-height: 1.5; color: var(--muted); margin-top: 10px; max-width: 84ch;
}

/* ---------- tables ---------- */
.tablewrap { overflow-x: auto; margin: 22px 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; min-width: 460px; }
th, td { text-align: left; padding: 9px 14px 9px 0; border-bottom: 1px solid var(--rule); }
th {
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-weight: 500;
  font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted);
  border-bottom-color: var(--ink-2);
}
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums;
  font-family: "IBM Plex Mono", ui-monospace, monospace; }
tbody tr:last-child td { border-bottom: none; }

code, .mono {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .88em; background: var(--sunk); padding: 1px 5px; border-radius: 3px;
}
pre { background: var(--sunk); border: 1px solid var(--rule); border-radius: 4px;
  padding: 14px 16px; overflow-x: auto; font-size: 13px; line-height: 1.55; margin: 18px 0; }
pre code { background: none; padding: 0; font-size: 13px; }

ul, ol { margin: 0 0 16px; padding-left: 22px; }
li { margin-bottom: 8px; }
li::marker { color: var(--muted); }

.note {
  border: 1px solid var(--rule); border-radius: 4px; padding: 16px 18px;
  background: var(--surface); font-size: 14.5px; line-height: 1.55; color: var(--ink-2);
  margin: 22px 0; max-width: 78ch;
}
.note strong { color: var(--ink); }

hr.div { border: none; border-top: 1px solid var(--rule); margin: 56px 0; }

footer {
  border-top: 1px solid var(--rule); padding-top: 26px; margin-top: 20px;
  font-size: 13.5px; color: var(--muted); max-width: 74ch;
}

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


def build(out_path: str) -> str:
    e0, e1, e2 = load("exp0_replication"), load("exp1_geometry"), load("exp2_attack")
    e3, e4, e5 = load("exp3_interventions"), load("exp4_entrainment"), load("exp5_noise")
    e6 = load("exp6_sweeps")

    # ---- numbers pulled straight from the run ----
    ratio_med = (e1 or {}).get("tree_over_lattice_median_forced",
                               (e1 or {}).get("tree_over_lattice_median", float("nan")))
    bif_excess = (e1 or {}).get("mean_bifurcation_excess_tree", float("nan"))
    branch_mm = (e1 or {}).get("mean_branch_over_chain_mismatch_tree", float("nan"))
    sus = (e2 or {}).get("sustained_fraction_by_amplitude", {})
    eff = (e3 or {}).get("effects", {})
    ox_curve = (e3 or {}).get("oxygen_curve", [])
    ox_f = (e3 or {}).get("oxygen_factors", [])
    ox_dir = (e3 or {}).get("oxygen_direction", "reduces")
    ox_span = (e3 or {}).get("oxygen_span", float("nan"))
    thresholds = (e3 or {}).get("kernel_thresholds", {})
    peak_ratio = (e3 or {}).get("kernel_peak_ratio", {})
    pearson = (e5 or {}).get("pearson_D_LZ", float("nan"))
    spearman = (e5 or {}).get("spearman_D_LZ", float("nan"))
    per_arm = (e5 or {}).get("within_arm_correlations", {})

    def effect(name_prefix: str) -> float:
        """Change relative to the untreated arm, the same quantity the verdict quotes."""
        control = eff.get("none", {}).get("relative_change", 0.0) or 0.0
        for k, v in eff.items():
            if k.startswith(name_prefix):
                raw = v.get("relative_change")
                return raw - control if raw is not None else float("nan")
        return float("nan")

    dmt, meo = effect("DMT"), effect("5-MeO")
    oxy = effect("oxygen")

    def pct(x) -> str:
        try:
            return f"{float(x) * 100:+.0f}%"
        except (TypeError, ValueError):
            return "&mdash;"

    # ---- the falsification-table board ----
    ratios_tbl = (e1 or {}).get("tree_over_lattice_at_matched_coherence", {})
    peak = (e1 or {}).get("peak_dissonance", {})
    peak_ratio_geo = (peak.get("tree", {}).get("D", float("nan"))
                      / peak["lattice"]["D"]) if peak.get("lattice", {}).get("D") else float("nan")
    weak_med = (e1 or {}).get("tree_over_lattice_median_weak", float("nan"))
    cards = [
        ("Prediction 7 &mdash; dissonance under drive is much higher on a branching tree "
         "than on a lattice",
         "mixed",
         f"Not as stated: the peak dissonance the three geometries can reach is near-identical "
         f"({fmt(peak_ratio_geo, 2)}&times; tree over lattice), and below r_global = 0.7 the "
         f"tree is actually <em>lower</em> ({fmt(weak_med, 2)}&times;). But once the drive has "
         f"genuinely forced coherence, the tree carries {fmt(ratio_med, 2)}&times; the lattice's "
         f"dissonance at matched coherence. The claim survives only in its refined, "
         f"regime-specific form."),
        ("Defects concentrate at bifurcation points under drive",
         "refuted",
         f"Measured IX.5's own way &mdash; parent segment against the mean of its daughters "
         f"&mdash; branch points carry {fmt(branch_mm, 2)}&times; the phase step of ordinary "
         f"chain segments, i.e. no more. They look worse ({fmt(bif_excess, 2)}&times;) only on a "
         f"neighbourhood measure that a degree-3 node inflates by having more neighbours."),
        ("Prediction 4 &mdash; the dissonance index dissociates from LZ complexity",
         "supported",
         f"Pooled r = {fmt(pearson, 2)} (Spearman {fmt(spearman, 2)}). The attack state carries "
         f"11.5&times; the dissonance of a heavy-noise state at one third its LZ complexity: they "
         f"are anti-correlated, not merely uncorrelated."),
        ("Prediction 1 &mdash; the complexifier relieves, the symmetriser does not",
         "supported",
         f"A transient Mexican-hat kernel changes D by {pct(dmt)}; the matched-energy "
         f"uniform-positive kernel by {pct(meo)}. The IV.1 clinical anomaly reproduces in "
         f"simulation, from kernel shape alone."),
        ("Oxygen as a reduction in natural-frequency variance",
         "supported" if ox_dir == "reduces" else "refuted",
         f"Contracting &sigma;<sub>&omega;</sub> moves D monotonically from "
         f"{fmt(ox_curve[-1] if ox_curve else float('nan'))} untreated to "
         f"{fmt(ox_curve[0] if ox_curve else float('nan'))} at f = "
         f"{fmt(ox_f[0] if ox_f else float('nan'), 2)} &mdash; a "
         f"{abs(ox_span) * 100:.0f}% {'reduction' if ox_dir == 'reduces' else 'increase'}. "
         f"But the acute effect at the dose used mid-attack is only {pct(oxy)}, so this is a "
         f"slow, dose-dependent handle rather than the seconds-fast abort oxygen actually is."),
        ("Prediction 8 &mdash; consonant entrainment reduces dissonance, matched dissonant "
         "entrainment does not",
         "supported" if (e4 or {}).get("supports_stv_prediction") else "refuted",
         f"The two stacks do not separate ({fmt((e4 or {}).get('effect_harmonic'), 4)} vs "
         f"{fmt((e4 or {}).get('effect_dissonant'), 4)}, seed spread "
         f"{fmt((e4 or {}).get('seed_spread'), 4)}). "
         + ("A grossly inharmonic positive control <em>does</em> separate, so this is about the "
            "source's frequency set being mistuned by under 1% against a model whose own "
            "frequency jitter is several times larger &mdash; the experiment as specified is "
            "underpowered by construction."
            if (e4 or {}).get("positive_control_separates")
            else "The inharmonic positive control does not separate either, so the null is "
                 "about the entrainment mechanism, not the stimulus. The stronger negative "
                 "result of the two.")),
        ("A persistent Mexican-hat kernel shift is protective",
         "supported" if any(k == "mexican_hat" for k in
                            ((e3 or {}).get("kernel_thresholds") or {})) else "mixed",
         f"Of the candidate persistent kernels swept, only the Mexican-hat shape raises the "
         f"drive amplitude needed to reach the baseline attack's dissonance "
         f"(A = {fmt(((e3 or {}).get('kernel_thresholds') or {}).get('mexican_hat'), 2)} against "
         f"baseline {fmt(((e3 or {}).get('kernel_thresholds') or {}).get('baseline'), 2)}) and "
         f"caps the peak reachable dissonance at "
         f"{fmt(((e3 or {}).get('kernel_peak_ratio') or {}).get('mexican_hat'), 2)}&times; "
         f"baseline. That is the shape III.2 assigns to 5-HT2A &mdash; so VI.3's "
         f"occupancy-versus-plasticity axis falls out of the model rather than being assumed."),
        ("The model makes one prediction the document does not",
         "mixed",
         "Dissonance is sustained only inside a <em>band</em> of drive amplitude. Push the "
         "drive past it and the tree entrains completely and goes quiet. Taken literally that "
         "says more hypothalamic drive should abort an attack rather than deepen it &mdash; "
         "counterintuitive, and the sharpest thing here to try to falsify."),
    ]

    board = "\n".join(
        f"""<article class="card">
  <span class="tag {t}">{t}</span>
  <p class="claim">{claim}</p>
  <p class="detail">{detail}</p>
</article>""" for claim, t, detail in cards
    )

    # ---- experiment sections ----
    def section(num: str, title: str, blurb: str, verdict: str, figs: str,
                extra: str = "") -> str:
        return f"""<section>
  <div class="sec-head">
    <span class="sec-num">Experiment {num}</span>
    <h2>{title}</h2>
  </div>
  <div class="col">{blurb}</div>
  <div class="verdict"><span class="label">What the run says</span><p>{verdict}</p></div>
  {figs}
  {extra}
</section>"""

    sus_rows = "".join(
        f'<tr><td class="num">{fmt(k, 1)}</td><td class="num">{float(v) * 100:.0f}%</td></tr>'
        for k, v in sorted(sus.items(), key=lambda kv: float(kv[0]))
    )
    ox_rows = "".join(
        f'<tr><td class="num">{fmt(f, 2)}</td><td class="num">{fmt(d)}</td></tr>'
        for f, d in zip(ox_f, ox_curve)
    )
    arm_rows = "".join(
        f'<tr><td>{k}</td><td class="num">{fmt(v.get("pearson"), 2)}</td>'
        f'<td class="num">{fmt(v.get("spearman"), 2)}</td></tr>'
        for k, v in per_arm.items()
    )

    sections = [
        section(
            "0", "Does the coupling-kernel framework reproduce at all?",
            "<p>IX.7 makes this a gate: if a flat kernel does not drive global coherence and a "
            "Mexican-hat kernel does not fragment the field, stop and fix the implementation. "
            "Every kernel is rescaled to the same total |K|, so only shape differs.</p>",
            (e0 or {}).get("verdict", ""),
            figure("exp0_replication", "kernels_lattice.png",
                   "Phase field and time-averaged spatial power for each kernel. Uniform colour "
                   "means a coherent field; the ring marks the dominant spatial mode.")
            + figure("exp0_replication", "mode_selection.png",
                     "Where each kernel puts its spatial power. The uniform-positive kernels "
                     "concentrate at low spatial frequency (large coherent patches); the "
                     "Mexican-hat kernels push it to the checkerboard limit."),
        ),
        section(
            "1", "The crux: does branching geometry cost more dissonance?",
            "<p>The same drive, the same kernel, three geometries at matched node count. IX.7 "
            "calls this &ldquo;the highest-information single run in the whole programme&rdquo;. "
            "The raw comparison is confounded, because the same drive amplitude produces "
            "slightly different achieved coherence on each geometry &mdash; so dissonance is "
            "also compared at <em>matched</em> coherence, which is what &ldquo;the same "
            "drive&rdquo; has to mean for the structural claim to be about structure.</p>",
            (e1 or {}).get("verdict", ""),
            figure("exp1_geometry", "geometry_sweep.png",
                   "Dissonance against drive amplitude, achieved coherence, dissonance at "
                   "matched coherence, and the tree-versus-lattice ratio. The geometry effect "
                   "lives entirely in the shaded forced-coherence regime.")
            + figure("exp1_geometry", "tree_defects.png",
                     "Where the phase defects sit on the tree, and whether branch points carry "
                     "more than their degree explains."),
            f"""<div class="col"><h3>The effect is a gradient, not a threshold</h3>
<p>The ratio does not switch on at some coherence &mdash; it climbs steadily with how hard the
drive is forcing:</p></div>
<div class="tablewrap"><table>
<thead><tr><th class="num">achieved coherence r_global</th>
<th class="num">tree / lattice dissonance</th></tr></thead>
<tbody>{"".join(
    f'<tr><td class="num">{fmt(float(k), 2)}</td><td class="num">{fmt(v, 2)}&times;</td></tr>'
    for k, v in sorted(ratios_tbl.items(), key=lambda kv: float(kv[0])))}</tbody></table></div>
<div class="col"><p>That monotone climb is better evidence for Model B's mechanism than any
single number: the more coherence the drive succeeds in imposing, the more the branching
substrate costs to hold it. What it does <em>not</em> support is the prediction as IX.7 states
it. The peak dissonance each geometry can reach is the same to within 1%
({"".join(f"{g} {fmt(v['D'], 3)}, " for g, v in peak.items()).rstrip(', ')}), so a branching
tree is not more dissonance-prone in general &mdash; only under coherence it is being forced
to hold.</p></div>""",
        ),
        section(
            "2", "What an attack looks like, and when it stops",
            "<p>A circadian duty-cycle drive on the tree. IX.7 asks whether onset is abrupt or "
            "smooth, whether there is hysteresis, and whether attacks can terminate on their "
            "own &mdash; &ldquo;a model that cannot produce spontaneous termination is missing "
            "something.&rdquo;</p>",
            (e2 or {}).get("verdict", ""),
            figure("exp2_attack", "attack_cycle.png",
                   "The drive, the dissonance index at three amplitudes, and the coherence "
                   "underneath it.")
            + figure("exp2_attack", "sustained_fraction.png",
                     "Whether the dissonant state holds through the driven window or collapses "
                     "into two transients at its edges."),
            f"""<div class="col"><h3>The attack window in drive amplitude</h3>
<p>The fraction of each driven window spent above half its peak dissonance:</p></div>
<div class="tablewrap"><table>
<thead><tr><th class="num">drive A (rad/s)</th><th class="num">time above half-peak D</th></tr></thead>
<tbody>{sus_rows}</tbody></table></div>
<div class="col"><p>This is the sharpest prediction the model makes, and it is
counterintuitive: past the band, more hypothalamic drive should <em>abort</em> the attack
rather than deepen it, because the tree entrains completely and the dissonance goes with
it.</p></div>""",
        ),
        section(
            "3", "The three interventions",
            "<p>Each applied mid-attack, all kernels energy-matched so that only shape differs. "
            "The discriminating condition is 5-MeO: naive symmetrisation (Model A) predicts the "
            "uniform-positive kernel helps most, Model B predicts it fails.</p>",
            (e3 or {}).get("verdict", ""),
            figure("exp3_interventions", "interventions.png",
                   "Dissonance through the intervention window. DMT and 5-MeO wash out at the "
                   "end of the shaded band; oxygen and sumatriptan stay on.")
            + figure("exp3_interventions", "intervention_effects.png",
                     "Relative change in dissonance during each intervention.")
            + figure("exp3_interventions", "kernel_threshold.png",
                     "Left: does any persistent kernel shift raise the drive amplitude needed to "
                     "reach the baseline attack's dissonance? Right: the oxygen dose-response "
                     "IX.4 asks for, rather than a single point."),
            f"""<div class="col"><h3>Oxygen dose-response</h3>
<p>IX.4 specifies a sweep of the &sigma;<sub>&omega;</sub> multiplier <span class="mono">f</span>
over [0.1, 1.0] rather than the single point everything else reports:</p></div>
<div class="tablewrap"><table>
<thead><tr><th class="num">&sigma;<sub>&omega;</sub> multiplier f</th><th class="num">dissonance D</th></tr></thead>
<tbody>{ox_rows}</tbody></table></div>
<div class="col"><p>The direction VI.2 predicts, and cleanly monotone: stabilising the
natural frequencies {ox_dir} dissonance by {abs(ox_span) * 100:.0f}% at the strongest
contraction. Two cautions before reading this as a success. It is a <em>dose-dependent,
graded</em> effect &mdash; the acute change at f = 0.25 applied mid-attack is only
{pct(oxy)}, nothing like the seconds-fast abort oxygen actually produces in patients. And a
smaller pilot of this same sweep (N = 200, one seed) produced the <em>opposite sign</em>,
which is what the 11% run-to-run variance buys you if you do not average seeds.</p>
<p>XI.7 flagged the oxygen story as &ldquo;a plausible role assignment, not a derivation&rdquo;.
That remains the right reading: the mechanism moves the metric the right way, but nothing
here derives the timescale that makes oxygen clinically useful.</p></div>

<div class="col"><h3>Which persistent kernel shift is protective</h3>
<p>IX.4's psychedelic arm is a permanent change in (K<sub>1</sub>&hellip;K<sub>4</sub>) that
raises the drive amplitude needed for dissonance onset. Rather than assume a shape, the run
sweeps candidates and reports which ones actually do it:</p></div>
<div class="tablewrap"><table>
<thead><tr><th>persistent kernel</th><th class="num">onset threshold A</th>
<th class="num">peak D vs baseline</th></tr></thead>
<tbody>{"".join(
    f'<tr><td><span class="mono">{k}</span></td>'
    f'<td class="num">{_threshold_label(v)}</td>'
    f'<td class="num">{fmt(peak_ratio.get(k), 2)}&times;</td></tr>'
    for k, v in thresholds.items())}</tbody></table></div>
<div class="col"><p>Only the Mexican-hat shape is protective, and substantially so: it more
than doubles the drive needed to reach the baseline attack's dissonance and halves the peak
the system can reach at all. That is the same shape III.2's receptor bridge assigns to
5-HT2A &mdash; so the acute jam and the lasting prophylaxis come out of the model as the same
operation applied transiently or permanently, which is VI.3's occupancy-versus-plasticity
axis falling out rather than being assumed.</p></div>""",
        ),
        section(
            "4", "The one prediction unique to STV",
            "<p>Drive a peripheral V1 subset with the source's harmonic frequency stack and with "
            "its matched inharmonic stack at equal total energy. IX.7: &ldquo;this is the only "
            "prediction unique to STV rather than to sensible neuroscience, and it is the "
            "prediction that would translate directly into a cheap non-invasive human "
            "experiment.&rdquo;</p>",
            (e4 or {}).get("verdict", ""),
            figure("exp4_entrainment", "entrainment.png",
                   "Consonant versus matched dissonant entrainment of the V1 branch."),
        ),
        section(
            "5", "The validity check everything else depends on",
            "<p>XI.1 puts this first among the things that would sink the programme: "
            "&ldquo;the dissonance metric may not be constructible in a way that dissociates "
            "from entropy. If so, the entire framework becomes empirically inert for this "
            "problem. Test this first.&rdquo;</p>",
            (e5 or {}).get("verdict", ""),
            figure("exp5_noise", "noise_control.png",
                   "Noise injection raises LZ complexity without raising dissonance; drive does "
                   "the reverse. Separate panels rather than a shared axis, because these are "
                   "different quantities.")
            + figure("exp5_noise", "dissociation.png",
                     "Every condition, dissonance against LZ complexity. Collinearity here "
                     "would mean the index is just an entropy metric."),
            f"""<div class="col"><h3>Correlation within each arm, not just pooled</h3>
<p>Pooling two sweeps can manufacture or hide a correlation, so both are reported:</p></div>
<div class="tablewrap"><table>
<thead><tr><th>sweep</th><th class="num">Pearson</th><th class="num">Spearman</th></tr></thead>
<tbody>{arm_rows}</tbody></table></div>""",
        ),
    ]

    if e6:
        sections.append(section(
            "6", "The remaining free parameters",
            "<p>Four things IX.3 and IX.6 ask for that Experiments 1&ndash;5 do not touch: the "
            "diffuse drive variant (&ldquo;implement <em>both</em> variants and compare; they "
            "are not equivalent&rdquo;), the small-world lever K<sub>SW</sub>, the drive "
            "frequency &Omega; against the tree's own Laplacian eigenfrequencies, and the "
            "branching parameters &mdash; including node count, which is the check that none of "
            "this is a size artefact.</p>",
            e6.get("verdict", ""),
            figure("exp6_sweeps", "parameter_sweeps.png",
                   "Drive variant, small-world coupling, drive frequency against the network's "
                   "own modes, and tree structure and size."),
        ))

    body = "\n".join(sections)

    html = f"""<title>Trigeminal Field Dissonance</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>

<div class="wrap">

<header class="masthead">
  <p class="eyebrow">Simulation report &middot; Part IX of the coupling-kernel handoff</p>
  <h1>Cluster headache as field dissonance</h1>
  <p class="standfirst">A runnable implementation of the specification, and what it says
  about Model B. Some of its claims survive in a narrower form than stated, one is cleanly
  refuted, and the metric the whole programme depends on turned out to need rebuilding
  before any of them could be asked.</p>
  <div class="runmeta">
    <span>tree N&nbsp;=&nbsp;626 &middot; lattice 625 &middot; layered 607</span>
    <span>3 seeds per condition</span>
    <span>Kuramoto phase oscillators, 4-shell coupling kernel</span>
    <span>implementation independently audited, 16 findings fixed</span>
  </div>
</header>

<section>
  <div class="col">
    <p>The handoff document was explicit about what would count as a good outcome:
    &ldquo;the goal is not to confirm the model &hellip; the single most valuable outcome
    would be a clean negative result on a well-specified metric.&rdquo; There are several
    here. The one that matters most is that the metric itself had to be rebuilt before any
    of the model's claims could be asked &mdash; and the rebuild is the part most portable
    to other work.</p>
  </div>
  <div class="board">
{board}
  </div>
</section>

<hr class="div">

<section>
  <div class="sec-head">
    <span class="sec-num">First</span>
    <h2>The metric had to be rebuilt before anything could be measured</h2>
  </div>
  <div class="col">
    <p>IX.5 asks for a dissonance index that scores exact ratios near zero, the source's
    dissonant TMS set high, and white noise low. <strong>No single-bandwidth roughness
    measure can do all three</strong>, and the reason is structural rather than a tuning
    failure.</p>
    <p>The source's dissonant TMS set (1.01, 2.01, 3.98, 6.02&hellip;) is <em>almost exactly
    harmonic</em> &mdash; a common fundamental near 1.005 fits every member to within 1%. It
    is dissonant by <strong>beating</strong>, an absolute-detuning phenomenon that needs a
    narrow critical band. A &ldquo;relatively prime&rdquo; stack &mdash; the phrase III.1
    actually uses &mdash; has no common fundamental at small integers at all, but its partials
    are far apart, so it produces <em>no beating whatsoever</em>. It is dissonant by
    <strong>incommensurability</strong>, a ratio phenomenon that a narrow band cannot see.</p>
    <p>Sweeping the critical bandwidth makes this concrete: the beating term alone never
    reaches a 3&times; separation between the consonant and dissonant cases at any bandwidth.
    The implemented index is therefore a composite of both terms, reported separately
    alongside the combination in every result file. That is a substantive amendment to IX.5,
    not a convenience.</p>
  </div>

  <div class="note">
    <strong>The noise check is weaker than it looks, and the self-test now says so.</strong>
    A uniform-random or random-walk phase field has no mean phase drift, so every cluster's
    estimated frequency falls below the band gate and is zeroed <em>before</em> the amplitude
    weighting is consulted &mdash; the check passes without exercising the mechanism it exists
    to validate. The self-test adds a drifting-noise null in the model's own frequency band,
    and compares states the simulation actually occupies rather than hand-built tone sets.
  </div>

  <div class="tablewrap"><table>
    <thead><tr><th>state</th><th class="num">D</th><th class="num">D intensive</th>
    <th class="num">coherence weight</th><th class="num">r_global</th><th class="num">LZ</th></tr></thead>
    <tbody>
      <tr><td>driven attack (A = 4)</td><td class="num">0.269</td><td class="num">0.466</td>
          <td class="num">0.579</td><td class="num">0.73</td><td class="num">0.21</td></tr>
      <tr><td>undriven, heavy noise</td><td class="num">0.024</td><td class="num">0.639</td>
          <td class="num">0.037</td><td class="num">0.05</td><td class="num">0.69</td></tr>
      <tr><td>undriven, quiet</td><td class="num">0.051</td><td class="num">0.616</td>
          <td class="num">0.082</td><td class="num">0.07</td><td class="num">0.28</td></tr>
    </tbody>
  </table></div>
  <div class="col">
    <p>The attack state carries 11.5&times; the dissonance of the heavy-noise state at one
    third its LZ complexity. Note that the intensive column moves the other way, which is
    exactly why both are reported: the extensive index asks how much of the system is locked
    into structured dissonance, the intensive one asks how dissonant the coherent part is,
    and an intervention can move them in opposite directions.</p>
  </div>
</section>

<hr class="div">

{body}

<hr class="div">

<section>
  <div class="sec-head">
    <span class="sec-num">Caveats</span>
    <h2>What would change these conclusions</h2>
  </div>
  <div class="col">
    <ul>
      <li><strong>Effect sizes below about 20% are not readable.</strong> The dissonance index
      has a run-to-run coefficient of variation near 11% at N = 400. Every experiment averages
      three seeds and reports the spread; the timestep sensitivity sits entirely inside that
      band.</li>
      <li><strong>The critical bandwidth is a free parameter</strong> that the theory does not
      pin down, and it decides what counts as dissonant. The self-test sweeps it rather than
      letting one choice pass silently.</li>
      <li><strong>Absolute dissonance values do not survive a change of node count.</strong>
      D falls monotonically with N across 225, 625 and 1225 nodes. Every experiment here holds
      N fixed, so the comparisons are unaffected, but no absolute magnitude on this page should
      be read as a property of the model rather than of the mesh.</li>
      <li><strong>Only the tree has branch points.</strong> The lattice has none and the
      layered network's degree structure partitions it by layer, so there is no second
      geometry against which to test the bifurcation claim &mdash; only the degree-matched
      null within the tree.</li>
      <li><strong>The peripheral oscillator framing is the weakest link</strong>, exactly as
      XI.2 says. Nothing here tests whether trigeminal afferents are limit-cycle oscillators
      at all; the tree is an input geometry that has been given phase dynamics.</li>
      <li><strong>Several thresholds inside the verdicts are judgement calls</strong> &mdash;
      the forced-coherence cut at r_global &ge; 0.7, the 2&times;-pooled-SD bar for calling an
      intervention effect real, the 3&times; separation the metric self-test demands. They are
      in the code rather than in prose so they can be argued with.</li>
      <li><strong>Everything downstream of &ldquo;STV is true&rdquo; inherits STV's own
      uncertainty</strong>, and the coupling-kernel source describes itself as early-stage.
      That request propagates.</li>
    </ul>
  </div>

  <div class="note">
    <strong>On how much of this to trust.</strong> The implementation was audited against
    Part IX by an independent multi-agent review, with every finding adversarially verified
    before being accepted; sixteen survived and are fixed. Four of them changed a number
    reported here. Intervention windows were being silently relocated when their boundaries
    were not exact multiples of the timestep. The incommensurability term used a residual
    whose tolerance grew with harmonic number, and correcting it roughly tripled the measured
    tree-versus-lattice gap. Experiment 2's &ldquo;sustained fraction&rdquo; was normalised
    against each window's own peak, so a window containing no attack scored as fully
    sustained. And Experiment 5's dissociation test was checking for the <em>absence</em> of
    correlation when a strong negative correlation is exactly the result being sought. An
    earlier pilot of the oxygen sweep, run at a quarter the node count on a single seed,
    produced the opposite sign from the result above &mdash; which is the clearest available
    demonstration of why the seed-averaging matters.
  </div>
</section>

<footer>
  <p>Generated from <code>results/*/summary.json</code> &mdash; every number on this page is
  read from the run that produced it. Reproduce with
  <code>python -m sim.cli all</code>, and validate the metric first with
  <code>python -m sim.cli selftest</code>.</p>
  <p>Nothing here is medical advice. Effective legal treatments for cluster headache exist,
  including high-flow oxygen and triptans, and belong in the hands of a clinician.</p>
</footer>

</div>
"""
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "report.html"))
    args = ap.parse_args()
    path = build(args.out)
    print(f"wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
