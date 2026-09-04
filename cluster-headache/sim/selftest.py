"""Sanity checks for the dissonance index (IX.5).

    a set of frequencies at exact ratios 1:2:3:4  must score near zero
    the source's dissonant TMS set                must score high
    white noise                                   must score low

The third is the one that matters.  "Any dissonance metric you build must distinguish
structured incommensurability from noise" - if it cannot, the metric is an entropy
metric and the whole exercise is uninformative.

The checks are stated as a **separation**: the worst dissonant case must score at least
`SEPARATION` times the best consonant case.  Absolute thresholds would be arbitrary,
since the index is a mean over tone pairs and most pairs in a large set contribute
nothing.

Running this also produces the document's most portable finding: the two halves of
IX.5's own sanity set need *different* measures.  The dissonant TMS frequencies are
almost exactly harmonic and are dissonant only by beating; an incommensurate stack has
no beating at all and is dissonant only by ratio.  A single-bandwidth roughness metric
cannot pass both, which is why the index is a composite.

Run:  python -m sim.cli selftest
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import DISSONANT_SET_HZ, HARMONIC_SET_HZ, MetricConfig, NetworkConfig
from .metrics import cluster_tones, complexity_metrics, dissonance_from_tones, order_parameters
from .networks import build_network

SEPARATION = 3.0

#: (name, frequencies, expected class)
CASES: List[Tuple[str, np.ndarray, str]] = [
    ("exact 1:2:3:4", np.array([1.0, 2.0, 3.0, 4.0]), "consonant"),
    ("exact 5:10:15:20 Hz", np.array([5.0, 10.0, 15.0, 20.0]), "consonant"),
    ("harmonic TMS set", np.asarray(HARMONIC_SET_HZ, dtype=float), "consonant"),
    ("major triad 4:5:6", 5.0 * np.array([1.0, 1.25, 1.5]), "consonant"),
    ("dissonant TMS set", np.asarray(DISSONANT_SET_HZ, dtype=float), "dissonant"),
    ("mistuned 1%", np.array([5.0, 10.05, 15.1, 20.2]), "dissonant"),
    ("golden-ratio stack", 5.0 * np.array([1.0, 1.618, 2.618, 4.236]), "dissonant"),
    ("relatively-prime stack", np.array([5.0, 7.13, 11.31, 17.77]), "dissonant"),
]

#: The critical bandwidth is the one lever IX.5 leaves undetermined, so it is swept
#: rather than chosen silently.  See `roughness_sethares` for why it matters.
BAND_WIDTHS: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.06, 0.12)


def tone_checks(cfg: MetricConfig) -> List[Dict[str, object]]:
    """Synthetic tone sets pushed straight through the index, at full coherence."""
    out: List[Dict[str, object]] = []
    for name, freqs, kind in CASES:
        d = dissonance_from_tones(freqs, np.ones(freqs.size), cfg)
        out.append({
            "case": name,
            "kind": kind,
            "dissonance": d["dissonance"],
            "d_roughness": d["d_roughness"],
            "d_inharmonic": d["d_inharmonic"],
        })
    return out


def separation(rows: List[Dict[str, object]], key: str = "dissonance") -> Dict[str, float]:
    cons = [float(r[key]) for r in rows if r["kind"] == "consonant"]
    diss = [float(r[key]) for r in rows if r["kind"] == "dissonant"]
    worst_c = max(cons) if cons else 0.0
    worst_d = min(diss) if diss else 0.0
    ratio = worst_d / worst_c if worst_c > 1e-9 else float("inf")
    return {"worst_consonant": worst_c, "worst_dissonant": worst_d, "ratio": ratio,
            "pass": bool(ratio >= SEPARATION)}


def noise_check(cfg: MetricConfig, n_target: int = 400, n_t: int = 800, seed: int = 0) -> Dict[str, object]:
    """White noise pushed through the *full* pipeline, clusters and all.

    Two fields: a uniform-random phase field (maximally incoherent) and a phase random
    walk (realistic temporal autocorrelation, and the harder case, because each cluster
    then does have a wandering mean phase).
    """
    rng = np.random.default_rng(seed)
    net = build_network(NetworkConfig(geometry="tree", n_target=n_target, seed=seed))
    fs = 50.0
    t = np.arange(n_t)[:, None] / fs

    # The third field is the one that matters.  The first two have zero mean phase drift,
    # so every cluster's estimated frequency falls below freq_min_hz and the band gate
    # zeroes the amplitudes - the check would pass even if the amplitude weighting did
    # nothing.  "drifting noise" gives each node a real natural frequency in the model's
    # own band plus a Wiener phase, so it clears the band gate and the index has to
    # discriminate it on structure alone.  That is the honest null.
    drift_f = rng.normal(4.0, 1.5, size=net.n)
    fields = {
        "uniform white noise": rng.uniform(0, 2 * np.pi, size=(n_t, net.n)),
        "phase random walk": (
            np.cumsum(rng.normal(0, 0.35, size=(n_t, net.n)), axis=0)
            + rng.uniform(0, 2 * np.pi, size=net.n)
        ),
        "drifting noise (in band)": (
            2 * np.pi * drift_f[None, :] * t
            + np.cumsum(rng.normal(0, 0.35, size=(n_t, net.n)), axis=0)
            + rng.uniform(0, 2 * np.pi, size=net.n)
        ),
    }

    out: Dict[str, object] = {}
    for name, field in fields.items():
        orders = order_parameters(field, net)
        freqs, amps, _ = cluster_tones(orders["z_cluster"], fs, cfg)
        d = dissonance_from_tones(freqs, amps, cfg)
        lz = complexity_metrics(field, net, cfg)
        out[name] = {
            "dissonance": d["dissonance"],
            "roughness_shape": d["roughness_shape"],
            "inharmonicity": d["inharmonicity"],
            "coherence_weight": d["coherence_weight"],
            "n_tones": d["n_tones"],
            "r_local_mean": orders["r_local_mean"],
            "lz_normalised": lz["lz_normalised"],
        }
    return out


def contrast_check(cfg: MetricConfig, n_target: int = 400, seed: int = 0) -> Dict[str, object]:
    """The criterion that actually transfers: does the index separate a *simulated* attack
    state from a *simulated* noise state?

    An absolute pass threshold on synthetic tone sets does not transfer to the pipeline,
    because the index divides by all cluster pairs and so depends on how many clusters stay
    coherent.  This runs the model itself - driven at the amplitude that maximises D, and
    undriven with heavy noise - and reports the ratio between them.
    """
    from .config import RunConfig, apply_overrides
    from .dynamics import Simulator
    from .metrics import analyse

    out: Dict[str, object] = {}
    for label, over in (
        ("driven attack", {"drive.amplitude": 4.0, "dynamics.sigma_noise": 0.05}),
        ("undriven, heavy noise", {"drive.amplitude": 0.0, "dynamics.sigma_noise": 3.0}),
        ("undriven, quiet", {"drive.amplitude": 0.0, "dynamics.sigma_noise": 0.05}),
    ):
        run = apply_overrides(RunConfig(), {
            "network.geometry": "tree", "network.n_target": n_target,
            "network.seed": seed, "dynamics.seed": seed, "dynamics.t_total": 20.0, **over,
        })
        run.metrics = cfg
        res = Simulator(run).run()
        t, th = res.steady()
        st = analyse(th, t, res.net, res.fs, cfg, with_complexity=True)
        out[label] = {k: float(st[k]) for k in
                      ("dissonance", "dissonance_intensive", "coherence_weight",
                       "r_global_mean", "lz_normalised", "n_tones")}

    attack = out["driven attack"]["dissonance"]
    noise = out["undriven, heavy noise"]["dissonance"]
    out["attack_over_noise"] = float(attack / noise) if noise > 1e-12 else float("inf")
    out["pass"] = bool(out["attack_over_noise"] >= SEPARATION)
    return out


def band_sweep(base: MetricConfig | None = None) -> List[Dict[str, object]]:
    """Separation achieved by each component, as a function of critical bandwidth."""
    base = base or MetricConfig()
    rows: List[Dict[str, object]] = []
    for bw in BAND_WIDTHS:
        cfg = MetricConfig(**{**base.__dict__, "band_width": bw})
        checks = tone_checks(cfg)
        rows.append({
            "band_width": bw,
            "composite": separation(checks, "dissonance"),
            "roughness_only": separation(checks, "d_roughness"),
            "inharmonic_only": separation(checks, "d_inharmonic"),
        })
    return rows


def model_checks(base: MetricConfig) -> List[Dict[str, object]]:
    """Run the tone checks under each roughness model, so an alternative that fails IX.5's
    first requirement cannot sit unexercised behind a config flag."""
    rows: List[Dict[str, object]] = []
    for model in ("composite", "sethares", "harmonicity"):
        cfg = MetricConfig(**{**base.__dict__, "roughness_model": model})
        checks = tone_checks(cfg)
        rows.append({
            "model": model,
            "separation": separation(checks, "dissonance"),
            "worst_consonant_case": max(checks, key=lambda c: c["dissonance"] if c["kind"] == "consonant" else -1)["case"],
            "values": {c["case"]: c["dissonance"] for c in checks},
        })
    return rows


def run(cfg: MetricConfig | None = None) -> Dict[str, object]:
    cfg = cfg or MetricConfig()
    tones = tone_checks(cfg)
    sep = {k: separation(tones, k) for k in ("dissonance", "d_roughness", "d_inharmonic")}
    noise = noise_check(cfg)
    contrast = contrast_check(cfg)
    sweep = band_sweep(cfg)
    models = model_checks(cfg)
    all_pass = sep["dissonance"]["pass"] and contrast["pass"]
    return {
        "tone_checks": tones,
        "separation": sep,
        "noise_checks": noise,
        "contrast_check": contrast,
        "band_sweep": sweep,
        "model_checks": models,
        "all_pass": all_pass,
        "metric_config": cfg.__dict__,
    }


def report(result: Dict[str, object]) -> str:
    L = ["Dissonance-index sanity checks (IX.5)", ""]
    L.append(f"{'case':<24} {'kind':<10} {'D':>8} {'beating':>9} {'inharm':>8}")
    for t in result["tone_checks"]:
        L.append(
            f"{t['case']:<24} {t['kind']:<10} {t['dissonance']:>8.4f}"
            f" {t['d_roughness']:>9.4f} {t['d_inharmonic']:>8.4f}"
        )
    L.append("")
    L.append(f"Separation (worst dissonant / worst consonant; need >= {SEPARATION:g}x):")
    for name, s in result["separation"].items():
        mark = "ok" if s["pass"] else "FAIL"
        L.append(
            f"  {name:<14} {s['worst_dissonant']:.4f} / {s['worst_consonant']:.4f}"
            f" = {s['ratio']:>6.2f}x  {mark}"
        )

    L.append("")
    L.append("Synthetic noise fields (the first two never clear the frequency band gate,")
    L.append("so only the third actually exercises the amplitude weighting):")
    L.append(f"{'noise case':<26} {'D':>8} {'shape':>8} {'inharm':>8} {'coh_w':>8} {'tones':>6} {'LZ':>7}")
    for name, v in result["noise_checks"].items():
        L.append(
            f"{name:<26} {v['dissonance']:>8.4f} {v['roughness_shape']:>8.3f}"
            f" {v['inharmonicity']:>8.3f} {v['coherence_weight']:>8.4f}"
            f" {v['n_tones']:>6.0f} {v['lz_normalised']:>7.3f}"
        )

    L.append("")
    L.append("Simulated states - the criterion that transfers to the experiments:")
    c = result["contrast_check"]
    L.append(f"{'state':<26} {'D':>8} {'D_int':>8} {'coh_w':>8} {'r_glob':>8} {'LZ':>7}")
    for name in ("driven attack", "undriven, heavy noise", "undriven, quiet"):
        v = c[name]
        L.append(
            f"{name:<26} {v['dissonance']:>8.4f} {v['dissonance_intensive']:>8.4f}"
            f" {v['coherence_weight']:>8.4f} {v['r_global_mean']:>8.3f}"
            f" {v['lz_normalised']:>7.3f}"
        )
    L.append(f"  attack / heavy-noise dissonance ratio = {c['attack_over_noise']:.2f}x"
             f"  (need >= {SEPARATION:g})  {'ok' if c['pass'] else 'FAIL'}")

    L.append("")
    L.append("Critical-bandwidth sweep - separation by component:")
    L.append(f"{'band_width':>10} {'composite':>12} {'beating':>12} {'inharmonic':>12}")
    for row in result["band_sweep"]:
        L.append(
            f"{row['band_width']:>10.3f} {row['composite']['ratio']:>12.2f}"
            f" {row['roughness_only']['ratio']:>12.2f} {row['inharmonic_only']['ratio']:>12.2f}"
        )

    L.append("")
    L.append("Roughness models (all three exercised, not just the default):")
    L.append(f"{'model':>14} {'separation':>12}  worst consonant case")
    for row in result["model_checks"]:
        L.append(f"{row['model']:>14} {row['separation']['ratio']:>12.2f}"
                 f"  {row['worst_consonant_case']}")

    L.append("")
    L.append("ALL CHECKS PASS" if result["all_pass"] else "SOME CHECKS FAILED")
    L.append(
        "\nReading the tables. The synthetic-noise rows are weaker evidence than they look:"
        "\nthe uniform and random-walk fields have no mean phase drift, so their clusters"
        "\nfall below freq_min_hz and are gated out before the amplitude weighting is even"
        "\nconsulted. The 'drifting noise' row is the honest null - it sits in the model's"
        "\nown frequency band - and the simulated-states table is the criterion that"
        "\ntransfers, because it compares the index on states the simulation actually"
        "\noccupies rather than on hand-built tone sets."
    )
    return "\n".join(L)
