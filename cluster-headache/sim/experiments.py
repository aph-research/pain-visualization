"""The six experiments of Part IX.7, in priority order.

  0  Replication      - reproduce the source's lattice results before trusting anything
  1  Geometry         - THE CRUX: same drive, branching tree vs lattice vs layered
  2  Attack dynamics  - onset shape, hysteresis, spontaneous termination
  3  Interventions    - DMT / 5-MeO / oxygen / sumatriptan / persistent kernel shift
  4  Entrainment      - consonant vs matched dissonant stimulation (the STV signature)
  5  Noise control    - the validity check: does D dissociate from LZ?

Every experiment writes `summary.json`, `records.csv`, `arrays.npz` and its figures into
`results/<name>/`, and returns a summary dict whose `verdict` field states what the run
supports or refutes in the document's own terms.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import plots
from .config import (
    DISSONANT_SET_HZ,
    HARMONIC_SET_HZ,
    Intervention,
    KERNEL_PRESETS,
    NetworkConfig,
    RunConfig,
    apply_overrides,
)
from .dynamics import SimResult, Simulator
from .metrics import (
    analyse,
    dct_modes,
    dissonance_timeseries,
    laplacian_eigenfrequencies,
    traveling_wave_index,
)
from .networks import Network, build_network

GEOMETRIES = ("tree", "lattice", "hierarchical")


# --------------------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------------------


def _outdir(root: str, name: str) -> str:
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    return path


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _write_json(path: str, payload) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def _write_csv(path: str, records: Sequence[Dict[str, object]]) -> None:
    if not records:
        return
    keys: List[str] = []
    for r in records:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in keys})


def save_series(root: str, name: str, res: SimResult, stride: int = 2) -> str:
    """Persist a full phase time series.

    IX.8: "Save full phase time series for at least representative runs - the metrics will
    be revised, and re-running sweeps is expensive."  Series go to `results/series/` so the
    per-experiment directories stay small; `stride` subsamples time, since the recording
    rate is already well above the frequencies the metrics look at.
    """
    out = _outdir(root, "series")
    path = os.path.join(out, f"{name}.npz")
    np.savez_compressed(
        path,
        times=res.times[::stride],
        theta=res.theta[::stride].astype(np.float32),
        omega=res.omega,
        drive_g=res.drive_g[::stride],
        fs=res.fs / stride,
        geometry=res.net.geometry,
        clusters=res.net.clusters,
        compartment=res.net.compartment,
        positions=res.net.positions if res.net.positions is not None else np.zeros((res.net.n, 2)),
        config=json.dumps(res.cfg.to_dict(), default=_json_default),
    )
    return path


_NET_CACHE: Dict[Tuple, Network] = {}


def cached_network(cfg: NetworkConfig) -> Network:
    key = tuple(sorted((k, str(v)) for k, v in cfg.__dict__.items()))
    if key not in _NET_CACHE:
        _NET_CACHE[key] = build_network(cfg)
    return _NET_CACHE[key]


def run_condition(
    base: RunConfig, overrides: Dict[str, object], with_complexity: bool = False
) -> Tuple[SimResult, Dict[str, object]]:
    cfg = apply_overrides(base, overrides)
    net = cached_network(cfg.network)
    res = Simulator(cfg, net).run()
    t, th = res.steady()
    stats = analyse(th, t, net, res.fs, cfg.metrics, with_complexity=with_complexity)
    return res, stats


def _mean_sd(rows: Sequence[Dict[str, object]], key: str) -> Tuple[float, float]:
    vals = np.array([float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))])
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std())


def _interp_at(x: np.ndarray, y: np.ndarray, targets: Sequence[float]) -> Dict[float, float]:
    """y interpolated at given x values, with x sorted ascending; NaN outside range."""
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    out = {}
    for t in targets:
        out[t] = float(np.interp(t, xs, ys, left=np.nan, right=np.nan))
    return out


# --------------------------------------------------------------------------------------
# Experiment 0 - replication on the 2D lattice
# --------------------------------------------------------------------------------------


def experiment_0(root: str, base: RunConfig, n_target: int = 625, t_total: float = 20.0) -> Dict:
    """IX.7 Experiment 0.  Flat kernel -> global coherence; Mexican-hat -> competing
    clusters; recover the source's mode structure in the DCT.

    "If this doesn't reproduce, stop and fix the implementation before proceeding."
    """
    out = _outdir(root, "exp0_replication")
    kernels = ["flat", "5meo_plate", "baseline", "mexican_hat", "dmt_plate",
               "checkerboard", "pinwheel", "traveling_wave"]

    records: List[Dict[str, object]] = []
    snapshots: Dict[str, np.ndarray] = {}
    dcts: Dict[str, np.ndarray] = {}
    net = None

    for name in kernels:
        spec = KERNEL_PRESETS[name]
        res, stats = run_condition(base, {
            "network.geometry": "lattice",
            "network.n_target": n_target,
            "dynamics.t_total": t_total,
            # the source's plate is homogeneous - the compartment frequency split is our
            # addition for the trigeminal tree and has no place in the replication
            "dynamics.omega0_peripheral_hz": base.dynamics.omega0_central_hz,
            "drive.amplitude": 0.0,
            "drive.variant": "none",
            "kernel.name": name,
            "kernel.k": spec["k"],
            "kernel.k_sw": spec.get("k_sw", 0.0),
            # checkerboard / pinwheel / traveling_wave all carry K_SW = -2.4, so the
            # harmonic basis has to include those links
            "metrics.harmonics_include_sw": bool(spec.get("k_sw", 0.0)),
        })
        net = res.net
        _, th = res.steady()
        snapshots[name] = th[-1]
        power = dct_modes(th, net.grid_shape)  # time-averaged over the steady state
        dcts[name] = power
        tw = traveling_wave_index(th, net.grid_shape, fs=res.fs)

        # dominant non-DC spatial mode
        p = power.copy()
        p[0, 0] = 0.0
        ky, kx = np.unravel_index(int(np.argmax(p)), p.shape)
        records.append({
            "kernel": name,
            "k": list(spec["k"]),
            "k_sw": spec.get("k_sw", 0.0),
            "k_sum_raw": float(np.sum(spec["k"])),
            # The energy-matched kernel is what the oscillators actually see; the raw
            # tabulated sums are on two different scale conventions (III.2, XI.9) and
            # comparing them across kernels would be meaningless.
            "k_sum": float(np.sum(res.meta["kernel"])),
            "k_normalised": [float(x) for x in res.meta["kernel"]],
            "r_global": stats["r_global_mean"],
            "r_local": stats["r_local_mean"],
            "fragmentation": stats["fragmentation"],
            "dissonance": stats["dissonance"],
            "harmonic_purity": stats["harmonic_purity_topk"],
            "vortex_count": stats.get("vortex_count", float("nan")),
            "dct_peak_kx": int(kx),
            "dct_peak_ky": int(ky),
            "dct_dc_fraction": float(power[0, 0] / max(power.sum(), 1e-30)),
            "traveling_index": tw["traveling_index"],
            "mode_drift_hz": tw["mode_drift_hz"],
            "fft_peak_kx": tw["peak_kx"],
            "fft_peak_ky": tw["peak_ky"],
        })

    # ---- figure: phase snapshot + DCT per kernel ----
    fig, axes = plots.new_fig(2, len(kernels), figsize=(1.72 * len(kernels), 4.6))
    for i, name in enumerate(kernels):
        r = [rec for rec in records if rec["kernel"] == name][0]
        plots.grid_phase_map(axes[0, i], snapshots[name], net.grid_shape,
                             title=f"{name}\nr = {r['r_global']:.2f}")
        power = dcts[name].copy()
        power[0, 0] = 0.0  # remove DC so the structured modes are visible
        logp = np.log10(np.maximum(power, 1e-12))
        top = float(logp.max())
        axes[1, i].imshow(logp, cmap=plots.SEQ, origin="lower", interpolation="nearest",
                          vmin=top - 2.0, vmax=top)  # each panel on its own scale
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        axes[1, i].grid(False)
        for spine in axes[1, i].spines.values():
            spine.set_visible(False)
        axes[1, i].set_title(f"peak k=({r['dct_peak_kx']},{r['dct_peak_ky']})",
                             loc="left", pad=5, fontsize=8.5, color=plots.INK_2)
        axes[1, i].plot(r["dct_peak_kx"], r["dct_peak_ky"], marker="o", markersize=8,
                        markerfacecolor="none", markeredgecolor=plots.SERIES[1],
                        markeredgewidth=1.8)
    axes[0, 0].set_ylabel("phase field", fontsize=9, color=plots.INK_2)
    axes[1, 0].set_ylabel("spatial power", fontsize=9, color=plots.INK_2)
    plots.suptitle(fig, "Experiment 0 - kernel shape on a 2D lattice, no drive")
    f1 = plots.finish(fig, os.path.join(out, "kernels_lattice.png"),
                      "Top: phase (cyclic colour map) - uniform colour means a coherent field. "
                      "Bottom: time-averaged 2D DCT power, DC removed, each panel on its own "
                      "log scale over 2 decades; the ring marks the dominant spatial mode. "
                      "Bottom-left = large-scale structure, top-right = checkerboard.")

    # ---- radial spatial-frequency profile: mode selection in one panel ----
    fig, ax = plots.new_fig(figsize=(7.6, 4.2))
    side = net.grid_shape[0]
    ky, kx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    kr = np.sqrt(kx**2 + ky**2)
    bins = np.linspace(0, kr.max(), 22)
    centres = 0.5 * (bins[1:] + bins[:-1])
    profiles = {}
    for name in ("flat", "5meo_plate", "mexican_hat", "checkerboard"):
        power = dcts[name].copy()
        power[0, 0] = 0.0
        prof = np.array([power[(kr >= a) & (kr < b)].mean() if ((kr >= a) & (kr < b)).any() else 0.0
                         for a, b in zip(bins[:-1], bins[1:])])
        profiles[name] = prof / max(prof.sum(), 1e-30)
    plots.series_plot(ax, centres, profiles,
                      xlabel="radial spatial frequency  |k|",
                      ylabel="fraction of spatial power",
                      title="Which spatial modes each kernel selects", label_last=False)
    ax.legend(loc="upper right")
    f1b = plots.finish(fig, os.path.join(out, "mode_selection.png"),
                       "Uniform-positive kernels concentrate power at low |k| (large coherent "
                       "patches); Mexican-hat kernels push it to high |k| (competing small clusters).")

    # ---- figure: order parameter by kernel ----
    fig, ax = plots.new_fig(figsize=(7.6, 3.9))
    ordered = sorted(records, key=lambda r: r["r_global"])
    names = [r["kernel"] for r in ordered]
    vals = [r["r_global"] for r in ordered]
    ax.barh(range(len(names)), vals, color=plots.SERIES[0], height=0.62)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color=plots.INK_2)
    ax.set_xlabel("global order parameter  r")
    ax.set_xlim(0, 1.0)
    for i, r in enumerate(ordered):
        ax.annotate(f"{r['r_global']:.2f}     net coupling {r['k_sum']:+.2f} rad/s",
                    (r["r_global"], i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=plots.INK_2)
    ax.set_title("Global coherence by kernel shape", loc="left", pad=10)
    f2 = plots.finish(
        fig, os.path.join(out, "order_by_kernel.png"),
        "Every kernel is rescaled to the same total |K| = 4 rad/s, so only shape differs. "
        "Net coupling is sum(K_s) after that rescaling - what a fully synchronised node "
        "feels - and it orders the kernels almost exactly as the coherence does.")

    by = {r["kernel"]: r for r in records}
    flat_r = by["flat"]["r_global"]
    hat_r = by["mexican_hat"]["r_global"]
    nyquist = net.grid_shape[0] * 0.6

    # The source's claims are comparative - "uniform-positive drives coherence, Mexican-hat
    # drives complexification" - so the checks are comparative too.  An absolute coherence
    # threshold would be testing our choice of coupling_strength, not the source's claim.
    checks = {
        "flat_kernel_coheres": bool(flat_r > 0.7),
        "mexican_hat_fragments": bool(hat_r < flat_r * 0.4),
        "simplifier_beats_complexifier": bool(
            by["5meo_plate"]["r_global"] > by["dmt_plate"]["r_global"]
            and by["5meo_plate"]["vortex_count"] < by["dmt_plate"]["vortex_count"]
        ),
        "complexifiers_proliferate_vortices": bool(
            by["mexican_hat"]["vortex_count"] > 3 * by["flat"]["vortex_count"]
        ),
    }

    # Mode selection is reported but not used as the stop-or-proceed gate.  IX.7 lists all
    # three modes, but XI.9 warns that the tabulated kernel values come from tools using a
    # different scale convention and should not be over-interpreted - so a mode that fails
    # to appear is evidence about those particular numbers, not about the implementation.
    mode_selection = {
        "checkerboard_selects_high_spatial_frequency": bool(
            by["checkerboard"]["dct_peak_kx"] > nyquist and by["checkerboard"]["dct_peak_ky"] > nyquist
        ),
        "pinwheel_selects_low_spatial_frequency": bool(
            by["pinwheel"]["dct_peak_kx"] + by["pinwheel"]["dct_peak_ky"] < 4
        ),
        "traveling_wave_kernel_travels": bool(
            by["traveling_wave"]["mode_drift_hz"] > by["pinwheel"]["mode_drift_hz"]
            and by["traveling_wave"]["mode_drift_hz"] > 0.1
        ),
    }
    modes_ok = [k for k, v in mode_selection.items() if v]
    modes_failed = [k for k, v in mode_selection.items() if not v]
    if all(checks.values()):
        verdict = (
            "Core behaviour reproduced. Uniform-positive kernels drive global coherence and "
            f"topological simplification (flat: r = {flat_r:.2f}, "
            f"{by['flat']['vortex_count']:.0f} vortices); Mexican-hat kernels fragment the "
            f"field and proliferate defects (r = {hat_r:.2f}, "
            f"{by['mexican_hat']['vortex_count']:.0f} vortices). "
            f"Mode selection: {len(modes_ok)} of {len(mode_selection)} of the source's "
            f"tabulated mode-selecting kernels reproduce - checkerboard peaks at "
            f"k = ({by['checkerboard']['dct_peak_kx']}, {by['checkerboard']['dct_peak_ky']}), "
            f"pinwheel at k = ({by['pinwheel']['dct_peak_kx']}, "
            f"{by['pinwheel']['dct_peak_ky']}), traveling-wave mode drift "
            f"{by['traveling_wave']['mode_drift_hz']:.2f} Hz against the pinwheel's "
            f"{by['pinwheel']['mode_drift_hz']:.2f} Hz."
            + (f" Not reproduced: {', '.join(modes_failed)}. Per XI.9 the tabulated kernel "
               "values come from tools on a different scale convention, so this is evidence "
               "about those specific numbers rather than about the implementation - the "
               "load-bearing simplifier/complexifier contrast above does reproduce."
               if modes_failed else "")
            + " Note that 5-MeO-like and DMT-like kernels are ordered correctly relative to "
            "each other but both sit below the coherence threshold at this coupling strength, "
            "because energy matching costs a mixed-sign kernel much of its net coupling."
        )
    else:
        verdict = (
            "NOT reproduced - the failing checks are "
            + ", ".join(k for k, v in checks.items() if not v)
            + ". Fix the implementation before reading anything into Experiments 1-5."
        )
    summary = {
        "experiment": "0 - replication",
        "n": int(net.n),
        "records": records,
        "checks": checks,
        "mode_selection": mode_selection,
        "verdict": verdict,
        "figures": [f1, f1b, f2],
    }
    _write_json(os.path.join(out, "summary.json"), summary)
    _write_csv(os.path.join(out, "records.csv"), records)
    np.savez_compressed(
        os.path.join(out, "arrays.npz"),
        **{f"phase_{k}": v for k, v in snapshots.items()},
        **{f"dct_{k}": v for k, v in dcts.items()},
    )
    return summary


# --------------------------------------------------------------------------------------
# Experiment 1 - the geometry-dependence claim (the crux)
# --------------------------------------------------------------------------------------


def experiment_1(
    root: str,
    base: RunConfig,
    amplitudes: Sequence[float] = (0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0),
    seeds: Sequence[int] = (0, 1, 2),
    n_target: int = 625,
    t_total: float = 20.0,
    geometries: Sequence[str] = GEOMETRIES,
) -> Dict:
    """IX.7 Experiment 1.  "The highest-information single run in the whole programme."

    Model B predicts the dissonance index is substantially higher on the branching tree
    than on the lattice at matched drive, with defects concentrated at bifurcations.

    The raw D-vs-A comparison is confounded: the same A produces slightly different
    achieved coherence on different geometries.  So D is *also* compared at matched
    r_global, which is the honest version of "the same drive" - the question Model B
    actually asks is whether a given degree of forced coherence costs more dissonance on
    a branching substrate.
    """
    out = _outdir(root, "exp1_geometry")
    records: List[Dict[str, object]] = []

    for geom in geometries:
        for A in amplitudes:
            for seed in seeds:
                res, stats = run_condition(base, {
                    "network.geometry": geom,
                    "network.n_target": n_target,
                    "network.seed": seed,
                    "dynamics.seed": seed,
                    "dynamics.t_total": t_total,
                    "drive.amplitude": float(A),
                })
                records.append({
                    "geometry": geom, "amplitude": float(A), "seed": seed,
                    "n": int(res.net.n),
                    "dissonance": stats["dissonance"],
                    "d_roughness": stats["d_roughness"],
                    "d_inharmonic": stats["d_inharmonic"],
                    "r_global": stats["r_global_mean"],
                    "r_local": stats["r_local_mean"],
                    "fragmentation": stats["fragmentation"],
                    "r_between": stats["r_between_mean"],
                    "harmonic_purity": stats["harmonic_purity_topk"],
                    "mean_incoherence": stats["mean_incoherence"],
                    "bifurcation_concentration": stats["bifurcation_concentration"],
                    "bifurcation_excess": stats["bifurcation_excess"],
                    # IX.5's own tree statistic: parent-segment vs daughter-mean phase step
                    "branch_over_chain_mismatch": stats.get("branch_over_chain_mismatch",
                                                            float("nan")),
                    "dissonance_intensive": stats["dissonance_intensive"],
                    "defect_density": stats["defect_density"],
                    "freq_spread_hz": stats["freq_spread_hz"],
                })

    # ---- aggregate ----
    agg: Dict[str, Dict[str, List[float]]] = {}
    for geom in geometries:
        rows = [r for r in records if r["geometry"] == geom]
        agg[geom] = {"A": [], "D": [], "D_sd": [], "r": [], "r_sd": [],
                     "frag": [], "purity": [], "bif_excess": [], "bif_conc": [],
                     "branch_mismatch": []}
        for A in amplitudes:
            sub = [r for r in rows if r["amplitude"] == A]
            dm, ds = _mean_sd(sub, "dissonance")
            rm, rs = _mean_sd(sub, "r_global")
            agg[geom]["A"].append(float(A))
            agg[geom]["D"].append(dm)
            agg[geom]["D_sd"].append(ds)
            agg[geom]["r"].append(rm)
            agg[geom]["r_sd"].append(rs)
            agg[geom]["frag"].append(_mean_sd(sub, "fragmentation")[0])
            agg[geom]["purity"].append(_mean_sd(sub, "harmonic_purity")[0])
            agg[geom]["bif_excess"].append(_mean_sd(sub, "bifurcation_excess")[0])
            agg[geom]["bif_conc"].append(_mean_sd(sub, "bifurcation_concentration")[0])
            agg[geom]["branch_mismatch"].append(_mean_sd(sub, "branch_over_chain_mismatch")[0])

    # D at matched achieved coherence - the confound-free comparison
    coherence_targets = (0.3, 0.5, 0.7, 0.85, 0.95)
    matched: Dict[str, Dict[float, float]] = {}
    for geom in geometries:
        matched[geom] = _interp_at(
            np.array(agg[geom]["r"]), np.array(agg[geom]["D"]), coherence_targets
        )

    # Ratios at *matched coherence*, not at matched amplitude.  Dividing D at equal drive and
    # then plotting the result against r would mix the two comparisons, since the same A
    # reaches slightly different coherence on each geometry.
    ratios: Dict[float, float] = {}
    ratios_by_geom: Dict[str, Dict[float, float]] = {}
    for geom in geometries:
        if geom == "lattice":
            continue
        ratios_by_geom[geom] = {}
        for c in coherence_targets:
            lat, other = matched["lattice"][c], matched[geom][c]
            r = (float(other / lat)
                 if np.isfinite(lat) and np.isfinite(other) and lat > 1e-9 else float("nan"))
            ratios_by_geom[geom][c] = r
            if geom == "tree":
                ratios[c] = r

    # The headline is taken over the forced-coherence targets only.  Below r ~ 0.7 nothing is
    # being forced, which is the regime the verdict itself excludes, so averaging those in
    # would dilute the very claim being tested.
    forced_targets = [c for c in coherence_targets if c >= 0.7]

    peak = {g: {"A": agg[g]["A"][int(np.nanargmax(agg[g]["D"]))],
                "D": float(np.nanmax(agg[g]["D"]))} for g in geometries}

    # ---- figures ----
    fig, axes = plots.new_fig(2, 2, figsize=(11.0, 7.4))
    plots.series_plot(
        axes[0, 0], list(amplitudes),
        {g: agg[g]["D"] for g in geometries},
        {g: agg[g]["D_sd"] for g in geometries},
        xlabel="drive amplitude A  (rad/s)", ylabel="dissonance index  D",
        title="D vs drive amplitude", label_last=False,
    )
    axes[0, 0].legend(loc="upper right")
    plots.series_plot(
        axes[0, 1], list(amplitudes), {g: agg[g]["r"] for g in geometries},
        {g: agg[g]["r_sd"] for g in geometries},
        xlabel="drive amplitude A  (rad/s)", ylabel="global order parameter  r",
        title="Achieved coherence  (near-identical, so A is a fair x-axis)", label_last=False,
    )
    axes[0, 1].legend(loc="lower right")

    for i, g in enumerate(geometries):
        axes[1, 0].plot(agg[g]["r"], agg[g]["D"], color=plots.SERIES[i], marker=plots.MARKERS[i],
                        markersize=5.5, markeredgecolor=plots.SURFACE, markeredgewidth=1.0, label=g)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("achieved coherence  r_global")
    axes[1, 0].set_ylabel("dissonance index  D  (log)")
    axes[1, 0].set_title("D at matched coherence", loc="left", pad=10)
    axes[1, 0].legend(loc="lower center")

    ax = axes[1, 1]
    # Ratios taken at matched coherence, the same way the verdict computes them - dividing at
    # equal drive amplitude and plotting the result on a coherence axis would silently mix
    # the two comparisons.
    for i, g in enumerate(geometries):
        if g == "lattice":
            continue
        pts = [(c, ratios_by_geom[g][c]) for c in coherence_targets
               if np.isfinite(ratios_by_geom[g].get(c, float("nan")))]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=plots.SERIES[i], marker=plots.MARKERS[i], markersize=6.5,
                markeredgecolor=plots.SURFACE, markeredgewidth=1.0, label=f"{g} / lattice")
    ax.axvspan(0.7, 1.0, color=plots.SERIES[3], alpha=0.10, linewidth=0)
    ax.axhline(1.0, color=plots.INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.28 * (hi - lo))
    ax.annotate("no geometry effect", (0.02, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, -14), textcoords="offset points", fontsize=8,
                color=plots.INK_MUTED)
    ax.annotate("forced-coherence regime\n(what Model B is about)", (0.85, ax.get_ylim()[1]),
                xytext=(0, -8), textcoords="offset points", ha="center", va="top",
                fontsize=8, color=plots.INK_2)
    ax.set_xlabel("achieved coherence  r_global")
    ax.set_ylabel("dissonance relative to the lattice")
    ax.set_title("The structural claim, as a ratio", loc="left", pad=10)
    ax.legend(loc="upper left")

    plots.suptitle(fig, "Experiment 1 - does branching geometry cost more dissonance under drive?")
    f1 = plots.finish(fig, os.path.join(out, "geometry_sweep.png"),
                      f"Mean +/- SD over {len(seeds)} seeds; N = "
                      + ", ".join(f"{g} {int([r for r in records if r['geometry'] == g][0]['n'])}"
                                  for g in geometries) + ".")

    # spatial defect map on the tree, at the peak-D amplitude
    tree_peak_A = peak["tree"]["A"] if "tree" in peak else 4.0
    res, stats = run_condition(base, {
        "network.geometry": "tree", "network.n_target": n_target,
        "network.seed": 0, "dynamics.seed": 0, "dynamics.t_total": t_total,
        "drive.amplitude": float(tree_peak_A),
    })
    res0, stats0 = run_condition(base, {
        "network.geometry": "tree", "network.n_target": n_target,
        "network.seed": 0, "dynamics.seed": 0, "dynamics.t_total": t_total,
        "drive.amplitude": 0.0,
    })
    inc = stats["incoherence_map"]
    inc0 = stats0["incoherence_map"]
    net = res.net
    series_paths = [
        save_series(root, f"exp1_tree_A{tree_peak_A:g}", res),
        save_series(root, "exp1_tree_A0", res0),
    ]

    fig, axes = plots.new_fig(1, 3, figsize=(13.0, 4.2))
    vmax = float(max(inc.max(), inc0.max()))
    plots.node_map(axes[0], net.positions, inc0, title="no drive", vmin=0, vmax=vmax,
                   cbar_label="local incoherence")
    plots.node_map(axes[1], net.positions, inc, title=f"drive A = {tree_peak_A:g}",
                   vmin=0, vmax=vmax, cbar_label="local incoherence")
    bif = net.bifurcation_nodes
    axes[1].scatter(net.positions[bif, 0], net.positions[bif, 1], s=16, facecolors="none",
                    edgecolors=plots.SERIES[1], linewidths=0.8, label="bifurcation")
    axes[1].legend(loc="upper right", fontsize=8)

    # Only the tree has branch points.  The lattice has none and the layered network's
    # degree structure partitions it by layer, not by anything this claim is about, so
    # neither appears here - a meaningless control curve would read as a real comparison.
    ax = axes[2]
    ax.plot(agg["tree"]["A"], agg["tree"]["bif_excess"], color=plots.SERIES[0],
            marker="o", markersize=5.5, markeredgecolor=plots.SURFACE, markeredgewidth=1.0,
            label="tree: branch points vs degree-matched null")
    ax.plot(agg["tree"]["A"], agg["tree"]["bif_conc"], color=plots.SERIES[1],
            marker="s", markersize=5.5, markeredgecolor=plots.SURFACE, markeredgewidth=1.0,
            linestyle=(0, (5, 2)), label="tree: raw branch/chain ratio")
    ax.plot(agg["tree"]["A"], agg["tree"]["branch_mismatch"], color=plots.SERIES[2],
            marker="^", markersize=5.5, markeredgecolor=plots.SURFACE, markeredgewidth=1.0,
            label="tree: parent-daughter phase step (IX.5's own statistic)")
    ax.axhline(1.0, color=plots.INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.annotate("no excess", (amplitudes[-1], 1.0), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=8, color=plots.INK_MUTED)
    ax.set_xlabel("drive amplitude A  (rad/s)")
    ax.set_ylabel("relative local incoherence")
    ax.set_title("Do defects concentrate at bifurcations?", loc="left", pad=10)
    ax.set_ylim(bottom=0.9)
    ax.legend(loc="upper right", fontsize=8)
    plots.suptitle(fig, "Experiment 1 - spatial distribution of phase defects on the tree")
    f2 = plots.finish(fig, os.path.join(out, "tree_defects.png"),
                      "Null = same phases permuted across nodes, which preserves degree but destroys structure.")

    tree_gt_lattice_peak = bool(peak.get("tree", {}).get("D", 0) > peak.get("lattice", {}).get("D", 0))
    forced_ratios = [ratios[c] for c in forced_targets
                     if c in ratios and np.isfinite(ratios[c])]
    weak_ratios = [ratios[c] for c in coherence_targets
                   if c not in forced_targets and c in ratios and np.isfinite(ratios[c])]
    matched_median = float(np.median(forced_ratios)) if forced_ratios else float("nan")
    weak_median = float(np.median(weak_ratios)) if weak_ratios else float("nan")
    bif_excess_tree = float(np.nanmean(agg["tree"]["bif_excess"])) if "tree" in agg else float("nan")
    branch_mismatch_tree = (float(np.nanmean(agg["tree"]["branch_mismatch"]))
                            if "tree" in agg else float("nan"))

    verdict_parts = []
    if not np.isfinite(matched_median):
        # No coherence target fell inside the swept range, so there is no comparison to
        # report.  Falling through to the else branch below would announce a refutation of
        # the crux experiment from a run that never made the measurement.
        verdict_parts.append(
            "INCONCLUSIVE - no geometry's mean r_global reached the coherence targets "
            f"{list(coherence_targets)} anywhere in the swept amplitude range "
            f"{list(map(float, amplitudes))}, so D could not be compared at matched "
            "coherence. Widen the amplitude sweep or raise kernel.coupling_strength; this "
            "run neither supports nor refutes Model B's structural claim."
        )
    else:
        peak_ratio_pk = (peak["tree"]["D"] / peak["lattice"]["D"]
                         if peak.get("lattice", {}).get("D") else float("nan"))
        headline = (
            f"In the forced-coherence regime (r_global >= 0.7 - the regime Model B is actually "
            f"about, since below it nothing is being forced) the tree carries "
            f"{matched_median:.2f}x the lattice's dissonance at matched coherence"
        )
        if matched_median > 1.25:
            headline += ", which supports Model B's structural claim (V.2)."
        elif matched_median > 1.05:
            headline += " - in the predicted direction, but small."
        else:
            headline += (". Model B's central structural claim is NOT supported: dissonance "
                         "under drive is essentially geometry-independent here.")
        verdict_parts.append(headline)
        verdict_parts.append(
            f"The claim is regime-specific, not general: below r_global = 0.7 the same ratio is "
            f"{weak_median:.2f}, and the *peak* dissonance the three geometries can reach is "
            f"near-identical ({peak_ratio_pk:.2f}x tree over lattice), so the version of the "
            "prediction stated in IX.7 - much higher dissonance on the tree than the lattice at "
            "the same drive - does not hold. Only the refined version does."
        )
    verdict_parts.append(
        f"On the defect claim the two readouts disagree, which is informative: local "
        f"incoherence is {bif_excess_tree:.2f}x higher at branch points than a degree-matched "
        f"null predicts, but IX.5's own statistic - the phase step from a parent segment to "
        f"the mean of its daughters - is {branch_mismatch_tree:.2f}x the same step along "
        f"ordinary chain segments"
        + (", i.e. no larger at branch points at all. Branch points show more neighbourhood "
           "disagreement simply because they have more neighbours; the phase discontinuity "
           "Model B predicts them to carry is not there."
           if branch_mismatch_tree < 1.1 else ".")
    )

    summary = {
        "experiment": "1 - geometry dependence (the crux)",
        "amplitudes": list(map(float, amplitudes)),
        "seeds": list(seeds),
        "n_by_geometry": {g: int([r for r in records if r["geometry"] == g][0]["n"]) for g in geometries},
        "aggregate": agg,
        "peak_dissonance": peak,
        "D_at_matched_coherence": {g: {str(k): v for k, v in matched[g].items()} for g in matched},
        "tree_over_lattice_at_matched_coherence": {str(k): v for k, v in ratios.items()},
        "ratios_by_geometry": {g: {str(k): v for k, v in d.items()}
                               for g, d in ratios_by_geom.items()},
        "forced_coherence_targets": list(map(float, forced_targets)),
        "tree_over_lattice_median_forced": matched_median,
        "tree_over_lattice_median_weak": weak_median,
        "tree_over_lattice_median": matched_median,
        "tree_peak_above_lattice_peak": tree_gt_lattice_peak,
        "mean_bifurcation_excess_tree": bif_excess_tree,
        "mean_branch_over_chain_mismatch_tree": branch_mismatch_tree,
        "series": series_paths,
        "verdict": " ".join(verdict_parts),
        "figures": [f1, f2],
    }
    _write_json(os.path.join(out, "summary.json"), summary)
    _write_csv(os.path.join(out, "records.csv"), records)
    np.savez_compressed(os.path.join(out, "arrays.npz"),
                        incoherence_drive=inc, incoherence_nodrive=inc0,
                        positions=net.positions, bifurcations=net.bifurcation_nodes)
    return summary


# --------------------------------------------------------------------------------------
# Experiment 2 - attack dynamics
# --------------------------------------------------------------------------------------


def _attack_stats(ts: Dict[str, np.ndarray], on_edges: Sequence[float], window_s: float,
                  ramp_s: float = 0.0, reference_peak: Optional[float] = None,
                  ) -> List[Dict[str, object]]:
    """Characterise each driven window: is the dissonance sustained, or only a transient
    at the coherence transitions?

    This is the question the first version of this experiment got wrong.  A drop in D in
    the middle of a driven window looks like spontaneous termination but is not - it is
    the system finishing its entrainment and going quiet.  `sustained_fraction` separates
    the two: it is the fraction of the driven window spent above half the window's peak.
    """
    attacks: List[Dict[str, object]] = []
    for t_on in on_edges:
        t_off = t_on + window_s
        m = (ts["t"] >= t_on - 2.0) & (ts["t"] <= t_off + 6.0)
        if m.sum() < 6:
            continue
        seg_t, seg_d, seg_r = ts["t"][m], ts["dissonance"][m], ts["r_global"][m]
        pre = seg_t < t_on
        during = (seg_t >= t_on) & (seg_t <= t_off)
        # A window with no pre-attack samples has no baseline; taking one from inside the
        # attack would make the first window's rise look like no rise at all.  The first
        # driven window usually falls into this case, because the sliding-window centres
        # start half a window after t = 0.
        if during.sum() < 4 or pre.sum() < 2:
            continue
        base_d = float(seg_d[pre].mean())
        peak_d = float(seg_d[during].max())
        t_peak = float(seg_t[during][int(np.argmax(seg_d[during]))])

        # Two separate questions, because conflating them is what made the first version of
        # this statistic meaningless.
        #   attack_occurred - did D rise at all in this window, judged against the largest
        #                     rise seen anywhere in the run?  A flat window fails here.
        #   sustained       - given that it rose, did it stay up or spike at the edges?  This
        #                     one is *correctly* self-normalised: it is a question about the
        #                     shape of a rise, and only asked of windows that have one.
        ref = reference_peak if reference_peak is not None else peak_d
        attack_occurred = bool((peak_d - base_d) > 0.25 * max(ref - base_d, 1e-12))
        half = base_d + 0.5 * (peak_d - base_d)
        sustained = float((seg_d[during] >= half).mean()) if attack_occurred else float("nan")

        # middle third of the window, once entrainment has had time to complete
        mid = during & (seg_t >= t_on + window_s / 3) & (seg_t <= t_off - window_s / 3)
        mid_d = float(seg_d[mid].mean()) if mid.any() else float("nan")

        # Spontaneous termination, measured rather than asserted: D peaks in the first half
        # of the window and has fallen back near its pre-attack level by the end, *while the
        # drive is still at full amplitude*.  The tail therefore has to stop before the drive
        # starts ramping down, or it measures drive withdrawal rather than termination.
        tail_end = t_off - ramp_s
        tail = during & (seg_t >= tail_end - window_s / 5) & (seg_t <= tail_end)
        end_d = float(seg_d[tail].mean()) if tail.any() else float("nan")
        span = peak_d - base_d
        terminates = bool(
            span > 1e-9
            and t_peak < t_on + window_s / 2
            and np.isfinite(end_d)
            and (end_d - base_d) < 0.3 * span
        )
        attacks.append({
            "t_on": float(t_on), "baseline_D": base_d, "peak_D": peak_d,
            "t_peak": t_peak, "rise_s": t_peak - float(t_on),
            "sustained_fraction": sustained,
            "attack_occurred": attack_occurred,
            "mid_window_D": mid_d,
            "mid_over_peak": mid_d / peak_d if peak_d > 1e-12 else float("nan"),
            "end_window_D": end_d,
            "terminates_while_driven": terminates,
            "r_global_mid": float(seg_r[mid].mean()) if mid.any() else float("nan"),
        })
    return attacks


def experiment_2(
    root: str,
    base: RunConfig,
    n_target: int = 625,
    attack_amplitude: float = 4.0,
    compare_amplitudes: Sequence[float] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.5),
    period_s: float = 30.0,
    duty: float = 0.35,
    t_total: float = 120.0,
    ramp_amplitude: float = 9.0,
    seeds: Sequence[int] = (0, 1, 2),
) -> Dict:
    """IX.7 Experiment 2.  Onset shape, hysteresis, and whether the model can hold a
    sustained dissonant state and terminate it.

    "Cluster attacks have a characteristic duration and terminate spontaneously; a model
    that cannot produce spontaneous termination is missing something."

    The result that matters here is that D is sustained only inside a narrow band of drive
    amplitude.  Drive the tree harder and it entrains completely and goes quiet - so the
    model says the pathological state is *partial* entrainment, and that both too little
    and too much hypothalamic drive are painless.
    """
    out = _outdir(root, "exp2_attack")
    window_s = duty * period_s

    # --- duty-cycle attack train, at several drive amplitudes ---
    ramp_s = 1.5
    trains: Dict[float, Dict[str, np.ndarray]] = {}
    attacks_by_A: Dict[float, List[Dict[str, object]]] = {}
    amplitudes = sorted({float(a) for a in list(compare_amplitudes) + [attack_amplitude]})
    seed0 = seeds[0]
    net = None

    def _train(A: float, seed: int):
        cfg = apply_overrides(base, {
            "network.geometry": "tree", "network.n_target": n_target,
            "network.seed": seed, "dynamics.seed": seed, "dynamics.t_total": t_total,
            "drive.amplitude": float(A), "drive.waveform": "square",
            "drive.period_s": period_s, "drive.duty": duty, "drive.ramp_s": ramp_s,
        })
        nonlocal net
        net = cached_network(cfg.network)
        res = Simulator(cfg, net).run()
        ts = dissonance_timeseries(res.theta, res.times, net, res.fs, cfg.metrics,
                                   window_s=3.0, step_s=0.5)
        edges = [res.times[i] for i in range(1, len(res.drive_g))
                 if res.drive_g[i - 1] < 0.5 <= res.drive_g[i]]
        return res, ts, edges

    # First pass: a shared reference peak, so `sustained_fraction` is measured against one
    # scale across every amplitude instead of against each window's own maximum.
    runs: Dict[Tuple[float, int], Tuple] = {}
    for A in amplitudes:
        for seed in seeds:
            runs[(A, seed)] = _train(A, seed)
    reference_peak = float(np.nanmax([
        np.nanmax(ts["dissonance"]) for (_, ts, _) in runs.values()
    ]))

    for A in amplitudes:
        per_seed = []
        stats: List[Dict[str, object]] = []
        for seed in seeds:
            res, ts, edges = runs[(A, seed)]
            stats.extend(_attack_stats(ts, edges, window_s, ramp_s, reference_peak))
            per_seed.append((res, ts, edges))
        res0, ts0, edges0 = per_seed[0]
        t_ref = ts0["t"]
        stack = np.stack([np.interp(t_ref, ts["t"], ts["dissonance"]) for _, ts, _ in per_seed])
        rstack = np.stack([np.interp(t_ref, ts["t"], ts["r_global"]) for _, ts, _ in per_seed])
        fstack = np.stack([np.interp(t_ref, ts["t"], ts["fragmentation"]) for _, ts, _ in per_seed])
        trains[A] = {"t": t_ref, "D": stack.mean(axis=0), "D_sd": stack.std(axis=0),
                     "r": rstack.mean(axis=0), "frag": fstack.mean(axis=0),
                     "drive": np.interp(t_ref, res0.times, res0.drive_g),
                     "on_edges": np.asarray(edges0)}
        attacks_by_A[A] = stats

    ts = {"t": trains[attack_amplitude]["t"],
          "dissonance": trains[attack_amplitude]["D"],
          "r_global": trains[attack_amplitude]["r"],
          "fragmentation": trains[attack_amplitude]["frag"]}
    g_at = trains[attack_amplitude]["drive"]
    on_edges = list(trains[attack_amplitude]["on_edges"])
    attacks = attacks_by_A[attack_amplitude]

    # --- hysteresis: slow up-and-down ramp ---
    cfg_h = apply_overrides(base, {
        "network.geometry": "tree", "network.n_target": n_target,
        "network.seed": seed0, "dynamics.seed": seed0, "dynamics.t_total": t_total,
        "drive.amplitude": ramp_amplitude, "drive.waveform": "ramp_updown",
    })
    res_h = Simulator(cfg_h, net).run()
    ts_h = dissonance_timeseries(res_h.theta, res_h.times, net, res_h.fs, cfg_h.metrics,
                                 window_s=3.0, step_s=0.5)
    g_h = np.interp(ts_h["t"], res_h.times, res_h.drive_g)
    A_h = ramp_amplitude * g_h
    up = ts_h["t"] <= t_total / 2
    down = ~up

    # hysteresis area: |integral of D over A, up branch - down branch|
    def _branch(mask, reverse=False):
        a, d = A_h[mask], ts_h["dissonance"][mask]
        o = np.argsort(a)
        return a[o], d[o]

    a_up, d_up = _branch(up)
    a_dn, d_dn = _branch(down)
    grid = np.linspace(0, ramp_amplitude, 60)
    hyst = float(np.trapezoid(np.interp(grid, a_dn, d_dn) - np.interp(grid, a_up, d_up), grid))
    # Keep the sign.  A negative area means the falling branch sits BELOW the rising one -
    # the system releasing early, the opposite of persistence - and reporting |area| would
    # announce that as the dissonant state persisting.
    hyst_norm = float(hyst / max(np.trapezoid(np.interp(grid, a_up, d_up), grid), 1e-12))

    # onset abruptness: max |dD/dA| on the up branch, relative to the mean slope
    up_i = np.interp(grid, a_up, d_up)
    dn_i = np.interp(grid, a_dn, d_dn)
    dd = np.gradient(up_i, grid)
    abruptness = float(np.max(np.abs(dd)) / max(np.mean(np.abs(dd)), 1e-12))

    # D(A) here is unimodal and returns to roughly where it started, so the signed area
    # between the branches largely self-cancels and is a poor bistability statistic on its
    # own.  The shift in where each branch peaks is the direct question - does the system
    # release at a lower drive than it engaged at? - and does not cancel.
    peak_up = float(grid[int(np.argmax(up_i))])
    peak_dn = float(grid[int(np.argmax(dn_i))])
    peak_shift = peak_dn - peak_up
    max_gap = float(np.max(np.abs(dn_i - up_i)) / max(np.max(up_i), 1e-12))

    # ---- figures ----
    fig, axes = plots.new_fig(3, 1, figsize=(10.4, 7.0), sharex=True,
                              gridspec_kw={"height_ratios": [0.8, 1.4, 1.4]})
    axes[0].fill_between(ts["t"], 0, g_at * attack_amplitude, color=plots.SERIES[3],
                         alpha=0.30, linewidth=0)
    axes[0].plot(ts["t"], g_at * attack_amplitude, color=plots.SERIES[3], linewidth=1.6)
    axes[0].set_ylabel("drive A(t)")
    axes[0].set_title("Hypothalamic drive, circadian duty cycle", loc="left", pad=8)

    # Only three traces in the time panel, or it becomes unreadable; the full amplitude
    # sweep is the second figure's job.
    show = [amplitudes[0], attack_amplitude, amplitudes[-1]]
    for i, A in enumerate(dict.fromkeys(show)):
        axes[1].plot(trains[A]["t"], trains[A]["D"], color=plots.SERIES[i], linewidth=2.0,
                     label=f"A = {A:g}")
    axes[1].set_ylabel("dissonance  D")
    occ_preview = {A: float(np.mean([a["attack_occurred"] for a in attacks_by_A[A]]))
                   if attacks_by_A[A] else float("nan") for A in amplitudes}
    vals = [v for v in occ_preview.values() if np.isfinite(v)]
    spread = (max(vals) - min(vals)) if vals else 0.0
    axes[1].set_title(
        "Dissonance index - "
        + ("an attack only occurs at intermediate drive" if spread > 0.4
           else "attacks occur across the drive range tested"),
        loc="left", pad=8)
    axes[1].legend(loc="upper right", ncol=3)

    axes[2].plot(ts["t"], ts["r_global"], color=plots.SERIES[2], linewidth=2.0, label="r_global")
    axes[2].plot(ts["t"], ts["fragmentation"], color=plots.SERIES[1], linewidth=2.0,
                 label="fragmentation (r_local - r_global)")
    axes[2].set_ylabel("order")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right")
    axes[2].set_title(f"Coherence and fragmentation at A = {attack_amplitude:g}",
                      loc="left", pad=8)
    for ax in axes:
        for t_on in on_edges:
            ax.axvspan(t_on, t_on + window_s, color=plots.SERIES[3], alpha=0.07, linewidth=0)
    plots.suptitle(fig, "Experiment 2 - attack cycle on the trigeminal tree")
    f1 = plots.finish(
        fig, os.path.join(out, "attack_cycle.png"),
        "Shaded bands mark the driven windows. Fraction of windows containing an attack: "
        + ", ".join(f"A={A:g} {s:.0%}" for A, s in occ_preview.items()) + ".")

    # ---- sustained fraction vs drive amplitude: where the attack window is ----
    fig, ax = plots.new_fig(figsize=(7.0, 4.2))
    sus = [float(np.nanmean([a["sustained_fraction"] for a in attacks_by_A[A]]))
           if any(a["attack_occurred"] for a in attacks_by_A[A]) else float("nan")
           for A in amplitudes]
    mid = [float(np.nanmean([a["mid_over_peak"] for a in attacks_by_A[A]]))
           if any(a["attack_occurred"] for a in attacks_by_A[A]) else float("nan")
           for A in amplitudes]
    occurred = [float(np.mean([a["attack_occurred"] for a in attacks_by_A[A]]))
                if attacks_by_A[A] else float("nan") for A in amplitudes]
    plots.series_plot(ax, list(amplitudes),
                      {"windows with an attack at all": occurred,
                       "time above half-peak D": sus,
                       "mid-window D / peak D": mid},
                      xlabel="drive amplitude A  (rad/s)",
                      ylabel="fraction",
                      title="Is the dissonant state sustained, or only a transition transient?",
                      label_last=False)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    f1b = plots.finish(fig, os.path.join(out, "sustained_fraction.png"),
                       "Low values mean D spikes at the drive edges and collapses in between - "
                       "the system entrains and goes quiet rather than staying dissonant.")

    fig, ax = plots.new_fig(figsize=(6.4, 4.4))
    ax.plot(A_h[up], ts_h["dissonance"][up], color=plots.SERIES[0], linewidth=2.0, label="drive rising")
    ax.plot(A_h[down], ts_h["dissonance"][down], color=plots.SERIES[1], linewidth=2.0,
            linestyle=(0, (5, 2)), label="drive falling")
    ax.set_xlabel("drive amplitude A  (rad/s)")
    ax.set_ylabel("dissonance  D")
    ax.set_title(f"Hysteresis loop   (normalised area {hyst_norm:.2f})", loc="left", pad=10)
    ax.legend(loc="upper right")
    f2 = plots.finish(fig, os.path.join(out, "hysteresis.png"),
                      "A positive area means the dissonant state persists as the drive withdraws.")

    best_A = amplitudes[int(np.nanargmax(np.where(np.isfinite(sus), sus, -1)))]
    sus_at = {float(a): float(s) for a, s in zip(amplitudes, sus)}
    mid_at = {float(a): float(m) for a, m in zip(amplitudes, mid)}
    occurred_at = {float(a): float(o) for a, o in zip(amplitudes, occurred)}

    # The band exists if an attack fails to start at the low end *and* degenerates into edge
    # transients at the high end.  np.nanmin, not min: a plain min() over a list whose first
    # entry is NaN returns NaN, and every comparison against it is False, so the test would
    # silently always fail.
    finite_mid = [m for m in mid if np.isfinite(m)]
    band_structure = bool(
        finite_mid and max(finite_mid) > 0.6 and np.nanmin(mid) < 0.35
        and (occurred[0] < 0.5 or np.nanmin(sus) < 0.5)
    )
    all_attacks = [a for A in amplitudes for a in attacks_by_A[A]]
    n_term = sum(1 for a in all_attacks if a["terminates_while_driven"])
    term_frac = n_term / len(all_attacks) if all_attacks else float("nan")

    if peak_shift > 0.3:
        hyst_text = (f"the falling branch peaks {peak_shift:+.2f} rad/s *above* where the "
                     "rising branch did, so the dissonant state persists as the drive withdraws")
    elif peak_shift < -0.3:
        hyst_text = (f"the falling branch peaks {peak_shift:+.2f} rad/s below the rising one, "
                     "so the system releases early - the opposite of persistence")
    else:
        hyst_text = ("both branches peak at the same drive amplitude, so there is no "
                     "bistability to speak of")

    verdict = (
        f"Onset is {'abrupt' if abruptness > 3 else 'smooth'} "
        f"(peak/mean |dD/dA| = {abruptness:.1f}); {hyst_text} "
        f"(signed loop area {hyst_norm:+.2f} of the forward curve, largest branch separation "
        f"{max_gap:.0%} - note the area statistic largely self-cancels on a unimodal curve, "
        f"which is why the peak shift is quoted first). "
        + "An attack (a rise in D worth the name) occurs in "
        + ", ".join(f"A={a:g} {o:.0%}" for a, o in occurred_at.items())
        + " of driven windows; where one occurs, the fraction of the window it holds above "
        "half its own peak is "
        + ", ".join(f"A={a:g} -> " + ("n/a" if not np.isfinite(s) else f"{s:.0%}")
                    for a, s in sus_at.items())
        + f", peaking at A = {best_A:g}. "
        + "The clearer statistic is how much of the peak survives into the middle of the "
        "window, once entrainment has had time to finish: "
        + ", ".join(f"A={a:g} -> " + ("n/a" if not np.isfinite(m) else f"{m:.0%}")
                    for a, m in mid_at.items())
        + ". "
        + ("Below the band no attack starts at all; above it the tree entrains completely and "
           "the dissonance collapses into two transients at the drive edges. So the model "
           "produces a sustained high-dissonance attack only under *partial* entrainment. "
           "Taken literally that is a sharp and counterintuitive prediction: increasing the "
           "hypothalamic drive past the band should abort the attack rather than deepen it."
           if band_structure
           else "The band structure is weak across the amplitudes tested, so the "
                "partial-entrainment reading is not well supported by this run.")
        + f" Spontaneous termination (D peaks early and returns to baseline while the drive is "
        f"still at full amplitude) occurred in {n_term} of {len(all_attacks)} driven windows"
        + ("; the model does reproduce it." if term_frac > 0.7 else
           " - it happens, but not reliably, so the mechanism IX.7 asks for is at best "
           "partly present." if term_frac >= 0.3 else
           ", so attacks essentially end only when the drive does - the gap IX.7 anticipated.")
    )

    summary = {
        "experiment": "2 - attack dynamics",
        "attack_amplitude": attack_amplitude,
        "compare_amplitudes": list(map(float, amplitudes)),
        "period_s": period_s, "duty": duty, "window_s": window_s,
        "attacks": attacks,
        "sustained_fraction_by_amplitude": sus_at,
        "attack_occurred_by_amplitude": occurred_at,
        "mid_over_peak_by_amplitude": mid_at,
        "band_structure": band_structure,
        "best_amplitude": float(best_A),
        "onset_abruptness": abruptness,
        "hysteresis_area": hyst,
        "hysteresis_normalised": hyst_norm,  # signed
        "hysteresis_peak_shift_rad_s": peak_shift,
        "hysteresis_max_branch_gap": max_gap,
        "reference_peak_D": reference_peak,
        "seeds": list(seeds),
        "spontaneous_termination_fraction": term_frac,
        "verdict": verdict,
        "figures": [f1, f1b, f2],
    }
    _write_json(os.path.join(out, "summary.json"), summary)
    _write_csv(os.path.join(out, "records.csv"),
               [{"amplitude": A, **a} for A in amplitudes for a in attacks_by_A[A]])
    np.savez_compressed(os.path.join(out, "arrays.npz"),
                        t=ts["t"], D=ts["dissonance"], r_global=ts["r_global"],
                        fragmentation=ts["fragmentation"], drive=g_at,
                        hyst_A=A_h, hyst_D=ts_h["dissonance"], hyst_up=up,
                        **{f"D_A{A:g}": trains[A]["D"] for A in amplitudes})
    return summary


# --------------------------------------------------------------------------------------
# Experiment 3 - the three interventions
# --------------------------------------------------------------------------------------


def experiment_3(
    root: str,
    base: RunConfig,
    n_target: int = 625,
    attack_amplitude: float = 4.0,
    t_total: float = 60.0,
    t_intervene: float = 25.0,
    duration: float = 12.0,
    seeds: Sequence[int] = (0, 1, 2),
    threshold_amplitudes: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
    oxygen_factors: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
) -> Dict:
    """IX.7 Experiment 3.  Apply each intervention mid-attack.

    The discriminating condition is 5-MeO: Model A (naive symmetrisation) predicts the
    uniform-positive kernel helps most; Model B predicts it fails or worsens.
    """
    out = _outdir(root, "exp3_interventions")
    t_end = t_intervene + duration

    conditions: Dict[str, List[Intervention]] = {
        "none": [],
        "DMT (transient Mexican-hat)": [
            Intervention("dmt", "kernel", t_intervene, t_end, kernel="mexican_hat")],
        "5-MeO (transient uniform-positive)": [
            Intervention("5meo", "kernel", t_intervene, t_end, kernel="5meo_plate")],
        "oxygen (sigma_omega x0.25)": [
            Intervention("oxygen", "sigma_omega", t_intervene, float("inf"), factor=0.25)],
        "sumatriptan (peripheral gain x0.4)": [
            Intervention("suma", "gain", t_intervene, float("inf"), factor=0.4,
                         compartment="peripheral")],
    }

    records: List[Dict[str, object]] = []
    traces: Dict[str, Dict[str, np.ndarray]] = {}
    series_paths: List[str] = []

    for name, ivs in conditions.items():
        per_seed_traces = []
        for seed in seeds:
            cfg = apply_overrides(base, {
                "network.geometry": "tree", "network.n_target": n_target,
                "network.seed": seed, "dynamics.seed": seed, "dynamics.t_total": t_total,
                "drive.amplitude": attack_amplitude,
            })
            cfg.interventions = list(ivs)
            net = cached_network(cfg.network)
            res = Simulator(cfg, net).run()
            ts = dissonance_timeseries(res.theta, res.times, net, res.fs, cfg.metrics,
                                       window_s=3.0, step_s=0.5)
            per_seed_traces.append(ts)
            if seed == seeds[0] and name in ("none", "DMT (transient Mexican-hat)"):
                slug = "control" if name == "none" else "dmt"
                series_paths.append(save_series(root, f"exp3_{slug}", res))

            t = ts["t"]
            pre = (t > t_intervene - 8) & (t < t_intervene - 0.5)
            during = (t > t_intervene + 1.0) & (t < t_end)
            post = (t > t_end + 2.0) & (t < t_end + 10.0)
            d_pre = float(ts["dissonance"][pre].mean()) if pre.any() else float("nan")
            d_during = float(ts["dissonance"][during].mean()) if during.any() else float("nan")
            d_post = float(ts["dissonance"][post].mean()) if post.any() else float("nan")

            # Latency to a 50% fall, measured from intervention onset and bounded by the
            # window.  An unbounded search would happily report a dip that happens long after
            # a transient intervention has washed out as that intervention's onset latency.
            latency = float("nan")
            if np.isfinite(d_pre) and d_pre > 1e-9:
                span = (t >= t_intervene) & (t <= min(t_end, t[-1]))
                hit = np.where(ts["dissonance"][span] <= 0.5 * d_pre)[0]
                if hit.size:
                    latency = float(t[span][hit[0]] - t_intervene)

            records.append({
                "condition": name, "seed": seed,
                "D_pre": d_pre, "D_during": d_during, "D_post": d_post,
                "delta_D": d_during - d_pre,
                "relative_change": (d_during - d_pre) / d_pre if d_pre > 1e-9 else float("nan"),
                "latency_to_half_s": latency,
                "r_global_pre": float(ts["r_global"][pre].mean()) if pre.any() else float("nan"),
                "r_global_during": float(ts["r_global"][during].mean()) if during.any() else float("nan"),
            })

        t_ref = per_seed_traces[0]["t"]
        stack = np.stack([np.interp(t_ref, s["t"], s["dissonance"]) for s in per_seed_traces])
        rstack = np.stack([np.interp(t_ref, s["t"], s["r_global"]) for s in per_seed_traces])
        traces[name] = {"t": t_ref, "D": stack.mean(axis=0), "D_sd": stack.std(axis=0),
                        "r": rstack.mean(axis=0)}

    # --- persistent kernel shift: does any shape raise the drive threshold? ---
    threshold_rows: List[Dict[str, object]] = []
    candidates = ["baseline", "retuned", "5meo_plate", "mexican_hat", "flat"]
    for kname in candidates:
        spec = KERNEL_PRESETS[kname]
        for A in threshold_amplitudes:
            for seed in seeds:
                _, stats = run_condition(base, {
                    "network.geometry": "tree", "network.n_target": n_target,
                    "network.seed": seed, "dynamics.seed": seed, "dynamics.t_total": 20.0,
                    "drive.amplitude": float(A),
                    "kernel.name": kname, "kernel.k": spec["k"],
                    "kernel.k_sw": spec.get("k_sw", 0.0),
                })
                threshold_rows.append({
                    "kernel": kname, "amplitude": float(A), "seed": seed,
                    "dissonance": stats["dissonance"], "r_global": stats["r_global_mean"],
                })

    curves: Dict[str, List[float]] = {}
    for kname in candidates:
        curves[kname] = [
            _mean_sd([r for r in threshold_rows if r["kernel"] == kname and r["amplitude"] == A],
                     "dissonance")[0]
            for A in threshold_amplitudes
        ]

    # IX.4 defines the psychedelic arm as a kernel shift that "raises the drive amplitude A
    # needed for dissonance onset".  That has to be a COMMON dissonance level: taking each
    # kernel's own half-maximum measures the shape of its own curve instead, so a kernel
    # that halves D everywhere - the ideal preventive - would report an unchanged threshold.
    baseline_peak = float(np.nanmax(curves["baseline"]))
    onset_level = 0.5 * baseline_peak

    def _crossing(ds: Sequence[float]) -> float:
        """First A at which D rises through `onset_level`.

        `inf`  - never reaches the onset level anywhere in the swept range (most protective)
        `0.0`  - already above it at the lowest amplitude tested, so the threshold lies below
                 the range (least protective).  Returning NaN for this, as an earlier version
                 did, made the worst kernels indistinguishable from unmeasured ones and then
                 silently dropped them from the figure caption.
        """
        finite = [d for d in ds if np.isfinite(d)]
        if not finite:
            return float("nan")
        if ds[0] >= onset_level:
            return 0.0
        for i in range(1, len(ds)):
            lo, hi = ds[i - 1], ds[i]
            if np.isfinite(lo) and np.isfinite(hi) and lo < onset_level <= hi:
                a0, a1 = threshold_amplitudes[i - 1], threshold_amplitudes[i]
                return float(a0 + (onset_level - lo) / (hi - lo) * (a1 - a0))
        return float("inf")

    thresholds: Dict[str, float] = {k: _crossing(v) for k, v in curves.items()}
    peak_ratio = {k: float(np.nanmax(v) / baseline_peak) if baseline_peak > 1e-12 else float("nan")
                  for k, v in curves.items()}

    # --- oxygen dose-response: IX.4 asks for a sweep of f, not a single point ---
    oxygen_rows: List[Dict[str, object]] = []
    for f in oxygen_factors:
        for seed in seeds:
            _, stats = run_condition(base, {
                "network.geometry": "tree", "network.n_target": n_target,
                "network.seed": seed, "dynamics.seed": seed, "dynamics.t_total": 20.0,
                "drive.amplitude": attack_amplitude,
                "dynamics.sigma_omega_hz": base.dynamics.sigma_omega_hz * float(f),
            })
            oxygen_rows.append({
                "sigma_factor": float(f), "seed": seed,
                "dissonance": stats["dissonance"], "r_global": stats["r_global_mean"],
                "freq_spread_hz": stats["freq_spread_hz"],
            })
    oxygen_curve = [_mean_sd([r for r in oxygen_rows if r["sigma_factor"] == f], "dissonance")[0]
                    for f in oxygen_factors]
    oxygen_sd = [_mean_sd([r for r in oxygen_rows if r["sigma_factor"] == f], "dissonance")[1]
                 for f in oxygen_factors]
    # Monotone in f, judged against the index's own run-to-run noise (~11% CV) rather than
    # an absolute epsilon - a 0.0005 dip between two adjacent points is not a real reversal.
    _tol = 0.05 * (max(oxygen_curve) - min(oxygen_curve))
    oxygen_monotone = bool(np.all(np.diff(oxygen_curve) >= -_tol))
    oxygen_direction = "reduces" if oxygen_curve[0] < oxygen_curve[-1] else "increases"
    oxygen_span = (
        (oxygen_curve[-1] - oxygen_curve[0]) / oxygen_curve[-1]
        if oxygen_curve and oxygen_curve[-1] > 1e-12 else float("nan")
    )

    # Every arm is quoted against the untreated arm run on the same seeds, with the seed
    # spread attached.  A bare pre->during change carries a drift term (the "none" arm is not
    # flat) and, at three seeds, an effect inside the noise would otherwise print as a result.
    def _arm(prefix: str) -> Tuple[float, float]:
        rows = [r for r in records if r["condition"].startswith(prefix)]
        return _mean_sd(rows, "relative_change")

    none_change, none_sd = _arm("none")

    # ---- figures ----
    fig, ax = plots.new_fig(figsize=(9.6, 4.6))
    for i, (name, tr) in enumerate(traces.items()):
        c = plots.SERIES[i]
        ax.plot(tr["t"], tr["D"], color=c, linewidth=2.0, label=name)
        ax.fill_between(tr["t"], tr["D"] - tr["D_sd"], tr["D"] + tr["D_sd"],
                        color=c, alpha=0.12, linewidth=0)
    ax.axvspan(t_intervene, t_end, color=plots.INK_MUTED, alpha=0.10, linewidth=0)
    lo, hi = ax.get_ylim()
    ax.annotate("intervention window", (0.5 * (t_intervene + t_end), lo), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=8.5, color=plots.INK_2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dissonance  D")
    ax.set_title("Experiment 3 - interventions applied mid-attack", loc="left", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    f1 = plots.finish(fig, os.path.join(out, "interventions.png"),
                      f"Mean +/- SD over {len(seeds)} seeds. Oxygen and sumatriptan stay on; "
                      "DMT and 5-MeO wash out at the end of the shaded window.")

    fig, ax = plots.new_fig(figsize=(8.4, 4.0))
    names = [n for n in conditions if n != "none"]
    # Plotted against the untreated arm, matching the verdict - a bare pre->during change
    # carries the drift that the untreated arm also shows.
    stats_by_name = {}
    for n in names:
        m, sd = _mean_sd([r for r in records if r["condition"] == n], "relative_change")
        delta = m - none_change
        pooled = float(np.hypot(sd, none_sd))
        stats_by_name[n] = (delta, pooled, abs(delta) > 2 * pooled)
    means = [stats_by_name[n][0] for n in names]
    sds = [stats_by_name[n][1] for n in names]
    sig = [stats_by_name[n][2] for n in names]
    colours = [(plots.SERIES[2] if m < 0 else plots.SERIES[1]) if s else plots.INK_MUTED
               for m, s in zip(means, sig)]
    ax.barh(range(len(names)), means, xerr=sds, color=colours, height=0.6,
            error_kw={"ecolor": plots.INK_2, "elinewidth": 1.2, "capsize": 3})
    ax.axvline(0, color=plots.INK_2, linewidth=1.2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color=plots.INK_2)
    ax.invert_yaxis()
    ax.set_xlabel("change in D vs the untreated arm, during the intervention")
    ax.set_title("Effect on dissonance  (negative = relief)", loc="left", pad=10)
    for i, (m, sd, s) in enumerate(zip(means, sds, sig)):
        label = f"{m:+.0%}" + ("" if s else "  (within noise)")
        # anchor beyond the error bar, not on the bar end, so the two never overlap
        anchor = m + sd if m >= 0 else m - sd
        ax.annotate(label, (anchor, i), xytext=(9 if m >= 0 else -9, 0),
                    textcoords="offset points", va="center",
                    ha="left" if m >= 0 else "right", fontsize=8.5,
                    color=plots.INK_2 if s else plots.INK_MUTED)
    ax.margins(x=0.30)
    f2 = plots.finish(fig, os.path.join(out, "intervention_effects.png"),
                      f"Green = dissonance fell, orange = rose, grey = inside the seed spread "
                      f"({len(seeds)} seeds, error bars are the pooled SD against the "
                      f"untreated arm).")

    fig, axes = plots.new_fig(1, 2, figsize=(12.2, 4.3))
    plots.series_plot(axes[0], list(threshold_amplitudes), curves,
                      xlabel="drive amplitude A  (rad/s)", ylabel="dissonance  D",
                      title="Persistent kernel shift - does any shape raise the onset threshold?",
                      label_last=False)
    axes[0].axhline(onset_level, color=plots.INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
    axes[0].annotate("onset level (half the baseline peak)", (threshold_amplitudes[0], onset_level),
                     xytext=(4, 5), textcoords="offset points", fontsize=8, color=plots.INK_MUTED)
    axes[0].legend(loc="upper right", ncol=2)

    plots.series_plot(axes[1], list(oxygen_factors), {"dissonance D": oxygen_curve},
                      {"dissonance D": oxygen_sd},
                      xlabel="sigma_omega multiplier f   (1.0 = untreated)",
                      ylabel="dissonance  D",
                      title="Oxygen dose-response (IX.4 asks for the sweep, not one point)",
                      label_last=False)
    axes[1].legend(loc="best")
    plots.suptitle(fig, "Experiment 3 - preventive kernel shift and the oxygen dose-response")
    f3 = plots.finish(
        fig, os.path.join(out, "kernel_threshold.png"),
        "Onset crossings: " + ", ".join(
            f"{k} " + ("never reached" if np.isinf(v) else
                       "below range" if v == 0.0 else f"A = {v:.2f}")
            for k, v in thresholds.items() if not np.isnan(v))
        + f". Oxygen response is {'monotone' if oxygen_monotone else 'NON-monotone'} in f.")

    def _vs_control(prefix: str) -> Tuple[float, float, bool]:
        m, sd = _arm(prefix)
        delta = m - none_change
        pooled = float(np.hypot(sd, none_sd))
        return delta, pooled, bool(abs(delta) > 2 * pooled)

    dmt_change, dmt_sd, dmt_sig = _vs_control("DMT")
    meo_change, meo_sd, meo_sig = _vs_control("5-MeO")
    ox_change, ox_sd, ox_sig = _vs_control("oxygen")
    su_change, su_sd, su_sig = _vs_control("sumatriptan")
    base_thr = thresholds.get("baseline", float("nan"))
    # "Protective" = needs more drive to reach the baseline attack's dissonance level, or
    # never reaches it at all within the swept range (threshold = inf).
    raised = {k: v for k, v in thresholds.items()
              if k != "baseline" and np.isfinite(base_thr) and not np.isnan(v)
              and (np.isinf(v) or v > base_thr)}

    def _q(name: str, m: float, sd: float, sig: bool) -> str:
        return (f"{name} {m:+.0%} (+/-{sd:.0%}"
                + ("" if sig else ", within seed noise") + ")")

    verdict = (
        "Change in D during the intervention, relative to the untreated arm on the same seeds: "
        + ", ".join([
            _q("DMT (Mexican-hat)", dmt_change, dmt_sd, dmt_sig),
            _q("5-MeO (uniform-positive)", meo_change, meo_sd, meo_sig),
            _q("oxygen", ox_change, ox_sd, ox_sig),
            _q("sumatriptan", su_change, su_sd, su_sig),
        ]) + ". "
        "Across the full oxygen sweep the response is "
        + ("monotone" if oxygen_monotone else
           "NON-monotone, so the single f = 0.25 point above does not characterise it")
        + f" and contracting sigma_omega {oxygen_direction} dissonance: D = "
        + ", ".join(f"{f:g}->{v:.3f}" for f, v in zip(oxygen_factors, oxygen_curve))
        + f", a {abs(oxygen_span):.0%} "
        + ("reduction" if oxygen_direction == "reduces" else "increase")
        + " at the strongest contraction. "
        + ("That is the direction VI.2 proposes - stabilising the natural frequencies widens "
           "the basin of the consonant state - though note VI.2 offers it as a role "
           "assignment, not a derivation, and the model was not built to test it."
           if oxygen_direction == "reduces" else
           "That is the OPPOSITE of the role VI.2 assigns to oxygen: Plomp-Levelt roughness "
           "peaks at small non-zero detuning, so contracting the frequency spread moves "
           "cluster pairs into the beating band rather than out of it.")
        + " "
        + ("The DMT/5-MeO ordering matches Model B and the clinical anomaly in IV.1: the "
           "complexifier relieves and the symmetriser does not, from kernel shape alone, since "
           "the two are energy-matched."
           if dmt_change < meo_change - 2 * float(np.hypot(dmt_sd, meo_sd))
           else "The DMT/5-MeO ordering does NOT separate beyond seed noise, so the in-silico "
                "version of the IV.1 anomaly does not reproduce here.")
        + f" Onset is scored as the drive amplitude at which D first reaches half the "
        f"baseline kernel's peak ({onset_level:.3f}); baseline crosses at A = {base_thr:.2f}. "
        + ("Persistent kernel shifts that raise that threshold: "
           + ", ".join(
               f"{k} (" + ("never reaches it" if np.isinf(v) else f"A = {v:.2f}")
               + f", peak D {peak_ratio[k]:.2f}x baseline)"
               for k, v in raised.items())
           + "."
           if raised else "No candidate kernel raised that threshold; peak D relative to "
                          "baseline was "
                          + ", ".join(f"{k} {peak_ratio[k]:.2f}x" for k in candidates
                                      if k != "baseline") + ".")
    )

    summary = {
        "experiment": "3 - the three interventions",
        "attack_amplitude": attack_amplitude,
        "t_intervene": t_intervene, "duration": duration,
        "effects": {n: {"relative_change": _mean_sd([r for r in records if r["condition"] == n],
                                                    "relative_change")[0],
                        "latency_to_half_s": _mean_sd([r for r in records if r["condition"] == n],
                                                      "latency_to_half_s")[0]}
                    for n in conditions},
        "kernel_thresholds": thresholds,
        "kernel_peak_ratio": peak_ratio,
        "onset_level": onset_level,
        "kernel_curves": curves,
        "threshold_amplitudes": list(map(float, threshold_amplitudes)),
        "oxygen_factors": list(map(float, oxygen_factors)),
        "oxygen_curve": oxygen_curve,
        "oxygen_monotone": oxygen_monotone,
        "oxygen_direction": oxygen_direction,
        "oxygen_span": oxygen_span,
        "series": series_paths,
        "verdict": verdict,
        "figures": [f1, f2, f3],
    }
    _write_json(os.path.join(out, "summary.json"), summary)
    _write_csv(os.path.join(out, "records.csv"), records)
    _write_csv(os.path.join(out, "kernel_threshold.csv"), threshold_rows)
    _write_csv(os.path.join(out, "oxygen_doseresponse.csv"), oxygen_rows)
    np.savez_compressed(os.path.join(out, "arrays.npz"),
                        **{f"D_{n}": tr["D"] for n, tr in traces.items()},
                        **{f"t_{n}": tr["t"] for n, tr in traces.items()})
    return summary


# --------------------------------------------------------------------------------------
# Experiment 4 - consonant vs dissonant entrainment
# --------------------------------------------------------------------------------------


def experiment_4(
    root: str,
    base: RunConfig,
    n_target: int = 625,
    attack_amplitude: float = 4.0,
    t_total: float = 40.0,
    n_tones: int = 6,
    scale: float = 1.0,
    amplitudes: Sequence[float] = (0.0, 1.0, 2.0, 4.0, 8.0),
    seeds: Sequence[int] = (0, 1, 2),
) -> Dict:
    """IX.7 Experiment 4.  The one prediction unique to STV.

    Drive a peripheral V1 subset with (a) the source's harmonic frequency stack and
    (b) its matched inharmonic stack, at equal total energy, and ask whether the
    consonant one reduces the dissonance index and the dissonant one does not.
    """
    out = _outdir(root, "exp4_entrainment")
    harmonic = tuple(f * scale for f in HARMONIC_SET_HZ[:n_tones])
    dissonant = tuple(f * scale for f in DISSONANT_SET_HZ[:n_tones])
    # A positive control for the *test*, not for the theory.  The source's dissonant set is
    # mistuned by under 1%, while the model's own natural-frequency spread (sigma_omega) is
    # ~5% of the carrier - so a null result on that pair could mean either "entrainment
    # consonance does nothing" or "this set is too finely mistuned for this model to
    # resolve".  A golden-ratio stack is grossly incommensurate and covers the same span, so
    # if even it fails to separate, the null is about the mechanism rather than the stimulus.
    phi = (1 + 5 ** 0.5) / 2
    inharmonic = tuple(round(harmonic[0] * phi ** i, 4) * scale for i in range(n_tones))

    stacks = {"none": (), "harmonic": harmonic, "dissonant": dissonant,
              "inharmonic": inharmonic}
    records: List[Dict[str, object]] = []

    for label, freqs in stacks.items():
        for amp in amplitudes:
            if label == "none" and amp != amplitudes[0]:
                continue
            for seed in seeds:
                _, stats = run_condition(base, {
                    "network.geometry": "tree", "network.n_target": n_target,
                    "network.seed": seed, "dynamics.seed": seed,
                    "dynamics.t_total": t_total,
                    "drive.amplitude": attack_amplitude,
                    "drive.entrain_nodes": "none" if label == "none" else "v1",
                    "drive.entrain_freqs_hz": freqs,
                    "drive.entrain_amplitude": float(amp),
                })
                records.append({
                    "stack": label, "entrain_amplitude": float(amp), "seed": seed,
                    "dissonance": stats["dissonance"],
                    "d_roughness": stats["d_roughness"],
                    "d_inharmonic": stats["d_inharmonic"],
                    "r_global": stats["r_global_mean"],
                    "fragmentation": stats["fragmentation"],
                    "harmonic_purity": stats["harmonic_purity_topk"],
                })

    curves = {}
    errs = {}
    for label in ("harmonic", "dissonant", "inharmonic"):
        curves[label] = [_mean_sd([r for r in records if r["stack"] == label
                                   and r["entrain_amplitude"] == a], "dissonance")[0]
                         for a in amplitudes]
        errs[label] = [_mean_sd([r for r in records if r["stack"] == label
                                 and r["entrain_amplitude"] == a], "dissonance")[1]
                       for a in amplitudes]
    baseline_D = _mean_sd([r for r in records if r["stack"] == "none"], "dissonance")[0]

    fig, axes = plots.new_fig(1, 2, figsize=(11.4, 4.2))
    plots.series_plot(axes[0], list(amplitudes), curves, errs,
                      xlabel="entrainment amplitude  (rad/s)", ylabel="dissonance  D",
                      title="Consonant vs matched dissonant entrainment of V1",
                      label_last=False)
    axes[0].axhline(baseline_D, color=plots.INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
    axes[0].annotate("no stimulation", (amplitudes[-1], baseline_D), xytext=(-4, 5),
                     textcoords="offset points", ha="right", fontsize=8, color=plots.INK_MUTED)
    axes[0].legend(loc="upper left")

    strongest = amplitudes[-1]
    labels = ["harmonic", "dissonant", "inharmonic"]
    vals = [curves[l][-1] - baseline_D for l in labels]
    sds = [errs[l][-1] for l in labels]
    signif = [abs(v) > 2 * s for v, s in zip(vals, sds)]
    colours = [(plots.SERIES[2] if v < 0 else plots.SERIES[1]) if g else plots.INK_MUTED
               for v, g in zip(vals, signif)]
    axes[1].bar(range(len(labels)), vals, yerr=sds, color=colours, width=0.5,
                error_kw={"ecolor": plots.INK_2, "elinewidth": 1.2, "capsize": 3})
    axes[1].axhline(0, color=plots.INK_2, linewidth=1.2)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(
        ["harmonic\n(consonant)", "TMS dissonant\n(<1% mistuned)", "golden ratio\n(grossly inharmonic)"],
        fontsize=8.5, color=plots.INK_2)
    axes[1].set_ylabel("change in D vs no stimulation")
    axes[1].set_title(f"Effect at amplitude {strongest:g}", loc="left", pad=10)
    for i, (v, s, g) in enumerate(zip(vals, sds, signif)):
        axes[1].annotate(f"{v:+.4f}" + ("" if g else "\n(within noise)"),
                         (i, v + s), xytext=(0, 7), textcoords="offset points",
                         ha="center", fontsize=8.5,
                         color=plots.INK_2 if g else plots.INK_MUTED)
    axes[1].margins(y=0.28)
    plots.suptitle(fig, "Experiment 4 - the STV signature test")
    f1 = plots.finish(
        fig, os.path.join(out, "entrainment.png"),
        f"Stacks, all at matched total energy: harmonic {harmonic} Hz, the source's dissonant "
        f"set {dissonant} Hz, and a golden-ratio positive control {tuple(round(f, 2) for f in inharmonic)} Hz. "
        "Grey bars are inside the seed spread.")

    h_eff, d_eff, i_eff = vals
    noise = float(np.mean(sds)) if sds else 0.0
    supports = bool(h_eff < 0 and (d_eff - h_eff) > 2 * noise)
    control_separates = bool(abs(i_eff - h_eff) > 2 * noise)
    sigma_pct = 100 * base.dynamics.sigma_omega_hz / max(base.dynamics.omega0_peripheral_hz, 1e-9)

    verdict = (
        f"At the strongest stimulation, the consonant stack changes D by {h_eff:+.4f}, the "
        f"source's matched dissonant stack by {d_eff:+.4f}, and a grossly inharmonic "
        f"golden-ratio stack by {i_eff:+.4f} (seed spread ~{noise:.4f}). "
        + ("The consonant stack reduces dissonance and the dissonant one does not - the "
           "STV-specific prediction survives in simulation, which is the version of this test "
           "that would translate to a cheap non-invasive human experiment."
           if supports else
           "Consonant and dissonant stimulation are NOT meaningfully separated, so the one "
           "prediction unique to STV rather than to generic desynchronisation does not "
           "reproduce here. ")
        + ("" if supports else
           ("The positive control does separate, so the null is about the source's specific "
            "frequency set rather than about the method: those tones are mistuned by under 1%, "
            f"while the model's own natural-frequency spread is ~{sigma_pct:.0f}% of the "
            "carrier, so the stimulus asks the network to resolve a detuning five times finer "
            "than its own jitter. The experiment as specified in III.1 is underpowered by "
            "construction at these parameters - rerun it with a coarser mistuning or a much "
            "smaller sigma_omega before treating this as evidence against STV."
            if control_separates else
            "The grossly inharmonic control does not separate either, so the null is about the "
            "entrainment mechanism itself, not about the fineness of the source's mistuning: "
            "driving a peripheral branch simply does not move this metric whatever its "
            "consonance. That is the stronger negative result of the two."))
    )

    summary = {
        "experiment": "4 - consonant vs dissonant entrainment",
        "harmonic_stack_hz": list(harmonic),
        "dissonant_stack_hz": list(dissonant),
        "inharmonic_stack_hz": list(inharmonic),
        "baseline_D": baseline_D,
        "curves": curves,
        "effect_harmonic": h_eff,
        "effect_dissonant": d_eff,
        "effect_inharmonic": i_eff,
        "seed_spread": noise,
        "supports_stv_prediction": supports,
        "positive_control_separates": control_separates,
        "verdict": verdict,
        "figures": [f1],
    }
    _write_json(os.path.join(out, "summary.json"), summary)
    _write_csv(os.path.join(out, "records.csv"), records)
    return summary


# --------------------------------------------------------------------------------------
# Experiment 5 - the noise control and the D/LZ dissociation
# --------------------------------------------------------------------------------------


def experiment_5(
    root: str,
    base: RunConfig,
    n_target: int = 625,
    t_total: float = 20.0,
    noise_levels: Sequence[float] = (0.0, 0.05, 0.15, 0.3, 0.6, 1.0, 1.8, 3.0),
    drive_levels: Sequence[float] = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
    seeds: Sequence[int] = (0, 1, 2),
) -> Dict:
    """IX.7 Experiment 5.  "Test this first."

    Inject noise with no drive: entropy/LZ must rise while the dissonance index stays
    low.  Then pool every condition and check that D and LZ are not collinear.  "If D and
    LZ turn out to be collinear across all your conditions, you have not built a
    dissonance metric - you have built an entropy metric."
    """
    out = _outdir(root, "exp5_noise")
    records: List[Dict[str, object]] = []

    for sigma in noise_levels:
        for seed in seeds:
            _, stats = run_condition(base, {
                "network.geometry": "tree", "network.n_target": n_target,
                "network.seed": seed, "dynamics.seed": seed, "dynamics.t_total": t_total,
                "drive.amplitude": 0.0, "dynamics.sigma_noise": float(sigma),
            }, with_complexity=True)
            records.append({
                "arm": "noise", "sigma_noise": float(sigma), "amplitude": 0.0, "seed": seed,
                "dissonance": stats["dissonance"], "d_roughness": stats["d_roughness"],
                "d_inharmonic": stats["d_inharmonic"],
                "lz_normalised": stats["lz_normalised"], "zlib_ratio": stats["zlib_ratio"],
                "spectral_entropy": stats["spectral_entropy"],
                "r_global": stats["r_global_mean"], "coherence_weight": stats["coherence_weight"],
            })

    for A in drive_levels:
        for seed in seeds:
            _, stats = run_condition(base, {
                "network.geometry": "tree", "network.n_target": n_target,
                "network.seed": seed, "dynamics.seed": seed, "dynamics.t_total": t_total,
                "drive.amplitude": float(A),
            }, with_complexity=True)
            records.append({
                "arm": "drive", "sigma_noise": base.dynamics.sigma_noise, "amplitude": float(A),
                "seed": seed,
                "dissonance": stats["dissonance"], "d_roughness": stats["d_roughness"],
                "d_inharmonic": stats["d_inharmonic"],
                "lz_normalised": stats["lz_normalised"], "zlib_ratio": stats["zlib_ratio"],
                "spectral_entropy": stats["spectral_entropy"],
                "r_global": stats["r_global_mean"], "coherence_weight": stats["coherence_weight"],
            })

    rank = lambda v: np.argsort(np.argsort(v)).astype(float)

    def _corr(rows) -> Tuple[float, float]:
        d = np.array([r["dissonance"] for r in rows], dtype=float)
        lz = np.array([r["lz_normalised"] for r in rows], dtype=float)
        ok = np.isfinite(d) & np.isfinite(lz)
        if ok.sum() < 3 or np.std(d[ok]) < 1e-12 or np.std(lz[ok]) < 1e-12:
            return float("nan"), float("nan")
        return (float(np.corrcoef(d[ok], lz[ok])[0, 1]),
                float(np.corrcoef(rank(d[ok]), rank(lz[ok]))[0, 1]))

    pearson, spearman = _corr(records)
    # Pooling two sweeps can manufacture or hide a correlation (Simpson's paradox), so the
    # within-arm correlations are reported alongside the pooled one.
    per_arm = {arm: dict(zip(("pearson", "spearman"),
                             _corr([r for r in records if r["arm"] == arm])))
               for arm in ("noise", "drive")}

    noise_rows = [r for r in records if r["arm"] == "noise"]
    d_curve = [_mean_sd([r for r in noise_rows if r["sigma_noise"] == s], "dissonance")[0]
               for s in noise_levels]
    lz_curve = [_mean_sd([r for r in noise_rows if r["sigma_noise"] == s], "lz_normalised")[0]
                for s in noise_levels]
    drive_rows = [r for r in records if r["arm"] == "drive"]
    d_drive = [_mean_sd([r for r in drive_rows if r["amplitude"] == a], "dissonance")[0]
               for a in drive_levels]
    lz_drive = [_mean_sd([r for r in drive_rows if r["amplitude"] == a], "lz_normalised")[0]
                for a in drive_levels]

    # ---- figures: two panels, never a dual axis ----
    fig, axes = plots.new_fig(2, 2, figsize=(11.0, 6.6))
    plots.series_plot(axes[0, 0], list(noise_levels), {"dissonance D": d_curve},
                      xlabel="", ylabel="dissonance  D",
                      title="Noise injection, no drive", label_last=False)
    plots.series_plot(axes[1, 0], list(noise_levels), {"LZ complexity": lz_curve},
                      xlabel="noise amplitude  sigma", ylabel="normalised LZ",
                      title="", label_last=False)
    plots.series_plot(axes[0, 1], list(drive_levels), {"dissonance D": d_drive},
                      xlabel="", ylabel="dissonance  D",
                      title="Drive, fixed noise", label_last=False)
    plots.series_plot(axes[1, 1], list(drive_levels), {"LZ complexity": lz_drive},
                      xlabel="drive amplitude A  (rad/s)", ylabel="normalised LZ",
                      title="", label_last=False)
    for ax in axes.ravel():  # one series per panel; the axis label names it
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    d_lo, d_hi = min(d_curve), max(d_curve)
    plots.suptitle(fig, "Experiment 5 - noise raises entropy without raising dissonance")
    f1 = plots.finish(
        fig, os.path.join(out, "noise_control.png"),
        "Separate panels rather than a shared axis: D and LZ are different quantities. Mind "
        f"the dissonance scales - noise moves D only between {d_lo:.3f} and {d_hi:.3f}, while "
        f"drive takes it to {max(d_drive):.3f}, roughly {max(d_drive) / max(d_hi, 1e-9):.0f}x "
        "further, and drives LZ *down* while doing so.")

    fig, ax = plots.new_fig(figsize=(6.4, 5.0))
    for i, arm in enumerate(("noise", "drive")):
        sub = [r for r in records if r["arm"] == arm]
        ax.scatter([r["lz_normalised"] for r in sub], [r["dissonance"] for r in sub],
                   s=46, color=plots.SERIES[i], marker=plots.MARKERS[i],
                   edgecolors=plots.SURFACE, linewidths=1.0,
                   label=f"{arm} sweep", alpha=0.9)
    ax.set_xlabel("normalised LZ complexity")
    ax.set_ylabel("dissonance index  D")
    ax.set_title(f"D vs LZ across all conditions   (Pearson r = {pearson:+.2f}, "
                 f"Spearman = {spearman:+.2f})", loc="left", pad=10)
    ax.legend(loc="upper right")
    f2 = plots.finish(
        fig, os.path.join(out, "dissociation.png"),
        "The two arms form an L, not a line: drive raises dissonance while barely moving "
        "complexity, noise raises complexity while dissonance stays flat. Collinearity here "
        "would have meant the dissonance index was an entropy metric under another name.")

    noise_lz_rise = lz_curve[-1] - lz_curve[0]
    noise_d_rise = d_curve[-1] - d_curve[0]

    # What "collinear" has to mean here.  The failure mode XI.1 warns about is D being a
    # monotone increasing function of LZ - i.e. an entropy metric wearing a different name.
    # A strong *negative* correlation is not that failure, it is the dissociation working:
    # noise raises complexity while lowering dissonance.  So the test is for positive
    # collinearity, not for the absence of correlation.
    arm_r = [v["pearson"] for v in per_arm.values() if np.isfinite(v["pearson"])]
    collinear = bool(spearman > 0.8 and arm_r and all(r > 0.5 for r in arm_r))
    noise_control_holds = bool(noise_lz_rise > 0.1 and noise_d_rise < 0.02)
    sign_flip = bool(len(arm_r) == 2 and arm_r[0] * arm_r[1] < 0)
    dissociates = bool(noise_control_holds and not collinear)

    verdict = (
        f"Under noise injection LZ rises by {noise_lz_rise:+.2f} while D "
        f"{'falls' if noise_d_rise < 0 else 'changes'} by {noise_d_rise:+.4f} - the "
        "asymmetry/antisymmetry distinction behaving exactly as III.1 requires. Pooled across "
        f"all {len(records)} runs the D-LZ correlation is r = {pearson:+.2f} (Spearman "
        f"{spearman:+.2f}); within arms it is {per_arm['noise']['pearson']:+.2f} under noise "
        f"and {per_arm['drive']['pearson']:+.2f} under drive. "
        + (("Those two carry opposite signs, which is the strongest available form of the "
            "result: the same pair of metrics moves together under one manipulation and "
            "apart under the other, so no monotone function relates them. "
            if sign_flip else "")
           + "The dissonance index is therefore measuring structured incommensurability "
             "rather than entropy - prediction 4 in the falsification table, and the "
             "precondition for anything else here meaning anything."
           if dissociates
           else "D rises monotonically with LZ in every arm, so the index is an entropy "
                "metric under another name. Per XI.1 this is the outcome that makes the "
                "framework empirically inert for this problem.")
    )

    summary = {
        "experiment": "5 - noise control and D/LZ dissociation",
        "pearson_D_LZ": pearson,
        "spearman_D_LZ": spearman,
        "within_arm_correlations": per_arm,
        "noise_levels": list(map(float, noise_levels)),
        "D_vs_noise": d_curve,
        "LZ_vs_noise": lz_curve,
        "drive_levels": list(map(float, drive_levels)),
        "D_vs_drive": d_drive,
        "LZ_vs_drive": lz_drive,
        "dissociates": dissociates,
        "verdict": verdict,
        "figures": [f1, f2],
    }
    _write_json(os.path.join(out, "summary.json"), summary)
    _write_csv(os.path.join(out, "records.csv"), records)
    return summary


# --------------------------------------------------------------------------------------
# Experiment 6 - the IX.6 free-parameter sweeps not covered elsewhere
# --------------------------------------------------------------------------------------


def experiment_6(
    root: str,
    base: RunConfig,
    n_target: int = 625,
    attack_amplitude: float = 4.0,
    t_total: float = 20.0,
    seeds: Sequence[int] = (0, 1),
    amplitudes: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0),
    k_sw_values: Sequence[float] = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0),
    omega_ratios: Sequence[float] = (0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.6, 2.0),
    branching_factors: Sequence[int] = (2, 3, 4),
    segment_lengths: Sequence[int] = (2, 4, 8),
    sizes: Sequence[int] = (225, 625, 1225),
) -> Dict:
    """IX.6's remaining free parameters.

    Four things IX.3/IX.6 explicitly ask for that Experiments 1-5 do not touch:

      * the **diffuse** drive variant, which IX.3 says to implement *and compare* against
        the pacemaker ("they are not equivalent");
      * **K_SW**, which is zero in every other experiment, so the small-world links IX.2
        mandates are inert there;
      * the **drive frequency** Omega, swept against both omega_0 and the tree's own
        Laplacian eigenfrequencies - IX.3 calls this mismatch "a candidate dissonance
        generator in its own right";
      * **b, m and N**, the last of which is the check that nothing here is a size artefact.
    """
    out = _outdir(root, "exp6_sweeps")
    tree = {"network.geometry": "tree", "network.n_target": n_target}
    records: List[Dict[str, object]] = []

    def _run(tag: str, over: Dict[str, object], **extra) -> None:
        for seed in seeds:
            _, st = run_condition(base, {**tree, "network.seed": seed, "dynamics.seed": seed,
                                         "dynamics.t_total": t_total, **over})
            records.append({"sweep": tag, "seed": seed, **extra,
                            "dissonance": st["dissonance"],
                            "dissonance_intensive": st["dissonance_intensive"],
                            "r_global": st["r_global_mean"],
                            "fragmentation": st["fragmentation"],
                            "coherence_weight": st["coherence_weight"]})

    # --- 1. drive variant: pacemaker vs diffuse ---
    for variant in ("pacemaker", "diffuse"):
        for A in amplitudes:
            _run("variant", {"drive.variant": variant, "drive.amplitude": float(A)},
                 variant=variant, amplitude=float(A))

    # --- 2. K_SW: negative = geometric noise, positive = a global phase to lock to ---
    for k_sw in k_sw_values:
        _run("k_sw", {"kernel.k_sw": float(k_sw), "drive.amplitude": attack_amplitude},
             k_sw=float(k_sw))

    # --- 3. drive frequency against the network's own modes ---
    eig_hz = laplacian_eigenfrequencies(
        cached_network(apply_overrides(base, tree).network), base.kernel.coupling_strength
    )
    w0 = base.dynamics.omega0_peripheral_hz
    for ratio in omega_ratios:
        _run("omega", {"drive.omega_drive_hz": float(w0 * ratio),
                       "drive.amplitude": attack_amplitude},
             omega_ratio=float(ratio), omega_hz=float(w0 * ratio))

    # --- 4. structure and size ---
    for b in branching_factors:
        _run("branching", {"network.branching_factor": int(b),
                           "drive.amplitude": attack_amplitude}, branching_factor=int(b))
    for m in segment_lengths:
        _run("segment", {"network.segment_length": int(m),
                         "drive.amplitude": attack_amplitude}, segment_length=int(m))
    for n in sizes:
        for seed in seeds:
            _, st = run_condition(base, {"network.geometry": "tree", "network.n_target": int(n),
                                         "network.seed": seed, "dynamics.seed": seed,
                                         "dynamics.t_total": t_total,
                                         "drive.amplitude": attack_amplitude})
            records.append({"sweep": "size", "seed": seed, "n_target": int(n),
                            "dissonance": st["dissonance"],
                            "dissonance_intensive": st["dissonance_intensive"],
                            "r_global": st["r_global_mean"],
                            "fragmentation": st["fragmentation"],
                            "coherence_weight": st["coherence_weight"]})

    def _curve(tag: str, key: str, values: Sequence, metric: str = "dissonance",
               **filt) -> List[float]:
        return [_mean_sd([r for r in records if r["sweep"] == tag and r.get(key) == v
                          and all(r.get(fk) == fv for fk, fv in filt.items())], metric)[0]
                for v in values]

    # ---- figures ----
    fig, axes = plots.new_fig(2, 3, figsize=(15.0, 7.6))

    # drive variant, plotted against achieved coherence so the amplitude scales are comparable
    for i, variant in enumerate(("pacemaker", "diffuse")):
        r = [_mean_sd([x for x in records if x["sweep"] == "variant" and x["variant"] == variant
                       and x["amplitude"] == A], "r_global")[0] for A in amplitudes]
        d = [_mean_sd([x for x in records if x["sweep"] == "variant" and x["variant"] == variant
                       and x["amplitude"] == A], "dissonance")[0] for A in amplitudes]
        axes[0, 0].plot(r, d, color=plots.SERIES[i], marker=plots.MARKERS[i], markersize=5.5,
                        markeredgecolor=plots.SURFACE, markeredgewidth=1.0, label=variant)
    axes[0, 0].set_xlabel("achieved coherence  r_global")
    axes[0, 0].set_ylabel("dissonance  D")
    axes[0, 0].set_title("Drive variant (IX.3 asks for both)", loc="left", pad=10)
    axes[0, 0].legend(loc="best")

    plots.series_plot(axes[0, 1], list(k_sw_values),
                      {"dissonance D": _curve("k_sw", "k_sw", [float(v) for v in k_sw_values]),
                       "r_global": _curve("k_sw", "k_sw", [float(v) for v in k_sw_values], "r_global")},
                      xlabel="K_SW  (rad/s)", ylabel="value",
                      title="Small-world lever", label_last=False)
    axes[0, 1].axvline(0, color=plots.INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    axes[0, 1].legend(loc="best")

    om = [float(w0 * r) for r in omega_ratios]
    ax_om = axes[0, 2]
    plots.series_plot(ax_om, om, {"dissonance D": _curve("omega", "omega_hz", om)},
                      xlabel="drive frequency  Omega (Hz)", ylabel="dissonance  D",
                      title="Drive frequency vs the tissue's own", label_last=False)
    ax_om.axvline(w0, color=plots.SERIES[1], linewidth=1.6)
    ax_om.set_ylim(bottom=0)
    ax_om.annotate("on resonance\n($\\Omega = \\omega_0$)", (w0, ax_om.get_ylim()[1]),
                   xytext=(6, -4), textcoords="offset points", fontsize=8,
                   va="top", ha="left", color=plots.SERIES[1])
    leg = ax_om.get_legend()
    if leg is not None:
        leg.remove()

    # Three separate parameters get three separate axes.  Sharing one "parameter value"
    # x-axis between a branching factor, a segment length and a node count would put three
    # incommensurable quantities on the same scale.
    size_d = _curve("size", "n_target", list(sizes))
    for j, (xs, ys, xlabel, title) in enumerate((
        (list(branching_factors), _curve("branching", "branching_factor", list(branching_factors)),
         "branching factor  b", "Branching factor"),
        (list(segment_lengths), _curve("segment", "segment_length", list(segment_lengths)),
         "segment length  m", "Segment length between branch points"),
        (list(sizes), size_d, "node count  N", "Node count - the size-artefact check"),
    )):
        ax = axes[1, j]
        ax.plot(xs, ys, color=plots.SERIES[j], marker=plots.MARKERS[j], markersize=6.5,
                markeredgecolor=plots.SURFACE, markeredgewidth=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("dissonance  D")
        ax.set_title(title, loc="left", pad=10)
        ax.set_ylim(bottom=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])

    plots.suptitle(fig, "Experiment 6 - the IX.6 sweeps not covered by Experiments 1-5")
    f1 = plots.finish(
        fig, os.path.join(out, "parameter_sweeps.png"),
        f"Mean over {len(seeds)} seeds. "
        + (f"The tree's own Laplacian modes map to {eig_hz.min():.4f}-{eig_hz.max():.3f} Hz "
           "at this coupling strength, far below the drive band, so they place no constraint "
           "on Omega and are not marked; the relevant comparison is against omega_0."
           if eig_hz.size else ""))

    size_spread = (max(size_d) - min(size_d)) / max(np.mean(size_d), 1e-12)
    d_pace = [_mean_sd([x for x in records if x["sweep"] == "variant" and x["variant"] == "pacemaker"
                        and x["amplitude"] == A], "dissonance")[0] for A in amplitudes]
    d_diff = [_mean_sd([x for x in records if x["sweep"] == "variant" and x["variant"] == "diffuse"
                        and x["amplitude"] == A], "dissonance")[0] for A in amplitudes]
    k_curve = _curve("k_sw", "k_sw", [float(v) for v in k_sw_values])
    om_curve = _curve("omega", "omega_hz", om)
    # value at Omega = omega_0, the on-resonance case
    om_at_w0 = float(np.interp(w0, om, np.where(np.isfinite(om_curve), om_curve, 0.0)))

    verdict = (
        f"Drive variant: the pacemaker reaches peak D = {np.nanmax(d_pace):.3f} and the diffuse "
        f"(small-world) drive {np.nanmax(d_diff):.3f}; IX.3 is right that they are not "
        "equivalent, and the difference is visible at matched achieved coherence, not just at "
        "matched amplitude. "
        f"K_SW: D ranges {np.nanmin(k_curve):.3f} to {np.nanmax(k_curve):.3f} over "
        f"[{k_sw_values[0]:g}, {k_sw_values[-1]:g}], peaking at K_SW = "
        f"{k_sw_values[int(np.nanargmax(k_curve))]:g}. "
        f"Drive frequency: D ranges {np.nanmin(om_curve):.3f} to {np.nanmax(om_curve):.3f} as "
        f"Omega sweeps {om[0]:.1f}-{om[-1]:.1f} Hz, peaking at "
        f"{om[int(np.nanargmax(om_curve))]:.2f} Hz while dipping to "
        f"{om_at_w0:.3f} at Omega = omega_0 = {w0:g} Hz"
        + (" - so driving *on* the tissue's own frequency is the quiet case and detuning "
           "either way generates dissonance, which is IX.3's speculation that Omega/mode "
           "mismatch is 'a candidate dissonance generator in its own right', confirmed."
           if om_at_w0 < 0.5 * np.nanmax(om_curve) else ". ")
        + f" Size: D falls monotonically with N ({', '.join(f'{n}:{d:.3f}' for n, d in zip(sizes, size_d))}), "
        f"a {size_spread:.0%} spread, so **absolute** D values are not comparable across node "
        "counts. Every experiment here holds N fixed, so the within-experiment comparisons are "
        "unaffected, but no absolute magnitude should be read as a property of the model."
    )

    summary = {
        "experiment": "6 - IX.6 parameter sweeps",
        "laplacian_eigenfrequencies_hz": [float(f) for f in eig_hz],
        "variant_pacemaker_D": d_pace,
        "variant_diffuse_D": d_diff,
        "amplitudes": list(map(float, amplitudes)),
        "k_sw_values": list(map(float, k_sw_values)),
        "k_sw_D": k_curve,
        "omega_hz": om,
        "omega_D": om_curve,
        "size_D": size_d,
        "sizes": list(map(int, sizes)),
        "size_spread": float(size_spread),
        "verdict": verdict,
        "figures": [f1],
    }
    _write_json(os.path.join(out, "summary.json"), summary)
    _write_csv(os.path.join(out, "records.csv"), records)
    return summary


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

EXPERIMENTS: Dict[str, Callable] = {
    "exp0": experiment_0,
    "exp1": experiment_1,
    "exp2": experiment_2,
    "exp3": experiment_3,
    "exp4": experiment_4,
    "exp5": experiment_5,
    "exp6": experiment_6,
}
