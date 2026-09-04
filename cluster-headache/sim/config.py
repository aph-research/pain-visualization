"""Configuration objects for the cluster-headache field-dissonance simulations.

Every free parameter named in Part IX.6 of the theory document (and a few more that
turned out to be load-bearing) is a field on one of these dataclasses.  Nothing is
hard-coded in the dynamics or metric code.

Overrides are applied with dotted paths, e.g.

    cfg = RunConfig()
    cfg = apply_overrides(cfg, {"dynamics.sigma_omega": 0.4, "drive.amplitude": 2.0})

so the CLI can expose the whole tree without a flag per parameter.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------------------


@dataclass
class NetworkConfig:
    """Graph construction (IX.2)."""

    geometry: str = "tree"  # "tree" | "lattice" | "hierarchical"
    n_target: int = 600  # matched across geometries; 500 for sweeps, 1000-4000 for finals
    seed: int = 0

    # --- trigeminal branching tree (A) ---
    central_fraction: float = 0.05  # trigeminocervical complex / thalamic relay
    central_density: float = 0.6  # edge probability inside the central compartment
    branching_factor: int = 3  # b
    segment_length: int = 4  # m, nodes along a branch between bifurcations
    max_depth: int = 12  # L cap
    division_weights: Tuple[float, float, float] = (0.6, 0.2, 0.2)  # V1, V2, V3
    v1_only: bool = False
    # Log-normal segment lengths and Poisson branching.  Note this gives *mild* degree and
    # segment heterogeneity, not a heavy tail: peripheral degree stays within 1..6 and ~88%
    # of nodes are degree 1 or 2.  IX.2 offers this as optional realism toward a scale-free
    # degree distribution; it does not get there, so a null result under this switch is not
    # evidence that scale-free structure does not matter.
    heterogeneous: bool = True
    lognormal_sigma: float = 0.35

    # --- 2D lattice (B) ---
    lattice_periodic: bool = True

    # --- hierarchical layered network (C) ---
    hier_base_side: int = 16  # 16x16 -> 8x8 -> 4x4 -> 2x2, auto-scaled to n_target
    hier_layers: int = 4

    # --- distance shells (shared) ---
    n_shells: int = 4
    shell_cap: int = 8  # D4 = 4..shell_cap hops

    # --- small-world links (shared) ---
    sw_links_per_node: int = 2

    # --- cluster partition used by the local order parameter / dissonance index ---
    target_clusters: int = 16


# --------------------------------------------------------------------------------------
# Coupling kernel
# --------------------------------------------------------------------------------------


@dataclass
class KernelConfig:
    """The kernel is the vector (K1..K4) plus the small-world lever K_SW (III.2)."""

    name: str = "baseline"
    k: Tuple[float, float, float, float] = (0.6, 0.2, -0.1, -0.2)
    k_sw: float = 0.0

    # How the shell sums are normalised.
    #   "global"    - one 1/N_i in front of the whole bracket, exactly as written in III.2
    #   "per_shell" - each shell's sum divided by that shell's own neighbour count
    # This is not cosmetic.  D4 (4..cap hops) holds an order of magnitude more neighbours
    # than D1, so under "global" the K4 term dominates and the kernel *shape* the theory
    # cares about is swamped by a shell-size artefact.  Under "per_shell", sum(K_s) is the
    # coupling seen by a fully synchronised node, so the sign structure of (K1..K4) means
    # what III.2 says it means.  Default per_shell; "global" reproduces the literal formula.
    #
    # per_shell also makes IX.4 energy matching exact: max |C_i| = sum |K_s| for every
    # node, so two kernel shapes are matched node by node.  Under "global" a single scalar
    # rescale can only match the *mean* (to ~2%); per-node coupling still spreads ~25%
    # across kernel shapes, because each node has its own shell composition.
    shell_normalisation: str = "per_shell"

    # "auto" picks the energy measure that matches shell_normalisation (l1 for per_shell,
    # shell_weighted for global).  The wrong pairing silently breaks IX.4 energy matching.
    # "none" | "l1" (match sum |K_s|) | "shell_weighted" (match sum |K_s| * <shell size>)
    energy_match: str = "auto"
    # Target for max |C_i| in rad/s.  Every kernel - baseline and intervention alike - is
    # rescaled to this before use, which is how IX.4's "match total coupling energy across
    # kernel conditions" is enforced: only the *shape* of (K1..K4) can differ.
    coupling_strength: float = 4.0
    gain: float = 1.0  # extra global multiplier, applied after normalisation
    # "shared"   - divide the small-world term by N_i, exactly as written in III.2
    # "separate" - divide it by the small-world degree instead, so that K_SW is directly
    #              in rad/s and the diffuse and pacemaker drives are on the same scale
    sw_normalisation: str = "separate"
    # Rescale a preset's K_SW by the same factor its (K1..K4) receives.  The source's
    # tabulated kernels quote K_SW on the same [-20, 20] scale as the shell constants, so
    # the ratio K_SW / |K| is the meaningful quantity; normalising the shells without it
    # would silently make K_SW an order of magnitude stronger than the source intended.
    scale_k_sw: bool = True


#: Kernels tabulated in III.2, plus the ones the interventions need (IX.4).
KERNEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # --- source table, [-1, 1] convention ---
    "dmt_plate": {"k": (-1.00, 0.09, -0.63, -0.90), "k_sw": 0.0},
    "dmt_plate_variant": {"k": (-0.99, 0.17, -1.00, -0.64), "k_sw": 0.0},
    "5meo_plate": {"k": (0.93, 0.17, 0.61, -0.64), "k_sw": 0.0},
    "dmt_branching": {"k": (-1.0, 0.0, -0.9, 0.3), "k_sw": 0.0},
    # --- source table, [-20, 20] convention (normalised on load) ---
    "checkerboard": {"k": (-20.0, 15.8, 12.1, -16.8), "k_sw": -2.4},
    "pinwheel": {"k": (19.0, 15.8, 14.0, 1.2), "k_sw": -2.4},
    "traveling_wave": {"k": (19.0, 4.1, -5.4, 20.0), "k_sw": -2.4},
    # --- reference shapes ---
    "flat": {"k": (1.0, 1.0, 1.0, 1.0), "k_sw": 0.0},
    "mexican_hat": {"k": (-1.0, 0.1, -0.6, -0.9), "k_sw": 0.0},
    # --- interior/attack baseline: locally cohesive, mildly inhibitory at range ---
    "baseline": {"k": (0.6, 0.2, -0.1, -0.2), "k_sw": 0.0},
    # --- persistent post-psychedelic retune (see IX.4 "Psilocybin / LSD") ---
    "retuned": {"k": (0.35, 0.35, 0.05, -0.05), "k_sw": 0.0},
}


def kernel_from_preset(name: str, **overrides: Any) -> KernelConfig:
    if name not in KERNEL_PRESETS:
        raise KeyError(f"unknown kernel preset {name!r}; have {sorted(KERNEL_PRESETS)}")
    spec = dict(KERNEL_PRESETS[name])
    spec.update(overrides)
    return KernelConfig(name=name, **spec)


# --------------------------------------------------------------------------------------
# Hypothalamic drive
# --------------------------------------------------------------------------------------


@dataclass
class DriveConfig:
    """The synchronising drive D_i(t) (IX.3).

    Two variants, implemented separately because they are *not* equivalent:
      - "diffuse"   : folds into K_SW, K_SW(t) = k_sw_base + A g(t)
      - "pacemaker" : explicit driver oscillator at Omega, D_i = A g(t) sin(theta_drive - theta_i)
    """

    variant: str = "pacemaker"  # "pacemaker" | "diffuse" | "none"
    amplitude: float = 0.0  # A
    omega_drive_hz: float = 5.0  # Omega, only used by the pacemaker variant
    k_sw_base: float = 0.0  # baseline small-world coupling for the diffuse variant
    # Where the pacemaker injects.  IX.3 Variant 2 says "coupled to all nodes", which is
    # the default; "central" instead injects only at the trigeminocervical/thalamic relay
    # and lets the drive propagate out through the geometry, which is closer to V.1's
    # "feed that drive into the trigeminovascular tree" and is a decisive lever - see the
    # Experiment 1 results.
    target: str = "all"  # "all" | "central" | "peripheral" | "v1"

    # duty cycle g(t)
    waveform: str = "constant"  # "constant" | "square" | "raised_cosine" | "ramp" | "ramp_updown"
    period_s: float = 20.0  # T_circ (compressed circadian period, simulation seconds)
    duty: float = 0.2  # d in 0.05-0.3 (attack duration / inter-attack interval)
    ramp_s: float = 1.0  # edge smoothing for the square wave
    phase_offset: float = 0.0  # fraction of the period

    # optional focal entrainment (Experiment 4) - additive sinusoidal forcing on a subset
    entrain_nodes: str = "none"  # "none" | "v1" | "peripheral" | "central"
    entrain_freqs_hz: Tuple[float, ...] = ()
    entrain_amplitude: float = 0.0
    entrain_match_energy: bool = True


#: The signature STV stimulation sets from III.1 (Hz).
HARMONIC_SET_HZ: Tuple[float, ...] = (1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 148)
DISSONANT_SET_HZ: Tuple[float, ...] = (
    1.01, 2.01, 3.98, 6.02, 7.99, 12.03, 16.01, 24.02, 35.97, 48.05, 72.04, 95.94, 147.93,
)


# --------------------------------------------------------------------------------------
# Dynamics
# --------------------------------------------------------------------------------------


@dataclass
class DynamicsConfig:
    """Integration and the intrinsic-frequency distribution (IX.3)."""

    dt: float = 0.002  # s
    t_total: float = 40.0  # s
    transient_frac: float = 0.25  # discarded before metrics
    integrator: str = "euler"  # "euler" | "rk4"

    omega0_central_hz: float = 5.0  # thalamic / trigeminocervical burst band
    omega0_peripheral_hz: float = 5.6  # heterogeneous omega0 by compartment
    sigma_omega_hz: float = 0.25  # ***the oxygen handle***
    sigma_noise: float = 0.05  # rad / sqrt(s), Euler-Maruyama

    record_every: int = 10  # store every k-th step
    seed: int = 0


# --------------------------------------------------------------------------------------
# Interventions
# --------------------------------------------------------------------------------------


@dataclass
class Intervention:
    """A timed modification of the running system (IX.4).

    kind:
      "kernel"        - swap (K1..K4)/K_SW for the window; energy-matched to baseline
      "sigma_omega"   - contract natural frequencies toward omega0 by `factor`
      "gain"          - scale the coupling gain of a compartment
      "noise"         - scale sigma_noise
      "drive"         - scale the drive amplitude
    """

    name: str = ""
    kind: str = "kernel"
    t_start: float = 0.0
    t_end: float = float("inf")
    kernel: Optional[str] = None  # preset name for kind == "kernel"
    k: Optional[Tuple[float, float, float, float]] = None  # explicit override
    k_sw: Optional[float] = None
    factor: float = 1.0  # sigma_omega / gain / noise / drive multiplier
    compartment: str = "all"  # "all" | "central" | "peripheral" | "v1"


#: Intervention library from the IX.4 table.  Times are filled in by the experiments.
def intervention_presets(t_start: float, duration: float) -> Dict[str, Intervention]:
    t_end = t_start + duration
    inf = float("inf")
    return {
        "dmt": Intervention("dmt", "kernel", t_start, t_end, kernel="mexican_hat"),
        "5meo": Intervention("5meo", "kernel", t_start, t_end, kernel="5meo_plate"),
        "oxygen": Intervention("oxygen", "sigma_omega", t_start, inf, factor=0.25),
        "psychedelic": Intervention("psychedelic", "kernel", 0.0, inf, kernel="retuned"),
        "sumatriptan": Intervention(
            "sumatriptan", "gain", t_start, inf, factor=0.4, compartment="peripheral"
        ),
        "sham": Intervention("sham", "gain", t_start, t_end, factor=1.0),
    }


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


@dataclass
class MetricConfig:
    """IX.5.  The dissonance index is the part most likely to be got wrong, so all of
    its knobs are exposed rather than buried."""

    # --- cluster frequency extraction ---
    freq_estimator: str = "instantaneous"  # "instantaneous" | "spectral_peak"
    freq_min_hz: float = 0.5
    freq_max_hz: float = 20.0
    use_flatness_gate: bool = True  # down-weight clusters with flat (noise-like) spectra
    min_cluster_coherence: float = 0.0  # drop clusters below this |z_c|

    # --- dissonance model ---
    # "composite" = max/blend of the beating term and the incommensurability term.
    # Neither alone passes all three of IX.5's sanity checks; see selftest.
    roughness_model: str = "composite"  # "composite" | "sethares" | "harmonicity"
    combine: str = "max"  # "max" | "sum" | "roughness" | "inharmonicity"
    w_roughness: float = 0.5  # weight on the beating term when combine == "sum"

    # beating term (Plomp-Levelt / Sethares)
    n_partials: int = 6  # P, harmonics per cluster tone
    partial_rolloff: float = 0.88  # amplitude of partial n = a * rolloff^(n-1)
    band_mode: str = "relative"  # "relative" | "absolute" | "audio"
    band_width: float = 0.01  # fractional (relative) or Hz (absolute)
    reference_freq_hz: float = 5.0  # calibration carrier for the non-scale-free modes
    b1: float = 3.5
    b2: float = 5.75

    # incommensurability term (harmonic-template fit over the whole tone set)
    template_n_max: int = 16  # largest harmonic number a tone may be assigned
    template_grid: int = 2000  # f0 search resolution (log-spaced)
    ratio_tol: float = 0.02  # relative mistuning that still counts as "on the template"

    # pairwise small-integer-ratio model, IX.5 option (ii), available as an alternative
    max_ratio_order: int = 6  # Q in p/q, p,q <= Q
    ratio_tol_harmonic: float = 0.15  # weighted deviation that scores ~63% dissonant

    # --- spatial harmonic purity ---
    n_eigenmodes: int = 64  # K connectome harmonics to project onto
    purity_topk: int = 5
    # Include the small-world links in the harmonic basis.  Set this whenever K_SW != 0,
    # or the eigenmodes are those of a graph the oscillators are not coupling through.
    harmonics_include_sw: bool = False

    # --- topological defects ---
    # threshold on *local incoherence* (1 - |mean phase vector over a node and its
    # neighbours|), which lives in [0, 1]; 0.5 is roughly a pi/2 spread
    defect_threshold: float = 0.5

    # --- LZ / entropy (negative control) ---
    lz_max_nodes: int = 64
    lz_max_time: int = 512
    lz_normalise: str = "shuffle"  # "shuffle" | "theoretical" | "none"
    lz_seed: int = 1234


# --------------------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------------------


@dataclass
class RunConfig:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    kernel: KernelConfig = field(default_factory=KernelConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    interventions: List[Intervention] = field(default_factory=list)
    label: str = "run"

    # convenience
    def copy(self) -> "RunConfig":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# --------------------------------------------------------------------------------------
# Dotted-path overrides
# --------------------------------------------------------------------------------------


def _coerce(current: Any, value: Any) -> Any:
    """Best-effort cast of a CLI string onto the type of the existing field value."""
    if isinstance(value, str):
        text = value.strip()
        if isinstance(current, bool):
            return text.lower() in ("1", "true", "yes", "on")
        if isinstance(current, tuple):
            parts = [p for p in text.strip("()[] ").split(",") if p.strip()]
            return tuple(float(p) for p in parts)
        if isinstance(current, int) and not isinstance(current, bool):
            return int(float(text))
        if isinstance(current, float):
            return float(text)
        if current is None:
            for cast in (int, float):
                try:
                    return cast(text)
                except ValueError:
                    pass
        return text
    return value


def apply_overrides(cfg: RunConfig, overrides: Dict[str, Any]) -> RunConfig:
    """Apply `{"section.field": value}` overrides in place on a copy."""
    out = cfg.copy()
    for path, value in overrides.items():
        parts = path.split(".")
        target: Any = out
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"no config section {part!r} in {path!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            valid = [f.name for f in fields(target)] if is_dataclass(target) else []
            raise KeyError(f"no config field {leaf!r} in {path!r}; have {valid}")
        setattr(target, leaf, _coerce(getattr(target, leaf), value))
    return out


def config_from_json(path: str) -> RunConfig:
    with open(path) as fh:
        blob = json.load(fh)
    cfg = RunConfig()
    flat: Dict[str, Any] = {}

    def walk(prefix: str, node: Any) -> None:
        for key, val in node.items():
            if isinstance(val, dict):
                walk(f"{prefix}{key}.", val)
            else:
                flat[f"{prefix}{key}"] = val

    for section, val in blob.items():
        if section == "interventions":
            cfg.interventions = [Intervention(**iv) for iv in val]
        elif isinstance(val, dict):
            walk(f"{section}.", val)
        else:
            flat[section] = val
    return apply_overrides(cfg, flat)
