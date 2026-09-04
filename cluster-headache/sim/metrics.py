"""Metrics (Part IX.5).

Five families, and the relationships *between* them are the result:

  1. order parameters       - r_global, r_local per cluster, and the fragmentation gap
  2. dissonance index D     - structured incommensurability, NOT noise
  3. spatial harmonic purity- projection onto the graph Laplacian eigenmodes
  4. topological defects     - local phase incoherence, bifurcation vs chain nodes
  5. LZ / entropy            - the NEGATIVE control

IX.5 asks for an index that scores exact ratios near zero, the source's dissonant TMS set
high, and white noise low.  No single measure does all three, and the reason is
structural: the TMS set is *almost exactly harmonic* and dissonant only by **beating** (an
absolute-detuning phenomenon), while a "relatively prime" stack has no beating at all and
is dissonant only by **incommensurability** (a ratio phenomenon).  So the index is a
composite of two terms, both always reported:

    d_roughness    Plomp-Levelt/Sethares beating between cluster tones
    d_inharmonic   failure of the whole tone set to fit one harmonic template

Two further choices that matter, both of them consequences of what "low for noise" costs:

  * The index keeps the **amplitude scale** rather than normalising it away.  A weighted
    *mean* roughness scores white noise as maximally dissonant, because incoherent clusters
    still have arbitrary peak frequencies; weighting by cluster coherence is what makes
    noise collapse.  `roughness_shape` reports the un-weighted version alongside.
  * `dissonance` is **extensive** - it divides by every cluster pair, including ones the
    coherence gate silenced - so it answers "how much of the system is locked into
    structured dissonance".  `dissonance_intensive` divides by the realised weight instead
    and answers "how dissonant is the part that is coherent".  An intervention can move the
    two in opposite directions, and when it does, that is the finding.

`python -m sim.cli selftest` exercises all of this, including the critical bandwidth, which
IX.5 leaves undetermined and which decides what counts as dissonant.
"""

from __future__ import annotations

import math
import zlib
from functools import lru_cache
from math import gcd
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .config import MetricConfig
from .networks import Network

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------------------
# 1. Order parameters
# --------------------------------------------------------------------------------------


def order_parameters(theta: np.ndarray, net: Network) -> Dict[str, np.ndarray | float]:
    """r_global over all nodes; r_local over each cluster and the central compartment.

    Model B's signature is *high or moderate r_global with fragmented, mutually
    incoherent local clusters* - i.e. `fragmentation` well above zero while r_global
    stays up.  Model A predicts low r_global instead.
    """
    z = np.exp(1j * theta)  # (T, N)
    r_global = np.abs(z.mean(axis=1))

    clusters = net.clusters
    n_c = int(clusters.max()) + 1
    z_cluster = np.empty((theta.shape[0], n_c), dtype=complex)
    for c in range(n_c):
        m = clusters == c
        z_cluster[:, c] = z[:, m].mean(axis=1) if m.any() else 0.0
    r_local = np.abs(z_cluster)

    central = net.mask("central")
    r_central = np.abs(z[:, central].mean(axis=1)) if central.any() else np.zeros_like(r_global)

    # coherence *between* clusters, with each cluster's amplitude divided out
    unit = np.divide(z_cluster, np.maximum(np.abs(z_cluster), 1e-12))
    r_between = np.abs(unit.mean(axis=1))

    return {
        "r_global": r_global,
        "r_local": r_local,
        "z_cluster": z_cluster,
        "r_central": r_central,
        "r_between": r_between,
        "r_global_mean": float(r_global.mean()),
        "r_local_mean": float(r_local.mean()),
        "r_central_mean": float(r_central.mean()),
        "r_between_mean": float(r_between.mean()),
        "fragmentation": float(r_local.mean() - r_global.mean()),
    }


# --------------------------------------------------------------------------------------
# 2. Dissonance index
# --------------------------------------------------------------------------------------


def _spectral_flatness(power: np.ndarray) -> float:
    p = np.maximum(power, 1e-30)
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def cluster_tones(
    z_cluster: np.ndarray, fs: float, cfg: MetricConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce each coherent cluster to a (frequency, amplitude) tone.

    Returns (freqs_hz, amps, flatness).  `amps` is the cluster's mean coherence |z_c|,
    optionally gated by spectral peakiness - a cluster whose spectrum is flat has no
    well-defined tone and should not contribute structured dissonance.
    """
    n_t, n_c = z_cluster.shape
    freqs = np.zeros(n_c)
    amps = np.zeros(n_c)
    flat = np.ones(n_c)

    # Nyquist guard.  The instantaneous estimator wraps phase increments into (-pi, pi],
    # so a cluster running above fs/2 aliases - often to a negative frequency - and the
    # band test below would then silently zero its amplitude, giving D = 0 for every
    # condition with no error.  That failure mode looks exactly like "dissonance is
    # geometry-independent", so it is refused rather than tolerated.
    nyquist = 0.5 * fs
    if cfg.freq_max_hz > nyquist:
        raise ValueError(
            f"metrics.freq_max_hz = {cfg.freq_max_hz} Hz exceeds the Nyquist frequency "
            f"{nyquist:g} Hz of the recorded series (fs = {fs:g} Hz, set by dynamics.dt "
            f"and dynamics.record_every). Cluster tones above Nyquist alias and would be "
            f"silently dropped. Lower freq_max_hz or lower record_every."
        )

    fft_freqs = np.fft.fftfreq(n_t, d=1.0 / fs)
    band = (fft_freqs >= cfg.freq_min_hz) & (fft_freqs <= cfg.freq_max_hz)

    for c in range(n_c):
        sig = z_cluster[:, c]
        coherence = float(np.abs(sig).mean())

        if band.any():
            spec = np.fft.fft(sig)
            power = np.abs(spec[band]) ** 2
            flat[c] = _spectral_flatness(power)
            f_peak = float(fft_freqs[band][int(np.argmax(power))])
        else:
            f_peak = 0.0

        if cfg.freq_estimator == "instantaneous":
            phase = np.angle(sig)
            dphi = np.angle(np.exp(1j * np.diff(phase)))  # wrapped increments
            weights = np.abs(sig[1:])
            if weights.sum() > 1e-12:
                f_inst = float(np.average(dphi, weights=weights)) * fs / TWO_PI
            else:
                f_inst = 0.0
            freqs[c] = f_inst
        else:
            freqs[c] = f_peak

        amp = coherence
        if cfg.use_flatness_gate:
            amp *= max(0.0, 1.0 - flat[c])
        amps[c] = amp

    ok = (
        (freqs >= cfg.freq_min_hz)
        & (freqs <= cfg.freq_max_hz)
        & (amps >= cfg.min_cluster_coherence)
    )
    amps = np.where(ok, amps, 0.0)
    return freqs, amps, flat


def _sethares_gmax(b1: float, b2: float) -> float:
    ratio = b1 / b2
    e = 1.0 / (b2 - b1)
    return ratio ** (b1 * e) - ratio ** (b2 * e)


def _critical_band(f: np.ndarray, cfg: MetricConfig) -> np.ndarray:
    if cfg.band_mode == "relative":
        return np.maximum(cfg.band_width * f, 1e-9)
    if cfg.band_mode == "absolute":
        return np.full_like(np.asarray(f, dtype=float), max(cfg.band_width, 1e-9))
    if cfg.band_mode == "audio":  # Plomp-Levelt / Sethares, fitted to hearing
        return 0.0207 * np.asarray(f, dtype=float) + 18.96
    raise KeyError(f"unknown band_mode {cfg.band_mode!r}")


@lru_cache(maxsize=64)
def _sethares_reference(band_mode: str, band_width: float, n_partials: int,
                        rolloff: float, b1: float, b2: float, ref_hz: float) -> float:
    """The largest roughness two complex tones of this shape can reach.

    Dividing by it puts the index on a 0-1 scale where 1 is "as rough as this partial
    structure allows", instead of a scale set by however many far-apart partial pairs
    happen to be in the sum.  Computed once per metric configuration.
    """
    from .config import MetricConfig as _MC

    probe = _MC(band_mode=band_mode, band_width=band_width, n_partials=n_partials,
                partial_rolloff=rolloff, b1=b1, b2=b2)
    ratios = np.linspace(1.0, 2.0, 1500)
    best = max(_raw_sethares(ref_hz, ref_hz * r, probe) for r in ratios)
    return max(best, 1e-9)


def _raw_sethares(f_a: float, f_b: float, cfg: MetricConfig) -> float:
    """Unnormalised Sethares/Plomp-Levelt roughness between two complex tones."""
    if f_a <= 0 or f_b <= 0:
        return 0.0
    n = np.arange(1, cfg.n_partials + 1)
    amp = cfg.partial_rolloff ** (n - 1)
    fa = (n * f_a)[:, None]
    fb = (n * f_b)[None, :]

    d = np.abs(fa - fb)
    s = 0.24 / _critical_band(np.minimum(fa, fb), cfg)
    g = np.maximum(np.exp(-cfg.b1 * s * d) - np.exp(-cfg.b2 * s * d), 0.0)
    weights = amp[:, None] * amp[None, :]
    return float(np.sum(weights * g) / (_sethares_gmax(cfg.b1, cfg.b2) * amp.sum() ** 2))


def roughness_sethares(f_a: float, f_b: float, cfg: MetricConfig) -> float:
    """Roughness of two complex tones, on a 0-1 scale.

    Each tone gets `n_partials` harmonics with a geometric amplitude rolloff, and
    roughness is summed over all partial pairs.  Exact small-integer ratios put every
    near-coincident partial pair at zero detuning and score 0.

    **The critical bandwidth is the lever that decides what "dissonant" means**, and
    IX.5's own sanity checks do not fix it.  `relative` (band ~ band_width * f) is
    scale-free and is the right default here, since the model's frequencies live at
    1-20 Hz where the audio-fitted absolute bandwidth (~19 Hz) would call every pair
    consonant.  But `band_width` itself is not determined by the theory: the source's
    dissonant TMS set is mistuned by 0.3-1%, so it only scores as dissonant when
    band_width is around 0.01, while cluster-frequency splits in the simulation are a
    few percent and need band_width nearer 0.03.  `python -m sim.cli selftest` sweeps
    this explicitly rather than letting one choice pass silently.
    """
    ref = _sethares_reference(
        cfg.band_mode, cfg.band_width, cfg.n_partials, cfg.partial_rolloff,
        cfg.b1, cfg.b2, cfg.reference_freq_hz,
    )
    return min(1.0, _raw_sethares(f_a, f_b, cfg) / ref)


@lru_cache(maxsize=64)
def _small_ratios(p_max: int, q_max: int) -> Tuple[Tuple[int, int], ...]:
    """Coprime p/q >= 1 with p <= p_max, q <= q_max.

    p and q are bounded separately because a pair like 148:1 is a perfectly commensurate
    relationship - the 148th harmonic - and capping p at the same small Q as q would score
    every wide-interval harmonic stack as maximally dissonant.
    """
    return tuple(
        (p, q)
        for q in range(1, q_max + 1)
        for p in range(q, p_max + 1)
        if gcd(p, q) == 1
    )


def roughness_harmonicity(f_a: float, f_b: float, cfg: MetricConfig) -> float:
    """IX.5 option (ii): `min over small p,q of |f_a/f_b - p/q|` weighted by 1/(p+q).

    Read literally: the *deviation* is what 1/(p+q) weights, so a complex ratio has to sit
    proportionally closer to count as consonant.  Score = min over p,q <= Q of
    (p+q)/2 * |rho - p/q|, squashed into [0, 1].  An exact small-integer ratio scores 0, as
    IX.5's first sanity check requires.

    An earlier version applied the 1/(p+q) weight to the *fit* rather than the deviation,
    which capped the consonance credit for an exact octave at 2/3 and so scored a perfect
    1:2:3:4 stack as more dissonant than a mistuned one.

    This measure is scale-free and therefore blind to the absolute size of a mistuning: it
    cannot see the sub-percent detunings that make the source's TMS set beat.  That is the
    beating term's job, which is why the default model is the composite of the two.
    """
    if f_a <= 0 or f_b <= 0:
        return 1.0
    lo, hi = sorted((f_a, f_b))
    rho = hi / lo
    p_max = max(cfg.max_ratio_order, int(np.ceil(rho)) + 1)
    best = float("inf")
    for p, q in _small_ratios(p_max, cfg.max_ratio_order):
        best = min(best, 0.5 * (p + q) * abs(rho - p / q))
    if not math.isfinite(best):
        return 1.0
    return float(1.0 - math.exp(-best / max(cfg.ratio_tol_harmonic, 1e-9)))


def pairwise_roughness(f_a: float, f_b: float, cfg: MetricConfig) -> float:
    if cfg.roughness_model in ("sethares", "composite"):
        return roughness_sethares(f_a, f_b, cfg)
    if cfg.roughness_model == "harmonicity":
        return roughness_harmonicity(f_a, f_b, cfg)
    raise KeyError(f"unknown roughness_model {cfg.roughness_model!r}")


def harmonic_template_fit(
    freqs: Sequence[float], amps: Sequence[float], cfg: MetricConfig
) -> float:
    """How well the whole tone set fits a single harmonic series, in [0, 1].

    Searches a fundamental f0 and scores each tone by how close f_i/f0 sits to a small
    integer.  This is the "relatively prime" half of STV's antisymmetry: a set with no
    common fundamental at small integer multiples has a long common period, which is what
    "frequencies relatively prime to each other" means operationally.

    It is a genuinely different measurement from roughness.  Roughness is about absolute
    detuning inside a critical band (beating); this is about ratios (incommensurability).
    A stack at 1:phi:phi^2 is maximally incommensurate but has no beating; the source's
    dissonant TMS set beats audibly but is *almost exactly* harmonic.  Neither measure
    catches both, which is why `roughness_model="composite"` combines them.
    """
    f = np.asarray(freqs, dtype=float)
    a = np.asarray(amps, dtype=float)
    sel = (a > 0) & (f > 0)
    if sel.sum() < 2:
        return 1.0
    f, a = f[sel], a[sel]

    fmin, fmax = float(f.min()), float(f.max())
    n_max = max(cfg.template_n_max, int(np.ceil(fmax / fmin)))
    grid = np.exp(np.linspace(np.log(fmin / n_max), np.log(fmin), cfg.template_grid))

    def _score(f0_values: np.ndarray) -> np.ndarray:
        ratios = f[None, :] / f0_values[:, None]  # (G, K)
        n = np.clip(np.round(ratios), 1.0, n_max)
        # Residual in units of the fundamental, NOT relative to n.  A relative residual
        # makes the accepted window grow with harmonic number - at n = 16 a tolerance of
        # 0.02 would accept a deviation of 0.32, most of the way to the next harmonic, so
        # almost any tone set fits some template and the term becomes a tone counter.
        # |ratio - n| is also the physically meaningful quantity: it is the beat rate
        # between the tone and the template harmonic, in units of f0.
        resid = np.abs(ratios - n)
        fit = np.exp(-((resid / cfg.ratio_tol) ** 2))
        return (fit * a).sum(axis=1) / a.sum()

    coarse = _score(grid)
    best_i = int(np.argmax(coarse))
    # Refine around the best grid point: the coarse log grid resolves f0 to ~0.1%, which at
    # harmonic number 16 is already a residual comparable to ratio_tol, so an exactly
    # commensurate set can score as mildly inharmonic purely from grid spacing.
    step = grid[1] / grid[0] if grid.size > 1 else 1.001
    lo_f0 = grid[max(best_i - 1, 0)] / step
    hi_f0 = grid[min(best_i + 1, grid.size - 1)] * step
    fine = np.linspace(lo_f0, hi_f0, 400)
    return float(max(coarse[best_i], _score(fine).max()))


def dissonance_from_tones(
    freqs: Sequence[float], amps: Sequence[float], cfg: MetricConfig
) -> Dict[str, float]:
    """Amplitude-weighted pairwise roughness over the cluster tones.

    `dissonance`      - primary index; keeps the amplitude scale so noise scores low
    `roughness_shape` - amplitude-normalised mean roughness (shape only)
    `coherence_weight`- mean pairwise amplitude product
    """
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    n = freqs.size
    if n < 2:
        return {"dissonance": 0.0, "dissonance_intensive": 0.0, "roughness_shape": 0.0,
                "coherence_weight": 0.0, "d_roughness": 0.0, "d_inharmonic": 0.0,
                "roughness_intensive": 0.0, "inharmonicity": 0.0,
                "dissonance_total": 0.0, "n_tones": float(n), "active_pair_fraction": 0.0}

    total = 0.0
    weight = 0.0
    shape_num = 0.0
    n_pairs = 0
    for a in range(n):
        for b in range(a + 1, n):
            w = amps[a] * amps[b]
            r = pairwise_roughness(freqs[a], freqs[b], cfg)
            total += w * r
            weight += w
            shape_num += r
            n_pairs += 1

    d_roughness = total / n_pairs
    coherence_weight = weight / n_pairs
    inharmonicity = 1.0 - harmonic_template_fit(freqs, amps, cfg)
    d_inharmonic = inharmonicity * coherence_weight

    # Extensive vs intensive.  `d_roughness` divides by ALL cluster pairs, including the
    # ones the coherence gate silenced, so it answers "how much of the system is locked
    # into structured dissonance" - which is what makes noise score low.  The cost is that
    # it also falls when clusters merely go quiet: a state with 4 of 16 clusters coherent
    # and mutually incommensurate scores ~C(4,2)/C(16,2) = 1/20 of the same four tones
    # measured alone.  The intensive version divides by the realised weight instead and
    # answers "how dissonant is the part that IS coherent".  Both are reported, because an
    # intervention can move them in opposite directions and the difference is the finding.
    roughness_intensive = total / weight if weight > 1e-12 else 0.0
    d_intensive = max(roughness_intensive, inharmonicity) if cfg.roughness_model == "composite" \
        else roughness_intensive

    if cfg.roughness_model == "composite":
        if cfg.combine == "max":
            dissonance = max(d_roughness, d_inharmonic)
        elif cfg.combine == "sum":
            dissonance = cfg.w_roughness * d_roughness + (1 - cfg.w_roughness) * d_inharmonic
        elif cfg.combine == "roughness":
            dissonance = d_roughness
        elif cfg.combine == "inharmonicity":
            dissonance = d_inharmonic
        else:
            raise KeyError(f"unknown combine mode {cfg.combine!r}")
    else:
        dissonance = d_roughness

    return {
        "dissonance": float(dissonance),
        "dissonance_intensive": float(d_intensive),
        "d_roughness": float(d_roughness),
        "d_inharmonic": float(d_inharmonic),
        "roughness_intensive": float(roughness_intensive),
        "inharmonicity": float(inharmonicity),
        "dissonance_total": float(total),
        "roughness_shape": float(shape_num / n_pairs),
        "coherence_weight": float(coherence_weight),
        "n_tones": float(np.sum(amps > 0)),
        "active_pair_fraction": float(
            sum(1 for a in range(n) for b in range(a + 1, n) if amps[a] > 0 and amps[b] > 0)
            / n_pairs
        ),
    }


def dissonance_index(
    theta: np.ndarray, net: Network, fs: float, cfg: MetricConfig
) -> Dict[str, float]:
    orders = order_parameters(theta, net)
    freqs, amps, flat = cluster_tones(orders["z_cluster"], fs, cfg)
    out = dissonance_from_tones(freqs, amps, cfg)
    out["cluster_freqs"] = freqs
    out["cluster_amps"] = amps
    out["cluster_flatness"] = flat
    out["freq_spread_hz"] = float(np.std(freqs[amps > 0])) if np.any(amps > 0) else 0.0
    return out


def dissonance_timeseries(
    theta: np.ndarray,
    times: np.ndarray,
    net: Network,
    fs: float,
    cfg: MetricConfig,
    window_s: float = 4.0,
    step_s: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Sliding-window D(t), for the attack-cycle and intervention figures."""
    w = max(8, int(round(window_s * fs)))
    step = max(1, int(round(step_s * fs)))
    centres, dvals, rg, frag = [], [], [], []
    for start in range(0, max(1, theta.shape[0] - w + 1), step):
        seg = theta[start:start + w]
        orders = order_parameters(seg, net)
        freqs, amps, _ = cluster_tones(orders["z_cluster"], fs, cfg)
        d = dissonance_from_tones(freqs, amps, cfg)
        centres.append(times[start + w // 2])
        dvals.append(d["dissonance"])
        rg.append(orders["r_global_mean"])
        frag.append(orders["fragmentation"])
    return {
        "t": np.asarray(centres),
        "dissonance": np.asarray(dvals),
        "r_global": np.asarray(rg),
        "fragmentation": np.asarray(frag),
    }


# --------------------------------------------------------------------------------------
# 3. Spatial harmonic purity (connectome harmonics)
# --------------------------------------------------------------------------------------


def graph_harmonics(net: Network, k: int, include_sw: bool = False) -> np.ndarray:
    """K lowest eigenvectors of the symmetric normalised Laplacian, cached per network.

    `include_sw` adds the small-world links to the basis graph.  It should be set whenever
    K_SW is non-zero - the source's `checkerboard`, `pinwheel` and `traveling_wave` kernels
    all carry K_SW = -2.4, and the diffuse drive variant puts the entire drive on those
    links - because otherwise the "connectome harmonics" are the eigenmodes of a graph the
    oscillators are not running on.
    """
    cache_key = f"harmonics_{k}_{int(include_sw)}"
    if cache_key in net.meta:
        return net.meta[cache_key]
    eig_key = f"eigenvalues_{k}_{int(include_sw)}"

    A = net.adjacency.astype(np.float64)
    if include_sw and net.sw_adjacency is not None and net.sw_adjacency.nnz:
        A = (A + net.sw_adjacency.astype(np.float64)).tocsr()
        A.data[:] = np.minimum(A.data, 1.0)
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = sp.diags(dinv)
    L = sp.identity(net.n) - D @ A @ D

    k = min(k, net.n - 2)
    if net.n <= 2500:
        vals, vecs = np.linalg.eigh(L.toarray())
        U, lam = vecs[:, :k], vals[:k]
    else:
        vals, vecs = eigsh(L.tocsc(), k=k, sigma=-1e-3, which="LM")
        order = np.argsort(vals)
        U, lam = vecs[:, order], vals[order]
    net.meta[cache_key] = U
    net.meta[eig_key] = lam
    return U


def laplacian_eigenfrequencies(net: Network, coupling_strength: float,
                               k: int = 64, n_report: int = 6) -> np.ndarray:
    """The graph's own mode frequencies, in Hz, at a given coupling strength.

    IX.6 asks for the drive frequency to be swept "relative to the tree's dominant Laplacian
    eigenfrequencies".  A normalised-Laplacian eigenvalue is dimensionless; a mode with
    eigenvalue lambda evolves at K*lambda rad/s under coupling of strength K, so the natural
    frequency scale is K*lambda / 2pi.  The zero mode is dropped - it is the uniform state,
    which carries no spatial structure.

    The values are reported across the computed spectrum rather than as the lowest few.  On a
    tree of this size the smallest non-zero eigenvalues are ~1e-3, which map to well under
    0.01 Hz: the slow global modes sit orders of magnitude below the drive band and cannot
    constrain Omega, so quoting them would suggest a comparison that is not being made.
    """
    graph_harmonics(net, k)
    lam = net.meta.get(f"eigenvalues_{min(k, net.n - 2)}_0")
    if lam is None:
        return np.array([])
    lam = np.asarray(lam)
    lam = lam[lam > 1e-8]
    if lam.size == 0:
        return np.array([])
    idx = np.unique(np.linspace(0, lam.size - 1, n_report).astype(int))
    return coupling_strength * lam[idx] / (2 * np.pi)


def harmonic_purity(theta: np.ndarray, net: Network, cfg: MetricConfig) -> Dict[str, float]:
    """Project the complex field onto the graph harmonics.

    Consonant states should concentrate power in a few modes; dissonant states spread it
    across incommensurate ones.  This is the graph-native version of the source's DCT
    readout (which is used directly on the lattice, see `dct_modes`).
    """
    U = graph_harmonics(net, cfg.n_eigenmodes, cfg.harmonics_include_sw)
    z = np.exp(1j * theta)  # (T, N)
    coeffs = z @ U  # (T, K)
    power = (np.abs(coeffs) ** 2).mean(axis=0)
    total_field = (np.abs(z) ** 2).sum(axis=1).mean()
    p = power / max(power.sum(), 1e-30)

    k_top = min(cfg.purity_topk, p.size)
    purity = float(np.sort(p)[::-1][:k_top].sum())
    # `purity` is a fraction of the power *inside the retained modes*.  A state can be
    # concentrated in the top few of 64 harmonics while those 64 explain almost none of the
    # field, so the fraction of the whole field is reported alongside it.
    purity_absolute = float(np.sort(power)[::-1][:k_top].sum() / max(total_field, 1e-30))
    entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-30))))
    negentropy = float(1.0 - entropy / math.log(p.size)) if p.size > 1 else 1.0

    return {
        "harmonic_purity_topk": purity,
        "harmonic_purity_absolute": purity_absolute,
        "harmonic_negentropy": negentropy,
        "harmonic_entropy": entropy,
        "harmonic_captured": float(power.sum() / max(total_field, 1e-30)),
        "harmonic_power": p,
    }


def dct_modes(theta: np.ndarray, grid_shape: Tuple[int, int], n_frames: int = 60) -> np.ndarray:
    """2D DCT power of a lattice phase field - the source's readout for identifying
    checkerboard / traveling-wave / pinwheel modes.

    Accepts a single frame or a (T, N) series; a series is averaged over time, which is
    what makes a standing mode separable from a single noisy snapshot.
    """
    from scipy.fft import dctn

    theta = np.atleast_2d(theta)
    step = max(1, theta.shape[0] // n_frames)
    frames = theta[::step]
    power = np.zeros(grid_shape)
    for frame in frames:
        power += np.abs(dctn(np.cos(frame).reshape(grid_shape), norm="ortho")) ** 2
    return power / len(frames)


# --------------------------------------------------------------------------------------
# 4. Topological defects
# --------------------------------------------------------------------------------------


def local_incoherence(theta_frame: np.ndarray, net: Network) -> np.ndarray:
    """1 - |mean phase vector over {i} u neighbours(i)|, per node.

    0 = the node and its neighbours are phase-locked; 1 = maximal local disagreement.
    """
    z = np.exp(1j * theta_frame)
    A = net.adjacency
    neigh_sum = A @ z
    counts = np.asarray(A.sum(axis=1)).ravel()
    total = (neigh_sum + z) / np.maximum(counts + 1.0, 1.0)
    return 1.0 - np.abs(total)


def defect_statistics(theta: np.ndarray, net: Network, cfg: MetricConfig) -> Dict[str, float]:
    """Defect density, and where the defects sit.

    Model B's headline claim is that phase defects *concentrate at bifurcation points*
    under drive.  `bifurcation_concentration` is that claim as one number: mean local
    incoherence at branch points divided by mean local incoherence along chain segments.
    A ratio near 1 means the tree's branch points are doing nothing special.
    """
    frames = theta[:: max(1, len(theta) // 50)]
    inc = np.stack([local_incoherence(frame, net) for frame in frames])
    mean_inc = inc.mean(axis=0)

    bif = net.bifurcation_nodes
    chain = net.chain_nodes
    bif_mean = float(mean_inc[bif].mean()) if bif.size else float("nan")
    chain_mean = float(mean_inc[chain].mean()) if chain.size else float("nan")
    ratio = bif_mean / chain_mean if chain.size and chain_mean > 1e-9 else float("nan")

    # Degree-matched null.  A branch point has more neighbours than a chain node, so it
    # shows more local phase disagreement even in a field with no spatial structure at
    # all.  Permuting the phases across nodes destroys the structure while preserving
    # both the phase distribution and the graph, so `bifurcation_excess` is the part of
    # the concentration that is not explained by degree.
    rng = np.random.default_rng(cfg.lz_seed)
    null_inc = np.stack(
        [local_incoherence(frame[rng.permutation(net.n)], net) for frame in frames]
    ).mean(axis=0)
    null_bif = float(null_inc[bif].mean()) if bif.size else float("nan")
    null_chain = float(null_inc[chain].mean()) if chain.size else float("nan")
    null_ratio = null_bif / null_chain if chain.size and null_chain > 1e-9 else float("nan")

    out = {
        # Threshold per frame and then average.  Thresholding the time-averaged map first
        # smooths transient defects away and reports zero density in states that are full
        # of them.
        "defect_density": float((inc > cfg.defect_threshold).mean()),
        "mean_incoherence": float(mean_inc.mean()),
        "incoherence_bifurcation": bif_mean,
        "incoherence_chain": chain_mean,
        "bifurcation_concentration": ratio,
        "bifurcation_concentration_null": null_ratio,
        "bifurcation_excess": float(ratio / null_ratio) if null_ratio and null_ratio > 1e-9 else float("nan"),
        "incoherence_map": mean_inc,
    }

    if net.grid_shape is not None and net.geometry == "lattice":
        out["vortex_count"] = float(
            np.mean([count_vortices(f, net.grid_shape) for f in theta[:: max(1, len(theta) // 20)]])
        )

    if net.geometry == "tree":
        by_depth: Dict[int, float] = {}
        for d in np.unique(net.depth):
            m = net.depth == d
            if m.sum() >= 5:
                by_depth[int(d)] = float(mean_inc[m].mean())
        out["incoherence_by_depth"] = by_depth
        out.update(branch_point_mismatch(frames, net))
    return out


def branch_point_mismatch(frames: np.ndarray, net: Network) -> Dict[str, float]:
    """IX.5's own tree defect statistic: "for each bifurcation, the phase mismatch between
    the parent segment and the mean of the daughter segments".

    This is a different quantity from the neighbourhood incoherence above - it is signed by
    the tree's direction of travel, so it measures the phase step the drive has to cross
    going *outward* through a branch point, rather than local disagreement in general.
    The control is the same statistic at degree-2 chain nodes, where there is exactly one
    daughter and no branching.
    """
    parent = net.parent
    children: Dict[int, List[int]] = {}
    for node in range(net.n):
        p = int(parent[node])
        if p >= 0:
            children.setdefault(p, []).append(node)

    def _mismatch(nodes: Sequence[int]) -> float:
        vals = []
        for node in nodes:
            kids = children.get(int(node), [])
            if not kids:
                continue
            z_parent = np.exp(1j * frames[:, node])
            z_kids = np.exp(1j * frames[:, kids]).mean(axis=1)
            # circular distance between the parent phase and the daughter mean phase
            vals.append(np.abs(np.angle(z_kids * np.conj(z_parent))).mean())
        return float(np.mean(vals)) if vals else float("nan")

    bif = _mismatch(net.bifurcation_nodes)
    chain = _mismatch(net.chain_nodes)
    return {
        "branch_mismatch_rad": bif,
        "chain_mismatch_rad": chain,
        "branch_over_chain_mismatch": bif / chain if chain and chain > 1e-9 else float("nan"),
    }


def traveling_wave_index(theta: np.ndarray, grid_shape: Tuple[int, int],
                         fs: float = 50.0) -> Dict[str, float]:
    """Separate a traveling pattern from a standing one on a lattice.

    A DCT snapshot cannot tell them apart - both put power at the same |k| - which is why
    IX.7's "recover checkerboard, traveling-wave and pinwheel modes" needs a readout with
    time in it.

    Two readouts, because each has a blind spot:

    `spatial_asymmetry` - power at +k versus -k in the complex field.  A single plane wave
    puts everything at +k; a standing wave splits evenly.  Blind spot: two domains
    traveling in *opposite* directions cancel and look standing.

    `mode_drift_hz` - how fast the dominant spatial mode's own phase advances relative to
    the population mean phase.  A traveling pattern drifts through the lattice, so its
    Fourier coefficient rotates at a rate different from the bulk; a standing pattern is
    phase-locked to the bulk and does not drift.  This one survives counter-propagating
    domains, and is the primary index.
    """
    theta = np.atleast_2d(theta)
    n_t = theta.shape[0]
    ny, nx = grid_shape

    Z = np.fft.fft2(np.exp(1j * theta).reshape(n_t, ny, nx), axes=(1, 2))  # (T, ny, nx)
    power = (np.abs(Z) ** 2).mean(axis=0)
    p = power.copy()
    p[0, 0] = 0.0
    iy, ix = np.unravel_index(int(np.argmax(p)), p.shape)
    plus = float(p[iy, ix])
    minus = float(p[(-iy) % ny, (-ix) % nx])
    asym = abs(plus - minus) / max(plus + minus, 1e-30)

    def _rate(series: np.ndarray) -> float:
        """Mean rotation rate of a complex series, in Hz, amplitude weighted."""
        dphi = np.angle(np.exp(1j * np.diff(np.angle(series))))
        w = np.abs(series[1:])
        if w.sum() < 1e-12:
            return 0.0
        return float(np.average(dphi, weights=w) * fs / (2 * np.pi))

    bulk_rate = _rate(Z[:, 0, 0])                 # k = 0: the population mean phase
    mode_rate = _rate(Z[:, iy, ix])               # the dominant spatial mode
    return {
        "traveling_index": float(asym),
        "mode_drift_hz": float(abs(mode_rate - bulk_rate)),
        "bulk_rate_hz": bulk_rate,
        "mode_rate_hz": mode_rate,
        "peak_ky": int(iy if iy <= ny // 2 else iy - ny),
        "peak_kx": int(ix if ix <= nx // 2 else ix - nx),
        "peak_power_fraction": float(plus / max(p.sum(), 1e-30)),
    }


def count_vortices(theta_frame: np.ndarray, grid_shape: Tuple[int, int]) -> int:
    """Phase singularities on a 2D lattice: winding number around each unit plaquette."""
    g = theta_frame.reshape(grid_shape)
    a, b, c, d = g, np.roll(g, -1, axis=1), np.roll(np.roll(g, -1, axis=1), -1, axis=0), np.roll(g, -1, axis=0)

    def wrap(x):
        return np.angle(np.exp(1j * x))

    winding = wrap(b - a) + wrap(c - b) + wrap(d - c) + wrap(a - d)
    return int(np.sum(np.abs(winding) > np.pi))


# --------------------------------------------------------------------------------------
# 5. LZ complexity / entropy - the NEGATIVE control
# --------------------------------------------------------------------------------------


def lz76(seq: str) -> int:
    """Lempel-Ziv 1976 factorisation count.  Uses str.find so the inner scan runs in C."""
    n = len(seq)
    i = 0
    c = 0
    while i < n:
        length = 1
        while i + length <= n and seq.find(seq[i:i + length], 0, i + length - 1) != -1:
            length += 1
        c += 1
        i += length
    return c


def complexity_metrics(theta: np.ndarray, net: Network, cfg: MetricConfig) -> Dict[str, float]:
    """PCI-style binarisation, then LZ76 and zlib compression.

    This is the negative control from IX.5.  If the dissonance index and LZ complexity
    turn out to be collinear across conditions, we have built an entropy metric and the
    exercise is uninformative - so both are always reported together.
    """
    rng = np.random.default_rng(cfg.lz_seed)
    n_t, n_nodes = theta.shape

    nodes = np.linspace(0, n_nodes - 1, min(cfg.lz_max_nodes, n_nodes)).astype(int)
    tsel = np.linspace(0, n_t - 1, min(cfg.lz_max_time, n_t)).astype(int)
    x = np.sin(theta[np.ix_(tsel, nodes)])  # (T', M)
    bits = (x > np.median(x, axis=0, keepdims=True)).astype(np.uint8)

    flat = bits.T.reshape(-1)  # channel-major, PCI convention
    seq = "".join("1" if b else "0" for b in flat)
    c = lz76(seq)
    length = len(seq)
    lz_theoretical = c * math.log2(length) / length if length > 1 else 0.0

    shuffled = flat.copy()
    rng.shuffle(shuffled)
    c_shuf = lz76("".join("1" if b else "0" for b in shuffled))
    lz_norm = c / c_shuf if c_shuf > 0 else float("nan")

    packed = np.packbits(flat).tobytes()
    zratio = len(zlib.compress(packed, 9)) / max(len(packed), 1)

    # spectral entropy of the mean field, a second, independent entropy handle
    z = np.exp(1j * theta).mean(axis=1)
    spec = np.abs(np.fft.fft(z - z.mean())) ** 2
    p = spec / max(spec.sum(), 1e-30)
    spec_entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-30))) / math.log(max(p.size, 2)))

    return {
        "lz_raw": float(c),
        "lz_normalised": float(lz_norm),
        "lz_theoretical": float(lz_theoretical),
        "zlib_ratio": float(zratio),
        "spectral_entropy": spec_entropy,
    }


# --------------------------------------------------------------------------------------
# Top-level analysis
# --------------------------------------------------------------------------------------


def analyse(
    theta: np.ndarray,
    times: np.ndarray,
    net: Network,
    fs: float,
    cfg: MetricConfig,
    with_complexity: bool = True,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    orders = order_parameters(theta, net)
    out.update({k: v for k, v in orders.items() if isinstance(v, float)})
    out["_orders"] = orders

    freqs, amps, flat = cluster_tones(orders["z_cluster"], fs, cfg)
    diss = dissonance_from_tones(freqs, amps, cfg)
    out.update({k: v for k, v in diss.items() if isinstance(v, float)})
    out["cluster_freqs"] = freqs
    out["cluster_amps"] = amps
    out["cluster_flatness"] = flat
    out["freq_spread_hz"] = float(np.std(freqs[amps > 0])) if np.any(amps > 0) else 0.0

    out.update(harmonic_purity(theta, net, cfg))
    out.update(defect_statistics(theta, net, cfg))
    if with_complexity:
        out.update(complexity_metrics(theta, net, cfg))
    return out


def scalars(result: Dict[str, object]) -> Dict[str, float]:
    """Strip the array-valued entries, for tables and JSON."""
    return {k: float(v) for k, v in result.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
