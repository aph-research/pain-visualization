"""Kuramoto-style dynamics with a shell coupling kernel, a synchronising drive and
timed interventions (Part IX.3 / IX.4).

    theta_i(t+dt) = theta_i(t) + [ omega_i + C_i(t) + D_i(t) ] dt + sigma_noise sqrt(dt) xi

    C_i(t) = g_i / N_i * [ sum_s K_s sum_{j in D_s} sin(theta_j - theta_i)
                           + K_SW(t) sum_{k in SW_i} sin(theta_k - theta_i) ]

The shell sums are evaluated as two sparse mat-vecs against a single precombined
matrix M = sum_s K_s A_s, using

    sum_j A_ij sin(theta_j - theta_i) = cos(theta_i) (M sin(theta))_i
                                        - sin(theta_i) (M cos(theta))_i

M is rebuilt only when an intervention changes the kernel, so a run costs two sparse
mat-vecs per step regardless of how many shells are populated.

Note on the noise term: the document writes noise inside the bracket multiplied by dt,
which would make its effect vanish as dt -> 0.  We integrate it as Euler-Maruyama
(sqrt(dt) scaling) so that `sigma_noise` means the same thing at every step size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .config import DriveConfig, Intervention, KERNEL_PRESETS, RunConfig
from .networks import Network, build_network

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------------------
# Kernel handling
# --------------------------------------------------------------------------------------


def kernel_energy(k: Sequence[float], shell_sizes: Sequence[float], mode: str) -> float:
    """Total coupling energy of a kernel.

    "Match total coupling energy across kernel conditions.  Otherwise you will be
    measuring gain, not shape, and every result will be confounded." (IX.4)

    `l1` matches sum |K_s|; `shell_weighted` also accounts for how many neighbours each
    shell actually contains, which is what the coupling term sees.
    """
    if mode == "none":
        return 1.0
    if mode == "l1":
        return float(np.sum(np.abs(k)))
    if mode == "shell_weighted":
        return float(np.sum(np.abs(np.asarray(k)) * np.asarray(shell_sizes[: len(k)])))
    raise KeyError(f"unknown energy_match mode {mode!r}")


def resolve_energy_match(mode: str, shell_normalisation: str) -> str:
    """Pick the energy measure that actually matches the way the shells are summed.

    `l1` only equals max |C_i| when each shell has been row-normalised; under the literal
    III.2 convention (one 1/N_i over the whole bracket) the shells have wildly different
    sizes and only `shell_weighted` matches.  Getting this pairing wrong silently
    reintroduces the exact confound IX.4 warns about - measuring gain instead of shape -
    so "auto" resolves it from `shell_normalisation` rather than leaving it to the caller.
    """
    if mode != "auto":
        return mode
    return "l1" if shell_normalisation == "per_shell" else "shell_weighted"


def normalise_kernel(
    k: Sequence[float],
    shell_sizes: Sequence[float],
    mode: str,
    coupling_strength: float,
    mean_neighbours: float,
) -> np.ndarray:
    """Rescale a kernel so its maximum achievable |C_i| equals `coupling_strength`.

    Because every kernel - baseline and intervention alike - passes through this, two
    conditions can differ only in the *shape* of (K1..K4).  That is IX.4's energy-matching
    requirement made structural rather than a thing to remember.
    """
    k = np.asarray(k, dtype=float)
    if mode == "none" or coupling_strength <= 0:
        return k
    e = kernel_energy(k, shell_sizes, mode)
    if e <= 1e-12:
        return k
    if mode == "shell_weighted":
        return k * (coupling_strength * mean_neighbours / e)
    return k * (coupling_strength / e)


def resolve_kernel(iv: Intervention) -> Tuple[np.ndarray, Optional[float]]:
    """(K1..K4, K_SW or None) for a kernel-type intervention."""
    if iv.k is not None:
        return np.asarray(iv.k, dtype=float), iv.k_sw
    if iv.kernel is None:
        raise ValueError(f"intervention {iv.name!r} is kind='kernel' but names no kernel")
    spec = KERNEL_PRESETS[iv.kernel]
    k_sw = iv.k_sw if iv.k_sw is not None else spec.get("k_sw")
    return np.asarray(spec["k"], dtype=float), k_sw


# --------------------------------------------------------------------------------------
# Drive envelope
# --------------------------------------------------------------------------------------


def drive_envelope(t: np.ndarray | float, cfg: DriveConfig, t_total: float) -> np.ndarray:
    """g(t) in [0, 1] - the circadian duty cycle of the hypothalamic drive."""
    t = np.asarray(t, dtype=float)
    wf = cfg.waveform
    if wf == "constant":
        return np.ones_like(t)
    if wf == "ramp":
        return np.clip(t / max(t_total, 1e-9), 0.0, 1.0)
    if wf == "ramp_updown":
        u = np.clip(t / max(t_total, 1e-9), 0.0, 1.0)
        return 1.0 - np.abs(2.0 * u - 1.0)

    period = max(cfg.period_s, 1e-9)
    phase = (t / period + cfg.phase_offset) % 1.0
    duty = float(np.clip(cfg.duty, 1e-6, 1.0))

    if wf == "raised_cosine":
        u = phase / duty
        g = np.where(phase < duty, 0.5 * (1.0 - np.cos(TWO_PI * np.clip(u, 0, 1))), 0.0)
        return g
    if wf == "square":
        # Cap the ramp at half the on-window.  Without this the rise and fall overlap for
        # short duty cycles and the trapezoid never reaches 1 - at the duty = 0.05 end of
        # the range IX.3 recommends, the default ramp would deliver half the requested
        # amplitude, and a duty sweep would read that back as a threshold effect.
        r = min(max(cfg.ramp_s, 0.0) / period, duty / 2.0)
        if r <= 0:
            return (phase < duty).astype(float)
        rise = phase / r
        fall = (duty - phase) / r
        g = np.clip(np.minimum(rise, fall), 0.0, 1.0)
        return np.where(phase < duty, g, 0.0)
    raise KeyError(f"unknown drive waveform {cfg.waveform!r}")


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------


@dataclass
class SimResult:
    times: np.ndarray  # (T,) recorded sample times, seconds
    theta: np.ndarray  # (T, N) phases, radians, float32
    omega: np.ndarray  # (N,) natural angular frequencies actually used at t=0
    drive_g: np.ndarray  # (T,) drive envelope at the recorded times
    k_sw_eff: np.ndarray  # (T,) effective small-world coupling (diffuse variant)
    net: Network
    cfg: RunConfig
    fs: float = 0.0  # sampling rate of the recording, Hz
    meta: Dict = field(default_factory=dict)

    def steady(self) -> Tuple[np.ndarray, np.ndarray]:
        """Times and phases after the discarded transient."""
        i0 = int(self.cfg.dynamics.transient_frac * self.theta.shape[0])
        return self.times[i0:], self.theta[i0:]

    def window(self, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
        m = (self.times >= t0) & (self.times < t1)
        return self.times[m], self.theta[m]


# --------------------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------------------


class Simulator:
    def __init__(self, cfg: RunConfig, net: Optional[Network] = None):
        self.cfg = cfg
        self.net = net if net is not None else build_network(cfg.network)
        self.shell_sizes = self.net.meta["shell_sizes"]
        self.mean_neighbours = float(self.net.neighbour_counts.mean())

    # -- coupling matrix ---------------------------------------------------------------

    def _build_M(self, k: Sequence[float]) -> sp.csr_matrix:
        n = self.net.n
        shells = (
            self.net.shells_rownorm
            if self.cfg.kernel.shell_normalisation == "per_shell"
            else self.net.shells
        )
        M = sp.csr_matrix((n, n), dtype=np.float32)
        for coeff, shell in zip(k, shells):
            if abs(coeff) < 1e-12 or shell.nnz == 0:
                continue
            M = M + (np.float32(coeff) * shell)
        return M.tocsr()

    # -- natural frequencies ------------------------------------------------------------

    def _draw_omega(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        dyn = self.cfg.dynamics
        central = self.net.mask("central")
        mu = np.where(
            central, TWO_PI * dyn.omega0_central_hz, TWO_PI * dyn.omega0_peripheral_hz
        )
        sigma = TWO_PI * dyn.sigma_omega_hz
        omega = mu + sigma * rng.standard_normal(self.net.n)
        return omega, mu

    # -- interventions -------------------------------------------------------------------

    def _active(self, t: float) -> List[Tuple[int, Intervention]]:
        """Active interventions, paired with their index.

        The index, not the name, identifies an intervention: `Intervention.name` defaults
        to "" and nothing stops two windows sharing a name, so a name-keyed change detector
        silently skips the boundary between two same-named windows.
        """
        return [(i, iv) for i, iv in enumerate(self.cfg.interventions)
                if iv.t_start <= t < iv.t_end]

    def _derive_state(self, active: Sequence[Tuple[int, Intervention]]) -> Dict:
        """Everything an intervention can change, resolved for the current window."""
        active = [iv for _, iv in active]
        cfg = self.cfg
        k = np.asarray(cfg.kernel.k, dtype=float)
        k_sw_kernel = cfg.kernel.k_sw
        sigma_factor = 1.0
        noise_factor = 1.0
        drive_factor = 1.0
        gain = np.ones(self.net.n)

        for iv in active:
            if iv.kind == "kernel":
                k, k_sw = resolve_kernel(iv)
                if k_sw is not None:
                    k_sw_kernel = k_sw
            elif iv.kind == "sigma_omega":
                sigma_factor *= iv.factor
            elif iv.kind == "noise":
                noise_factor *= iv.factor
            elif iv.kind == "drive":
                drive_factor *= iv.factor
            elif iv.kind == "gain":
                gain[self.net.mask(iv.compartment)] *= iv.factor
            else:
                raise KeyError(f"unknown intervention kind {iv.kind!r}")

        k_raw = k
        k = normalise_kernel(
            k,
            self.shell_sizes,
            resolve_energy_match(cfg.kernel.energy_match, cfg.kernel.shell_normalisation),
            cfg.kernel.coupling_strength,
            self.mean_neighbours,
        )
        k = k * cfg.kernel.gain
        if cfg.kernel.scale_k_sw:
            raw_norm = float(np.sum(np.abs(k_raw)))
            if raw_norm > 1e-12:
                k_sw_kernel = k_sw_kernel * float(np.sum(np.abs(k))) / raw_norm

        # The diffuse drive rides on top of the kernel's own small-world coupling; its
        # amplitude is a drive parameter and is deliberately not rescaled with the kernel.
        k_sw_base = k_sw_kernel
        if cfg.drive.variant == "diffuse":
            k_sw_base = k_sw_base + cfg.drive.k_sw_base

        return {
            "k": k,
            "M": self._build_M(k),
            "k_sw_base": k_sw_base,
            "sigma_factor": sigma_factor,
            "noise_factor": noise_factor,
            "drive_factor": drive_factor,
            "gain": gain,
        }

    # -- entrainment (Experiment 4) -------------------------------------------------------

    def _entrain_mask(self) -> Optional[np.ndarray]:
        d = self.cfg.drive
        if d.entrain_nodes == "none" or not d.entrain_freqs_hz or d.entrain_amplitude == 0:
            return None
        return self.net.mask(d.entrain_nodes)

    # -- integration ----------------------------------------------------------------------

    def run(self, progress: bool = False) -> SimResult:
        cfg = self.cfg
        dyn = cfg.dynamics
        net = self.net
        n = net.n
        dt = dyn.dt
        n_steps = int(round(dyn.t_total / dt))
        rng = np.random.default_rng(dyn.seed)

        omega_raw, omega_mu = self._draw_omega(rng)
        theta = rng.uniform(0, TWO_PI, size=n)
        theta_drive_0 = rng.uniform(0, TWO_PI)
        omega_drive = TWO_PI * cfg.drive.omega_drive_hz

        # per-shell normalisation is folded into the matrix, so no further division
        Ncounts = (
            np.ones(n) if cfg.kernel.shell_normalisation == "per_shell" else net.neighbour_counts
        )
        A_sw = net.sw_adjacency
        has_sw = A_sw.nnz > 0
        if cfg.kernel.sw_normalisation == "separate":
            sw_counts = np.maximum(np.asarray(A_sw.sum(axis=1)).ravel(), 1.0)
        else:
            # "shared" means III.2's 1/N_i, which is net.neighbour_counts - NOT the
            # `Ncounts` above, which per-shell normalisation has already set to ones.
            sw_counts = net.neighbour_counts

        drive_mask = None if cfg.drive.target == "all" else net.mask(cfg.drive.target)

        ent_mask = self._entrain_mask()
        ent_freqs = np.asarray(cfg.drive.entrain_freqs_hz, dtype=float) * TWO_PI
        ent_amp = cfg.drive.entrain_amplitude
        if ent_mask is not None and cfg.drive.entrain_match_energy and ent_freqs.size:
            ent_amp = ent_amp / ent_freqs.size
        ent_phase = rng.uniform(0, TWO_PI, size=ent_freqs.size) if ent_freqs.size else None

        # Intervention windows: recompute the derived state only when the active set
        # changes.  Boundary times are converted with ceil, not round, so the step the
        # check happens on always satisfies step*dt >= t_start - rounding down would enter
        # a window one step early (before it opens, so nothing happens) and, symmetrically,
        # leave a window one step early or never leave it at all.
        state = self._derive_state(self._active(0.0))
        active_key = tuple(i for i, _ in self._active(0.0))
        boundaries = (
            {0.0}
            | {iv.t_start for iv in cfg.interventions}
            | {iv.t_end for iv in cfg.interventions if np.isfinite(iv.t_end)}
        )
        boundary_steps = {int(np.ceil(b / dt - 1e-9)) for b in boundaries if b >= 0}

        omega = omega_mu + state["sigma_factor"] * (omega_raw - omega_mu)

        rec_every = max(1, dyn.record_every)
        n_rec = n_steps // rec_every + 1
        theta_rec = np.empty((n_rec, n), dtype=np.float32)
        t_rec = np.empty(n_rec)
        g_rec = np.empty(n_rec)
        ksw_rec = np.empty(n_rec)
        rec_i = 0

        sqrt_dt = np.sqrt(dt)
        variant = cfg.drive.variant
        use_rk4 = dyn.integrator == "rk4"

        def deriv(th: np.ndarray, t: float, st: Dict) -> np.ndarray:
            s, c = np.sin(th), np.cos(th)
            M = st["M"]
            coup = c * (M @ s) - s * (M @ c)
            g = float(drive_envelope(t, cfg.drive, dyn.t_total))
            k_sw_eff = st["k_sw_base"]
            if variant == "diffuse":
                k_sw_eff = k_sw_eff + st["drive_factor"] * cfg.drive.amplitude * g
            out = omega + st["gain"] * coup / Ncounts
            if has_sw and abs(k_sw_eff) > 1e-12:
                sw = c * (A_sw @ s) - s * (A_sw @ c)
                out = out + st["gain"] * k_sw_eff * sw / sw_counts
            if variant == "pacemaker":
                th_d = theta_drive_0 + omega_drive * t
                amp = st["drive_factor"] * cfg.drive.amplitude * g
                if drive_mask is None:
                    out = out + amp * np.sin(th_d - th)
                else:
                    out[drive_mask] = out[drive_mask] + amp * np.sin(th_d - th[drive_mask])
            if ent_mask is not None:
                # phase coupling of the stimulated nodes to each external tone
                target = th[ent_mask]
                acc = np.sin((ent_freqs * t + ent_phase)[:, None] - target[None, :]).sum(axis=0)
                out[ent_mask] = out[ent_mask] + ent_amp * acc
            return out

        for step in range(n_steps + 1):
            t = step * dt
            if step in boundary_steps:
                act = self._active(t)
                key = tuple(i for i, _ in act)
                if key != active_key or step == 0:
                    state = self._derive_state(act)
                    active_key = key
                    omega = omega_mu + state["sigma_factor"] * (omega_raw - omega_mu)

            if step % rec_every == 0 and rec_i < n_rec:
                theta_rec[rec_i] = theta.astype(np.float32)
                t_rec[rec_i] = t
                g = float(drive_envelope(t, cfg.drive, dyn.t_total))
                g_rec[rec_i] = g
                ksw_rec[rec_i] = (
                    state["k_sw_base"] + state["drive_factor"] * cfg.drive.amplitude * g
                    if variant == "diffuse"
                    else state["k_sw_base"]
                )
                rec_i += 1

            if step == n_steps:
                break

            if use_rk4:
                k1 = deriv(theta, t, state)
                k2 = deriv(theta + 0.5 * dt * k1, t + 0.5 * dt, state)
                k3 = deriv(theta + 0.5 * dt * k2, t + 0.5 * dt, state)
                k4 = deriv(theta + dt * k3, t + dt, state)
                theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                theta = theta + dt * deriv(theta, t, state)

            sigma_noise = dyn.sigma_noise * state["noise_factor"]
            if sigma_noise > 0:
                theta = theta + sigma_noise * sqrt_dt * rng.standard_normal(n)
            theta = np.mod(theta, TWO_PI)

        theta_rec = theta_rec[:rec_i]
        return SimResult(
            times=t_rec[:rec_i],
            theta=theta_rec,
            omega=omega,
            drive_g=g_rec[:rec_i],
            k_sw_eff=ksw_rec[:rec_i],
            net=net,
            cfg=cfg,
            fs=1.0 / (dt * rec_every),
            meta={
                "n": n,
                "geometry": net.geometry,
                "kernel": list(map(float, state["k"])),
                "mean_neighbours": self.mean_neighbours,
                "n_clusters": net.n_clusters,
            },
        )


def simulate(cfg: RunConfig, net: Optional[Network] = None) -> SimResult:
    return Simulator(cfg, net).run()
