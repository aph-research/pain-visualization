"""Cluster headache as field dissonance - simulation package.

Implements the specification in Part IX of
`cluster-headache/cluster-headache-field-topology-theory.md`.

    from sim import RunConfig, simulate, analyse
    cfg = RunConfig()
    cfg.drive.amplitude = 4.0
    res = simulate(cfg)
    times, theta = res.steady()          # note the order: (times, theta)
    stats = analyse(theta, times, res.net, res.fs, cfg.metrics)
    print(stats["dissonance"], stats["r_global_mean"])
"""

from .config import (  # noqa: F401
    DISSONANT_SET_HZ,
    DriveConfig,
    DynamicsConfig,
    HARMONIC_SET_HZ,
    Intervention,
    KERNEL_PRESETS,
    KernelConfig,
    MetricConfig,
    NetworkConfig,
    RunConfig,
    apply_overrides,
    config_from_json,
    intervention_presets,
    kernel_from_preset,
)
from .dynamics import SimResult, Simulator, drive_envelope, simulate  # noqa: F401
from .metrics import (  # noqa: F401
    analyse,
    dissonance_from_tones,
    dissonance_index,
    dissonance_timeseries,
    order_parameters,
    scalars,
)
from .networks import Network, build_network, matched_networks  # noqa: F401

__all__ = [name for name in dir() if not name.startswith("_")]
