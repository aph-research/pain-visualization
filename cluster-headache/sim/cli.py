"""Command-line front end.

    python -m sim.cli selftest                      # validate the dissonance metric first
    python -m sim.cli exp1                          # one experiment
    python -m sim.cli all                           # the whole programme
    python -m sim.cli run --set drive.amplitude=5   # a single simulation, any parameters
    python -m sim.cli sweep --over drive.amplitude --values 0,2,4,6,8

Every parameter in `sim/config.py` is reachable with `--set section.field=value`, so the
levers are not limited to the ones that have their own flag:

    python -m sim.cli exp1 --set kernel.coupling_strength=6 --set dynamics.sigma_omega_hz=0.1
    python -m sim.cli run --set network.branching_factor=2 --set network.segment_length=8
    python -m sim.cli run --kernel mexican_hat --set metrics.band_width=0.02
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np

from . import plots, selftest
from .config import (
    KERNEL_PRESETS,
    RunConfig,
    apply_overrides,
    config_from_json,
    kernel_from_preset,
)
from .dynamics import Simulator
from .experiments import EXPERIMENTS, cached_network, run_condition
from .metrics import analyse, dissonance_timeseries, scalars

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def _parse_sets(pairs: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects section.field=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _base_config(args) -> RunConfig:
    cfg = config_from_json(args.config) if args.config else RunConfig()
    overrides = _parse_sets(args.set)
    if getattr(args, "kernel", None):
        k = kernel_from_preset(args.kernel)
        overrides.setdefault("kernel.name", k.name)
        overrides.setdefault("kernel.k", k.k)
        overrides.setdefault("kernel.k_sw", k.k_sw)
    if getattr(args, "geometry", None):
        overrides.setdefault("network.geometry", args.geometry)
    if getattr(args, "n", None):
        overrides.setdefault("network.n_target", args.n)
    if getattr(args, "seed", None) is not None:
        overrides.setdefault("network.seed", args.seed)
        overrides.setdefault("dynamics.seed", args.seed)
    if getattr(args, "t_total", None):
        overrides.setdefault("dynamics.t_total", args.t_total)
    return apply_overrides(cfg, overrides)


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def cmd_selftest(args) -> int:
    result = selftest.run()
    print(selftest.report(result))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=2, default=_json_default)
        print(f"\nwrote {args.json}")
    return 0 if result["all_pass"] else 1


def cmd_run(args) -> int:
    cfg = _base_config(args)
    t0 = time.time()
    net = cached_network(cfg.network)
    res = Simulator(cfg, net).run()
    t, th = res.steady()
    stats = analyse(th, t, net, res.fs, cfg.metrics, with_complexity=not args.fast)
    elapsed = time.time() - t0

    print(f"geometry={net.geometry}  N={net.n}  clusters={net.n_clusters}  "
          f"steps={int(cfg.dynamics.t_total / cfg.dynamics.dt)}  {elapsed:.1f}s")
    print(f"kernel {cfg.kernel.name} = {tuple(round(x, 3) for x in res.meta['kernel'])}"
          f"  K_SW={cfg.kernel.k_sw}  drive A={cfg.drive.amplitude} "
          f"({cfg.drive.variant}, {cfg.drive.waveform})")
    print()
    keys = ["dissonance", "dissonance_intensive", "d_roughness", "d_inharmonic",
            "coherence_weight", "active_pair_fraction",
            "r_global_mean", "r_local_mean", "r_central_mean", "fragmentation",
            "r_between_mean",
            "harmonic_purity_topk", "harmonic_purity_absolute", "harmonic_negentropy",
            "harmonic_captured",
            "defect_density", "mean_incoherence",
            "bifurcation_concentration", "bifurcation_excess",
            "branch_mismatch_rad", "chain_mismatch_rad", "branch_over_chain_mismatch",
            "freq_spread_hz"]
    if not args.fast:
        keys += ["lz_normalised", "zlib_ratio", "spectral_entropy"]
    for k in keys:
        if k in stats:
            print(f"  {k:<28} {float(stats[k]):+.4f}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "config.json"), "w") as fh:
            fh.write(cfg.to_json())
        with open(os.path.join(args.out, "metrics.json"), "w") as fh:
            json.dump(scalars(stats), fh, indent=2)
        np.savez_compressed(os.path.join(args.out, "series.npz"),
                            times=res.times, theta=res.theta, omega=res.omega,
                            drive_g=res.drive_g)
        ts = dissonance_timeseries(res.theta, res.times, net, res.fs, cfg.metrics)
        fig, ax = plots.new_fig(figsize=(9.0, 3.6))
        ax.plot(ts["t"], ts["dissonance"], color=plots.SERIES[0], linewidth=2.0, label="dissonance D")
        ax.plot(ts["t"], ts["r_global"], color=plots.SERIES[2], linewidth=2.0, label="r_global")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("value")
        ax.legend(loc="best")
        ax.set_title(f"{net.geometry}, A={cfg.drive.amplitude:g}", loc="left", pad=10)
        plots.finish(fig, os.path.join(args.out, "timeseries.png"))
        print(f"\nwrote {args.out}/")
    return 0


def cmd_sweep(args) -> int:
    base = _base_config(args)
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    rows: List[Dict[str, object]] = []
    print(f"{args.over:>28}  {'D':>9} {'r_global':>9} {'frag':>8} {'purity':>8} "
          + (f"{'LZ':>8}" if not args.fast else ""))
    for v in values:
        for seed in range(args.seeds):
            _, stats = run_condition(base, {args.over: v,
                                            "network.seed": seed, "dynamics.seed": seed},
                                     with_complexity=not args.fast)
            rows.append({args.over: v, "seed": seed, **scalars(stats)})
        sub = [r for r in rows if r[args.over] == v]
        m = lambda k: float(np.nanmean([r[k] for r in sub]))
        line = (f"{v:>28}  {m('dissonance'):>9.5f} {m('r_global_mean'):>9.3f} "
                f"{m('fragmentation'):>8.3f} {m('harmonic_purity_topk'):>8.3f}")
        if not args.fast:
            line += f" {m('lz_normalised'):>8.3f}"
        print(line)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "sweep.json"), "w") as fh:
            json.dump(rows, fh, indent=2, default=_json_default)
        print(f"\nwrote {args.out}/sweep.json")
    return 0


def cmd_experiment(args, name: str) -> int:
    base = _base_config(args)
    root = args.root or DEFAULT_ROOT
    os.makedirs(root, exist_ok=True)
    fn = EXPERIMENTS[name]
    accepted = set(inspect.signature(fn).parameters)
    kwargs = {}
    if getattr(args, "n", None):
        kwargs["n_target"] = args.n
    if getattr(args, "seeds", None):
        kwargs["seeds"] = tuple(range(args.seeds))
    if getattr(args, "t_total", None):
        kwargs["t_total"] = args.t_total
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    t0 = time.time()
    print(f"running {name} ...", flush=True)
    summary = EXPERIMENTS[name](root, base, **kwargs)
    print(f"\n{summary['experiment']}   [{time.time() - t0:.0f}s]")
    print("-" * 78)
    print(summary["verdict"])
    print("-" * 78)
    for fig in summary.get("figures", []):
        print(f"  figure: {fig}")
    return 0


def cmd_all(args) -> int:
    print(selftest.report(selftest.run()))
    print("\n" + "=" * 78 + "\n")
    verdicts = {}
    for name in ("exp0", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6"):
        cmd_experiment(args, name)
        root = args.root or DEFAULT_ROOT
        path = os.path.join(root, [d for d in os.listdir(root) if d.startswith(name)][0],
                            "summary.json")
        with open(path) as fh:
            verdicts[name] = json.load(fh)["verdict"]
        print()
    print("=" * 78)
    print("SUMMARY OF VERDICTS")
    print("=" * 78)
    for name, v in verdicts.items():
        print(f"\n{name}:\n  {v}")
    root = args.root or DEFAULT_ROOT
    with open(os.path.join(root, "verdicts.json"), "w") as fh:
        json.dump(verdicts, fh, indent=2)
    return 0


# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sim.cli",
        description="Cluster headache as field dissonance - simulations from Part IX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, with_geom=True):
        sp.add_argument("--set", action="append", metavar="section.field=value",
                        help="override any config field; repeatable")
        sp.add_argument("--config", help="JSON config file to start from")
        sp.add_argument("--n", type=int, help="target node count")
        sp.add_argument("--seed", type=int, help="network + dynamics seed")
        sp.add_argument("--t-total", dest="t_total", type=float, help="simulated seconds")
        sp.add_argument("--kernel", choices=sorted(KERNEL_PRESETS), help="kernel preset")
        if with_geom:
            sp.add_argument("--geometry", choices=["tree", "lattice", "hierarchical"])
        return sp

    sp = sub.add_parser("selftest", help="validate the dissonance metric (IX.5 checks)")
    sp.add_argument("--json", help="also write the full result here")
    sp.set_defaults(func=cmd_selftest)

    sp = common(sub.add_parser("run", help="one simulation, print every metric"))
    sp.add_argument("--out", help="directory for config, metrics, time series and a figure")
    sp.add_argument("--fast", action="store_true", help="skip the LZ/entropy metrics")
    sp.set_defaults(func=cmd_run)

    sp = common(sub.add_parser("sweep", help="sweep any single parameter"))
    sp.add_argument("--over", required=True, metavar="section.field")
    sp.add_argument("--values", required=True, help="comma-separated")
    sp.add_argument("--seeds", type=int, default=1)
    sp.add_argument("--out")
    sp.add_argument("--fast", action="store_true")
    sp.set_defaults(func=cmd_sweep)

    for name, blurb in (
        ("exp0", "replication of the source's lattice results"),
        ("exp1", "geometry dependence - the crux"),
        ("exp2", "attack dynamics, hysteresis, spontaneous termination"),
        ("exp3", "DMT / 5-MeO / oxygen / sumatriptan / persistent kernel shift"),
        ("exp4", "consonant vs dissonant entrainment - the STV signature"),
        ("exp5", "noise control and the D/LZ dissociation"),
        ("exp6", "the IX.6 parameter sweeps not covered elsewhere"),
    ):
        sp = common(sub.add_parser(name, help=blurb))
        sp.add_argument("--root", help=f"output root (default {DEFAULT_ROOT})")
        sp.add_argument("--seeds", type=int, help="number of seeds")
        sp.set_defaults(func=lambda a, _n=name: cmd_experiment(a, _n))

    sp = common(sub.add_parser("all", help="selftest plus every experiment"))
    sp.add_argument("--root", help=f"output root (default {DEFAULT_ROOT})")
    sp.add_argument("--seeds", type=int, help="number of seeds")
    sp.set_defaults(func=cmd_all)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
