# Cluster headache as field dissonance — simulations

An implementation of the simulation specification in Part IX of
[cluster-headache-field-topology-theory.md](cluster-headache-field-topology-theory.md).

The goal, in the document's own words, is *not* to confirm the model. It is to find out
whether a well-specified dissonance metric behaves as Model B claims under a set of network
geometries, coupling kernels and interventions — and, importantly, whether it fails to.

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy matplotlib
cd cluster-headache

../.venv/bin/python -m sim.cli selftest     # validate the metric FIRST
../.venv/bin/python -m sim.cli exp1         # the crux experiment
../.venv/bin/python -m sim.cli all          # selftest + all seven experiments (~25 min)
../.venv/bin/python make_report.py          # single-page findings report
```

Results land in `results/<experiment>/` as `summary.json` (including a plain-language
`verdict`), `records.csv`, `arrays.npz` and PNG figures.

---

## Layout

| File | What it holds |
|---|---|
| `sim/config.py` | Every parameter and lever, as dataclasses. Nothing is hard-coded elsewhere. |
| `sim/networks.py` | The three geometries, geodesic distance shells, small-world links, cluster partition. |
| `sim/dynamics.py` | The integrator: shell kernel, drive, timed interventions. |
| `sim/metrics.py` | Order parameters, dissonance index, harmonic purity, defects, LZ. |
| `sim/selftest.py` | The IX.5 sanity checks. Run this before believing anything else. |
| `sim/experiments.py` | Experiments 0–5 of IX.7, plus Experiment 6 for the IX.6 sweeps the others don't reach. |
| `sim/plots.py` | Figures. |
| `sim/cli.py` | Command line. |
| `make_report.py` | Builds a single-file HTML report from `results/`, reading every number from the run's own `summary.json`. |

Each experiment writes a plain-language `verdict` into its `summary.json` saying what the run
supports or refutes, in the theory document's own terms. Those verdicts are computed from the
numbers, not written by hand, so they cannot drift from the data.

---

## Levers

Every field of every config dataclass is reachable from the command line with
`--set section.field=value`, repeatable, on any subcommand:

```bash
../.venv/bin/python -m sim.cli run  --set drive.amplitude=5 --set dynamics.sigma_omega_hz=0.1
../.venv/bin/python -m sim.cli exp1 --set kernel.coupling_strength=6 --set network.branching_factor=2
../.venv/bin/python -m sim.cli sweep --over drive.amplitude --values 0,2,4,6,8 --seeds 3
../.venv/bin/python -m sim.cli sweep --over metrics.band_width --values 0.005,0.01,0.03
```

The parameters IX.6 asks to sweep, and where they live:

### Network — `network.*`

| Lever | Default | Meaning |
|---|---|---|
| `geometry` | `tree` | `tree` \| `lattice` \| `hierarchical` |
| `n_target` | 600 | Node count. Use a perfect square so the lattice matches exactly. |
| `branching_factor` | 3 | *b* |
| `segment_length` | 4 | *m*, nodes between bifurcations. Keeps branch points more than one hop apart, which IX.2 specifically requires. |
| `max_depth` | 12 | *L* cap |
| `division_weights` | (0.6, 0.2, 0.2) | V1 / V2 / V3. Cluster pain is overwhelmingly V1. |
| `v1_only` | False | First-pass simplification offered by IX.2 |
| `heterogeneous` | True | Log-normal segment lengths and Poisson branching. Gives *mild* heterogeneity, not a heavy tail — see the caveats. |
| `central_fraction` | 0.05 | Size of the trigeminocervical/thalamic relay |
| `n_shells`, `shell_cap` | 4, 8 | D₁=1 hop, D₂=2, D₃=3, D₄=4..cap |
| `sw_links_per_node` | 2 | Small-world wiring, per III.2 |
| `target_clusters` | 16 | Cluster count for r_local and the dissonance index |

### Kernel — `kernel.*`

| Lever | Default | Meaning |
|---|---|---|
| `k` | (0.6, 0.2, −0.1, −0.2) | The kernel vector (K₁…K₄) |
| `k_sw` | 0.0 | Small-world coupling |
| `coupling_strength` | 4.0 | Target max \|C_i\| in rad/s. **All kernels are rescaled to this**, so two conditions can differ only in *shape* — IX.4's energy-matching requirement made structural rather than something to remember. |
| `shell_normalisation` | `per_shell` | See the caveat below. `global` reproduces III.2's literal formula. |
| `energy_match` | `auto` | `auto` picks the energy measure that matches `shell_normalisation`. The wrong pairing silently breaks IX.4 energy matching, so it is resolved rather than left to the caller. Also `l1` \| `shell_weighted` \| `none`. |
| `sw_normalisation` | `separate` | Divide the small-world term by its own degree so K_SW is in rad/s. `shared` is III.2's literal 1/N_i. |
| `scale_k_sw` | True | Rescale a preset's K_SW by the same factor its shell constants get |

Kernel presets (`--kernel <name>`): `flat`, `mexican_hat`, `dmt_plate`, `dmt_plate_variant`,
`5meo_plate`, `dmt_branching`, `checkerboard`, `pinwheel`, `traveling_wave`, `baseline`,
`retuned`.

### Drive — `drive.*`

| Lever | Default | Meaning |
|---|---|---|
| `variant` | `pacemaker` | `pacemaker` (IX.3 Variant 2) \| `diffuse` (Variant 1, folded into K_SW) \| `none` |
| `amplitude` | 0.0 | *A* |
| `omega_drive_hz` | 5.0 | Ω. Sweeping Ω against the network's own modes is a dissonance generator in its own right. |
| `target` | `all` | Where the pacemaker injects. `central` sends it through the relay instead — see the finding below. |
| `waveform` | `constant` | `constant` \| `square` \| `raised_cosine` \| `ramp` \| `ramp_updown` |
| `period_s`, `duty`, `ramp_s` | 20, 0.2, 1.0 | Circadian duty cycle. IX.3 suggests d ≈ 0.05–0.3. |
| `entrain_nodes`, `entrain_freqs_hz`, `entrain_amplitude` | — | Focal entrainment for Experiment 4 |

### Dynamics — `dynamics.*`

| Lever | Default | Meaning |
|---|---|---|
| `sigma_omega_hz` | 0.25 | **The oxygen handle.** Variance of the natural-frequency distribution. |
| `omega0_central_hz` | 5.0 | Thalamic/trigeminocervical burst band |
| `omega0_peripheral_hz` | 5.6 | Heterogeneous ω₀ by compartment |
| `sigma_noise` | 0.05 | Noise amplitude, for the asymmetry control |
| `dt`, `t_total`, `transient_frac` | 0.002, 40, 0.25 | Integration |
| `integrator` | `euler` | `euler` \| `rk4` |
| `seed` | 0 | ω draw, initial phases, noise. Network wiring has its own seed. |

### Metrics — `metrics.*`

| Lever | Default | Meaning |
|---|---|---|
| `roughness_model` | `composite` | `composite` \| `sethares` \| `harmonicity` |
| `combine` | `max` | How the two dissonance components combine |
| `band_width` | 0.01 | Critical bandwidth. **The one lever the theory does not pin down** — see below. |
| `band_mode` | `relative` | `relative` (band ∝ f) \| `absolute` (Hz) \| `audio` (Plomp-Levelt hearing fit) |
| `template_n_max`, `ratio_tol` | 16, 0.02 | Incommensurability term |
| `n_eigenmodes`, `purity_topk` | 64, 5 | Connectome harmonics |
| `harmonics_include_sw` | False | Put the small-world links in the harmonic basis. Set it whenever `K_SW ≠ 0`, or the eigenmodes are those of a graph the oscillators are not coupling through. |
| `freq_estimator` | `instantaneous` | `instantaneous` \| `spectral_peak`. Guarded against Nyquist: a `freq_max_hz` above half the recording rate raises rather than silently aliasing every cluster tone to zero. |
| `defect_threshold` | 0.5 | Local incoherence, in [0, 1], above which a node counts as a defect. Thresholded per frame, not on the time-average. |

### Interventions

Programmatically, or via the presets used by Experiment 3:

```python
from sim import RunConfig, Intervention, simulate

cfg = RunConfig()
cfg.drive.amplitude = 5.0
cfg.interventions = [
    Intervention("dmt",    "kernel",      t_start=25, t_end=37, kernel="mexican_hat"),
    Intervention("oxygen", "sigma_omega", t_start=25, t_end=float("inf"), factor=0.25),
    Intervention("suma",   "gain",        t_start=25, t_end=float("inf"), factor=0.4,
                 compartment="peripheral"),
]
res = simulate(cfg)
```

`kind` is one of `kernel`, `sigma_omega`, `gain`, `noise`, `drive`; `compartment` is
`all`, `central`, `peripheral`, `v1`, `v2`, `v3`.

---

## Four implementation decisions that are not in the spec

These change results, so they are levers with documented defaults rather than silent choices.

**1. Per-shell normalisation (`kernel.shell_normalisation`).** III.2 puts a single `1/Nᵢ` in
front of the whole coupling bracket. But D₄ (4..8 hops) holds an order of magnitude more
neighbours than D₁ — on a 625-node lattice, roughly 120 versus 4 — so under the literal formula
the K₄ term dominates and the kernel *shape* the theory cares about is swamped by a shell-size
artefact. With a baseline kernel of (0.6, 0.2, −0.1, −0.2) the literal formula gives a net
*negative* coupling and nothing ever synchronises. Default is `per_shell`, where each shell's
sum is divided by that shell's own neighbour count, so `sum(K_s)` is what a fully synchronised
node feels and the sign structure means what III.2 says it means. `global` reproduces the
literal formula.

**2. Noise scaling.** IX.3 writes the noise inside the bracket multiplied by `dt`, which would
make its effect vanish as `dt → 0`. Integrated as Euler-Maruyama (`sqrt(dt)`) so `sigma_noise`
means the same thing at every step size.

**3. Critical bandwidth (`metrics.band_width`).** IX.5's three sanity checks do not jointly
determine it. `sim.cli selftest` sweeps it and prints the separation each component achieves —
see the next section.

**4. The dissonance index is reported twice, extensive and intensive.** `dissonance` divides by
*all* cluster pairs including the ones the coherence gate silenced, which is what makes noise
score low — but it also falls when clusters merely go quiet, so a state with 4 of 16 clusters
coherent and mutually incommensurate scores about 1/20 of the same four tones measured alone.
`dissonance_intensive` divides by the realised coherence weight instead and answers "how
dissonant is the part that *is* coherent". An intervention can move the two in opposite
directions, and when it does, that is the finding rather than a nuisance.

---

## What the self-test found

IX.5 asks for a dissonance metric that scores exact ratios near zero, the source's dissonant TMS
set high, and white noise low. **No single-bandwidth roughness measure can do all three**, and
the reason is structural rather than a tuning failure:

- The source's dissonant TMS set (1.01, 2.01, 3.98, 6.02, …) is *almost exactly harmonic* — a
  common fundamental near 1.005 fits every member to within 1%. It is dissonant by **beating**,
  which is an absolute-detuning phenomenon and needs a narrow critical band.
- A "relatively prime" stack — the phrase III.1 actually uses — has no common fundamental at
  small integers at all, but its partials are far apart, so it produces **no beating**. It is
  dissonant by **incommensurability**, which is a ratio phenomenon and invisible to a narrow band.

Sweeping the bandwidth makes this concrete. The beating term alone never reaches the required
3× separation at any bandwidth:

| band_width | composite | beating only | incommensurability only |
|---|---|---|---|
| 0.005 | 327× | 2.4× | very large |
| 0.010 | 28× | 2.6× | very large |
| 0.020 | 4.9× | 1.1× | very large |
| 0.030 | 2.4× | 0.6× | very large |

So the implemented index is a **composite** of both terms, and both are reported separately
(`d_roughness`, `d_inharmonic`) alongside the combined `dissonance` in every result file. This
is a substantive amendment to IX.5, not a convenience.

IX.5's option (ii) — pairwise distance to the nearest small-integer ratio — is implemented as
`roughness_model="harmonicity"` and **fails** the sanity set (separation 1.1×). The reason is
instructive: it is a *pairwise* measure, so on the source's harmonic TMS stack it scores the pair
96:148 Hz as dissonant (37/24 is not a small ratio) even though every member of the set is a
multiple of 1 Hz. Commensurability is a property of the set, not of its pairs, which is why the
implemented term fits a common fundamental across all tones at once.

**On the noise check.** The obvious version of this test is weaker than it looks, and the
self-test now says so. A uniform-random or random-walk phase field has *no mean phase drift*, so
every cluster's estimated frequency falls below `freq_min_hz` and is gated out before the
amplitude weighting is ever consulted — the check passes without exercising the mechanism it is
supposed to validate. The self-test therefore adds a **drifting-noise** null (each node given a
real natural frequency in the model's own band plus a Wiener phase) and, more importantly, a
comparison between states the simulation actually occupies:

| state | D | D intensive | coherence weight | r_global | LZ |
|---|---|---|---|---|---|
| driven attack (A = 4) | 0.269 | 0.466 | 0.579 | 0.73 | 0.21 |
| undriven, heavy noise | 0.024 | 0.639 | 0.037 | 0.05 | 0.69 |
| undriven, quiet | 0.051 | 0.616 | 0.082 | 0.07 | 0.28 |

The attack state carries 11.5× the dissonance of the heavy-noise state while having *one third*
its LZ complexity. That anti-correlation, not merely a low correlation, is the strongest form of
prediction 4 in the falsification table — and note the intensive column moves the other way,
which is exactly why both are reported.

---

## Experiments

| # | Question | Where the answer lands |
|---|---|---|
| 0 | Does the coupling-kernel framework reproduce at all? | Gate. Flat kernels cohere, Mexican-hat kernels fragment and proliferate defects. Checkerboard and pinwheel select their spatial modes; the traveling-wave kernel does not travel. |
| 1 | **The crux.** Does branching geometry cost more dissonance under drive? | Only in the forced-coherence regime, and modestly. Peak dissonance is geometry-independent, so IX.7's claim as stated does not hold — the refined version does. |
| 2 | What does an attack look like, and when does it stop? | Dissonance is sustained only inside a band of drive amplitude. Above it the tree entrains completely and goes quiet. No spontaneous termination. |
| 3 | The three interventions. | The complexifier relieves and the energy-matched symmetriser does not — the IV.1 clinical anomaly, from kernel shape alone. |
| 4 | Consonant vs matched dissonant entrainment — the one STV-specific prediction. | Does not separate. A positive control says whether that is about the mechanism or about the stimulus being too finely mistuned for this model. |
| 5 | **The validity check.** Does the dissonance index dissociate from entropy? | Yes, and in the strongest available form: the two correlate with opposite signs under noise and under drive. |
| 6 | The IX.6 sweeps nothing else reaches. | Drive variant, K_SW, drive frequency against the tree's own Laplacian modes, branching structure, and node count. |

## How big an effect has to be to mean anything

At N = 400, drive A = 4, the dissonance index has a **run-to-run coefficient of variation of
about 11%** (six seeds: D = 0.283 ± 0.031 varying the dynamics seed alone, 0.274 ± 0.031 varying
the network wiring too). So a difference below roughly 20% between two conditions is not
readable from single runs, and every experiment here averages over three seeds and reports the
spread.

Two numerical checks behind that:

- **Integrator.** Euler at the default `dt = 0.002` accumulates a mean phase error of about
  0.010 rad over 6 s relative to RK4 at `dt = 0.0005`, scaling linearly with `dt` as first-order
  integration should. Varying `dt` from 0.004 to 0.0005 moves D between 0.233 and 0.269 —
  entirely inside the seed-to-seed spread above, so the default is adequate. `--set
  dynamics.integrator=rk4` is there if drift matters.
- **LZ76** reproduces the textbook factorisations (`0001101001000101` → 6, `0101010101` → 3),
  approaches `n/log₂n` on random strings (ratio 1.02 at n = 16384), and gives 4 versus 1196 on a
  periodic versus a random string of the same length.

## Caveats worth carrying forward

- **Node counts are matched only approximately** across geometries (tree 626, lattice 625,
  layered 607 at `n_target=625`): the lattice wants a perfect square and the layered network
  wants s² + (s/2)² + …. The residual mismatch is under 3%.
- **Only the tree has branch points.** The lattice has none, and the layered network's degree
  structure partitions it by *layer*, not by anything the bifurcation claim is about — so the
  branch-point statistic is left undefined there rather than reported as a control. The tree's
  own statistic is compared against a degree-matched null (the same phases permuted across
  nodes), because a degree-3 node shows more local phase disagreement than a degree-2 node even
  in a field with no spatial structure at all.
- **The small-world links are largely redundant with the distance shells.** They are wired
  without a distance constraint, as III.2 specifies, but the graphs have small diameters, so most
  random links land within the 8-hop shell radius — on the layered network, all of them. "Long
  range" is not a meaningful category on these graphs, and `K_SW` is zero in Experiments 1–5
  anyway, so the small-world lever is largely untested here.
- **The traveling-wave kernel does not produce a traveling wave.** Of the source's three
  mode-selecting kernels, checkerboard and pinwheel reproduce their spatial signatures; the
  traveling-wave kernel produces a mode drift of 0.003 Hz, indistinguishable from the standing
  pinwheel. Per XI.9 the tabulated values come from tools on a different scale convention, so
  this is evidence about those specific numbers, not about the implementation — the load-bearing
  simplifier-versus-complexifier contrast does reproduce.
- **The drive target matters more than the spec suggests.** IX.3's Variant 2 couples the
  pacemaker to *all* nodes, which is the default here. Injecting it only at the central relay
  (`--set drive.target=central`) does not propagate at these coupling strengths — the relay locks
  and the periphery ignores it. If the hypothalamic drive is supposed to reach the trigeminal
  tree through the trigeminocervical complex, that route needs a stronger relay-to-periphery
  coupling than the kernel currently provides.
- **Absolute dissonance values do not survive a change of node count.** D falls monotonically
  with N (0.327 → 0.228 → 0.131 across N = 225, 625, 1225 at matched drive). Every experiment
  holds N fixed, so the within-experiment comparisons are unaffected, but no absolute magnitude
  should be read as a property of the model rather than of the mesh.
- **The peripheral oscillator framing is the weakest link**, exactly as XI.2 says. Nothing here
  tests it.
- **Every verdict is computed, and several thresholds inside them are judgement calls.** The
  forced-coherence cut at r_global ≥ 0.7, the "2 × pooled SD" bar for calling an intervention
  effect real, and the 3× separation the metric self-test demands are all choices. They live in
  the code rather than in prose so they can be argued with.
- Everything downstream of "STV is true" inherits STV's own uncertainty, and the coupling-kernel
  source describes itself as early-stage. That request propagates.

## Provenance

The implementation was audited by an independent multi-agent review against Part IX, every
finding adversarially verified before being accepted. Sixteen survived verification and are
fixed here, each annotated at the point of the fix. The ones that changed a reported number:

- Intervention windows were silently relocated — or never closed — when `t_start`/`t_end` were
  not exact multiples of `dt`, because boundaries were snapped with `round` instead of `ceil`.
- The incommensurability term used a residual relative to harmonic number, so its tolerance grew
  with that number until almost any tone set fit some template. Correcting it roughly tripled the
  measured tree-versus-lattice gap.
- Experiment 2's "sustained fraction" was normalised against each window's own peak, so a window
  containing no attack scored as fully sustained; and its spontaneous-termination test was
  measured across the drive's ramp-down, so it was detecting drive withdrawal.
- Experiment 5's dissociation test checked for the *absence* of correlation, when a strong
  negative correlation is precisely the result being sought.
- Experiment 1's headline averaged the geometry ratio over coherence regimes that the same
  verdict text explicitly excluded.

---

*Nothing here is medical advice. Effective legal treatments for cluster headache exist —
including high-flow oxygen and triptans — and belong in the hands of a clinician.*
