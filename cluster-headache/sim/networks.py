"""Network construction (Part IX.2).

Three geometries, node count matched as closely as each construction allows:

  (A) trigeminal branching tree  - the experimental geometry
  (B) 2D square lattice          - "a geometry that can host a consonant mode"
  (C) hierarchical layered net   - 16x16 -> 8x8 -> 4x4 -> 2x2, the visual-cortex analogue

For each we precompute:
  * four geodesic distance shells (D1 = 1 hop, D2 = 2, D3 = 3, D4 = 4..cap) as sparse
    matrices, computed on the *base* graph before small-world links are added - otherwise
    the random links collapse the very geodesic metric the kernel is defined over;
  * a separate small-world adjacency (each node wired to `sw_links_per_node` others);
  * a compartment label per node (central / peripheral, plus division for the tree);
  * a cluster partition built the same way on every geometry (farthest-point seeds +
    multi-source BFS), so the local order parameter and the dissonance index are
    comparable between them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .config import NetworkConfig

COMPARTMENTS = {"central": 0, "ganglion": 1, "v1": 2, "v2": 3, "v3": 4, "bulk": 5}


@dataclass
class Network:
    """A fully precomputed substrate for the integrator."""

    geometry: str
    n: int
    adjacency: sp.csr_matrix  # base graph, no small-world links
    shells: List[sp.csr_matrix] = field(default_factory=list)
    shells_rownorm: List[sp.csr_matrix] = field(default_factory=list)  # each row sums to 1
    sw_adjacency: Optional[sp.csr_matrix] = None  # small-world links only
    neighbour_counts: Optional[np.ndarray] = None  # N_i, shells + small-world
    compartment: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    degree: Optional[np.ndarray] = None
    clusters: Optional[np.ndarray] = None
    positions: Optional[np.ndarray] = None  # (N, 2), for figures
    grid_shape: Optional[Tuple[int, int]] = None
    bifurcation_nodes: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    chain_nodes: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    parent: Optional[np.ndarray] = None  # tree only, -1 at the root
    meta: Dict = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return int(self.clusters.max()) + 1 if self.clusters is not None else 0

    def mask(self, compartment: str) -> np.ndarray:
        if compartment == "all":
            return np.ones(self.n, dtype=bool)
        if compartment == "peripheral":
            return self.compartment != COMPARTMENTS["central"]
        if compartment not in COMPARTMENTS:
            raise KeyError(f"unknown compartment {compartment!r}")
        return self.compartment == COMPARTMENTS[compartment]


# --------------------------------------------------------------------------------------
# Edge-list helpers
# --------------------------------------------------------------------------------------


def _edges_to_csr(edges: Sequence[Tuple[int, int]], n: int) -> sp.csr_matrix:
    if len(edges) == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    arr = np.asarray(edges, dtype=np.int32)
    rows, cols = arr[:, 0], arr[:, 1]
    data = np.ones(rows.size, dtype=np.float32)
    mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    mat = (mat + mat.T).tocsr()
    mat.data[:] = 1.0  # collapse multi-edges
    mat.setdiag(0.0)
    mat.eliminate_zeros()
    return mat


# --------------------------------------------------------------------------------------
# (A) trigeminal branching tree
# --------------------------------------------------------------------------------------


def build_tree(cfg: NetworkConfig) -> Network:
    rng = np.random.default_rng(cfg.seed)
    n_target = cfg.n_target

    n_central = max(4, int(round(cfg.central_fraction * n_target)))
    edges: List[Tuple[int, int]] = []
    compartment: List[int] = []
    depth: List[int] = []
    parent: List[int] = []

    def add_node(comp: int, d: int, par: int) -> int:
        idx = len(compartment)
        compartment.append(comp)
        depth.append(d)
        parent.append(par)
        return idx

    # --- central compartment: densely interconnected relay ---
    central = [add_node(COMPARTMENTS["central"], 0, -1) for _ in range(n_central)]
    for i in range(n_central):
        for j in range(i + 1, n_central):
            if rng.random() < cfg.central_density:
                edges.append((central[i], central[j]))
    for i in range(1, n_central):  # guarantee the central block is connected
        edges.append((central[i - 1], central[i]))

    # --- trigeminal ganglion ---
    ganglion = add_node(COMPARTMENTS["ganglion"], 1, central[0])
    for hub in central[: max(1, n_central // 3)]:
        edges.append((ganglion, hub))

    # --- divisions: cluster pain is overwhelmingly V1, so V1 carries most nodes ---
    if cfg.v1_only:
        div_names, weights = ["v1"], [1.0]
    else:
        div_names = ["v1", "v2", "v3"]
        weights = list(cfg.division_weights)
    total_w = float(sum(weights))
    weights = [w / total_w for w in weights]

    n_peripheral = n_target - len(compartment)
    if n_peripheral < len(div_names) * 4:
        raise ValueError(f"n_target={n_target} too small for a tree with {n_central} central nodes")

    for name, weight in zip(div_names, weights):
        comp = COMPARTMENTS[name]
        budget = int(round(weight * n_peripheral))
        root = add_node(comp, 2, ganglion)
        edges.append((ganglion, root))
        budget -= 1

        frontier: deque = deque([(root, 2)])
        while budget > 0 and frontier:
            node, d = frontier.popleft()
            if d >= cfg.max_depth:
                continue
            if cfg.heterogeneous:
                b = int(np.clip(rng.poisson(max(cfg.branching_factor - 1, 0)) + 1, 1, 5))
            else:
                b = cfg.branching_factor
            for _ in range(b):
                if budget <= 0:
                    break
                if cfg.heterogeneous:
                    m = int(np.clip(
                        round(rng.lognormal(np.log(max(cfg.segment_length, 1)), cfg.lognormal_sigma)),
                        1, 4 * cfg.segment_length,
                    ))
                else:
                    m = cfg.segment_length
                m = min(m, budget)
                prev = node
                for _ in range(m):
                    nxt = add_node(comp, d + 1, prev)
                    edges.append((prev, nxt))
                    prev = nxt
                    budget -= 1
                frontier.append((prev, d + 1))

    n = len(compartment)
    adjacency = _edges_to_csr(edges, n)
    compartment_arr = np.asarray(compartment, dtype=np.int8)
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int32)

    # Bifurcations vs chain segments in the *peripheral* tree.  The central block is dense
    # by construction and the ganglion is a hub wired into it, so both are excluded from
    # each set: the Model B claim is about branch points in the trigeminal tree, and a
    # relay hub with a dozen neighbours is not one.
    branchable = (compartment_arr != COMPARTMENTS["central"]) & (
        compartment_arr != COMPARTMENTS["ganglion"]
    )
    bif = np.where(branchable & (degree >= 3))[0]
    chain = np.where(branchable & (degree == 2))[0]

    net = Network(
        geometry="tree",
        n=n,
        adjacency=adjacency,
        compartment=compartment_arr,
        depth=np.asarray(depth, dtype=np.int32),
        degree=degree,
        parent=np.asarray(parent, dtype=np.int32),
        bifurcation_nodes=bif,
        chain_nodes=chain,
        meta={"n_central": n_central, "ganglion": ganglion, "divisions": div_names},
    )
    net.positions = _tree_layout(net)
    return net


def _tree_layout(net: Network) -> np.ndarray:
    """Radial layout; figures only."""
    pos = np.zeros((net.n, 2))
    depth = net.depth.astype(float)
    rng = np.random.default_rng(0)
    angle = np.zeros(net.n)
    counters: Dict[int, int] = {}
    for node in np.argsort(net.depth, kind="stable"):
        par = int(net.parent[node])
        if par < 0:
            angle[node] = rng.uniform(0, 2 * np.pi)
        else:
            k = counters.get(par, 0)
            counters[par] = k + 1
            spread = 1.6 / (1.0 + depth[node]) ** 1.05
            angle[node] = angle[par] + spread * ((k + 1) // 2) * (1 if k % 2 else -1)
    r = 0.4 + depth
    pos[:, 0] = r * np.cos(angle)
    pos[:, 1] = r * np.sin(angle)
    return pos


# --------------------------------------------------------------------------------------
# (B) 2D lattice
# --------------------------------------------------------------------------------------


def build_lattice(cfg: NetworkConfig) -> Network:
    side = max(4, int(round(np.sqrt(cfg.n_target))))
    n = side * side
    idx = np.arange(n).reshape(side, side)
    edges: List[Tuple[int, int]] = []
    for i in range(side):
        for j in range(side):
            a = int(idx[i, j])
            if j + 1 < side:
                edges.append((a, int(idx[i, j + 1])))
            elif cfg.lattice_periodic:
                edges.append((a, int(idx[i, 0])))
            if i + 1 < side:
                edges.append((a, int(idx[i + 1, j])))
            elif cfg.lattice_periodic:
                edges.append((a, int(idx[0, j])))

    adjacency = _edges_to_csr(edges, n)
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int32)
    positions = np.stack(
        [np.repeat(np.arange(side), side), np.tile(np.arange(side), side)], axis=1
    ).astype(float)

    # A comparable "relay" region, so compartment-targeted interventions mean something
    # on the control geometry too.
    r2 = (positions[:, 0] - side / 2) ** 2 + (positions[:, 1] - side / 2) ** 2
    n_central = max(4, int(round(cfg.central_fraction * n)))
    compartment = np.full(n, COMPARTMENTS["bulk"], dtype=np.int8)
    compartment[np.argsort(r2)[:n_central]] = COMPARTMENTS["central"]

    return Network(
        geometry="lattice",
        n=n,
        adjacency=adjacency,
        compartment=compartment,
        depth=np.zeros(n, dtype=np.int32),
        degree=degree,
        positions=positions,
        grid_shape=(side, side),
        chain_nodes=np.arange(n),
        meta={"side": side},
    )


# --------------------------------------------------------------------------------------
# (C) hierarchical layered network
# --------------------------------------------------------------------------------------


def _hier_sides(side0: int, layers: int) -> List[int]:
    sides = [side0]
    for _ in range(layers - 1):
        sides.append(max(2, -(-sides[-1] // 2)))
    return sides


def build_hierarchical(cfg: NetworkConfig) -> Network:
    layers = cfg.hier_layers
    best_side, best_err = 4, None
    for side0 in range(4, 200):
        total = sum(s * s for s in _hier_sides(side0, layers))
        err = abs(total - cfg.n_target)
        if best_err is None or err < best_err:
            best_side, best_err = side0, err
    sides = _hier_sides(best_side, layers)

    offsets: List[int] = []
    total = 0
    for s in sides:
        offsets.append(total)
        total += s * s
    n = total

    edges: List[Tuple[int, int]] = []
    compartment = np.full(n, COMPARTMENTS["bulk"], dtype=np.int8)
    depth = np.zeros(n, dtype=np.int32)
    positions = np.zeros((n, 2))

    def node_id(layer: int, i: int, j: int) -> int:
        return offsets[layer] + i * sides[layer] + j

    for layer, s in enumerate(sides):
        for i in range(s):
            for j in range(s):
                a = node_id(layer, i, j)
                depth[a] = layer
                positions[a] = (i * (sides[0] / s), j * (sides[0] / s))
                if j + 1 < s:
                    edges.append((a, node_id(layer, i, j + 1)))
                if i + 1 < s:
                    edges.append((a, node_id(layer, i + 1, j)))
                if layer + 1 < layers:  # forward/backward inter-layer coupling
                    up = sides[layer + 1]
                    edges.append((a, node_id(layer + 1, min(i // 2, up - 1), min(j // 2, up - 1))))

    # Relay = the topmost nodes, but sized by central_fraction like the other geometries.
    # Hard-coding it to the top layer would give this arm a 1.5% relay against the tree's
    # and lattice's 5%, and since omega0 differs by compartment that alone changes the
    # frequency dispersion - a dissonance driver - between Experiment 1's arms.
    n_central = max(4, int(round(cfg.central_fraction * n)))
    order = np.argsort(-depth, kind="stable")  # deepest layer (the apex) first
    compartment[order[:n_central]] = COMPARTMENTS["central"]

    adjacency = _edges_to_csr(edges, n)
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int32)

    return Network(
        geometry="hierarchical",
        n=n,
        adjacency=adjacency,
        compartment=compartment,
        depth=depth,
        degree=degree,
        positions=positions,
        # No true grid: N = sum of s^2 over layers, not sides[0]^2, so anything that
        # reshapes a node vector into a square (the DCT readout, the lattice phase map)
        # must not be handed this geometry.
        grid_shape=None,
        # This network has no branch points.  Degree here partitions it by layer, not by
        # anything the Model B bifurcation claim is about, so the branch-point statistic is
        # left undefined rather than reported as a meaningless control.
        bifurcation_nodes=np.array([], dtype=int),
        chain_nodes=np.arange(n),
        meta={"sides": sides, "offsets": offsets, "layer_side": sides[0]},
    )


# --------------------------------------------------------------------------------------
# Shells, small-world links, cluster partition
# --------------------------------------------------------------------------------------


def compute_shells(adjacency: sp.csr_matrix, n_shells: int, cap: int) -> List[sp.csr_matrix]:
    """Geodesic distance shells by repeated boolean adjacency products.

    Shell s < n_shells holds exactly-s-hop pairs; the last shell holds n_shells..cap hops.
    Cheaper and far less memory-hungry than an all-pairs shortest-path matrix.
    """
    n = adjacency.shape[0]
    a = adjacency.astype(np.int8)
    a.data[:] = 1

    reached = (sp.identity(n, dtype=np.int8, format="csr") + a).tocsr()
    reached.data[:] = 1
    frontier = a.copy()
    shells: List[sp.csr_matrix] = [a.astype(np.float32)]

    for hop in range(2, max(n_shells, cap) + 1):
        nxt = (frontier @ a).tocsr()
        nxt.data[:] = 1
        overlap = nxt.multiply(reached)
        new = (nxt - overlap).tocsr()
        new.setdiag(0)
        new.eliminate_zeros()
        frontier = new
        if new.nnz == 0:
            break
        reached = (reached + new).tocsr()
        reached.data[:] = 1
        if hop < n_shells:
            shells.append(new.astype(np.float32))
        elif hop == n_shells:
            shells.append(new.astype(np.float32))
        else:
            shells[-1] = (shells[-1] + new.astype(np.float32)).tocsr()

    while len(shells) < n_shells:
        shells.append(sp.csr_matrix((n, n), dtype=np.float32))
    out = []
    for s in shells[:n_shells]:
        s = s.tocsr()
        if s.nnz:
            s.data[:] = 1.0
        out.append(s)
    return out


def _row_normalise(mat: sp.csr_matrix) -> sp.csr_matrix:
    """Divide each row by its own degree, leaving empty rows at zero."""
    if mat.nnz == 0:
        return mat.copy()
    deg = np.asarray(mat.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    nz = deg > 0
    inv[nz] = 1.0 / deg[nz]
    return (sp.diags(inv.astype(np.float32)) @ mat).tocsr()


def add_small_world(n: int, links_per_node: int, seed: int) -> sp.csr_matrix:
    """Per III.2: at initialisation each oscillator is wired to `links_per_node`
    randomly chosen others."""
    if links_per_node <= 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    rng = np.random.default_rng(seed + 9871)
    rows = np.repeat(np.arange(n), links_per_node)
    cols = rng.integers(0, n, size=rows.size)
    keep = rows != cols
    return _edges_to_csr(np.stack([rows[keep], cols[keep]], axis=1), n)


def _bfs_distances(adjacency: sp.csr_matrix, sources: Sequence[int]) -> np.ndarray:
    """Multi-source BFS hop distances (inf where unreachable)."""
    n = adjacency.shape[0]
    dist = np.full(n, np.inf)
    indptr, indices = adjacency.indptr, adjacency.indices
    queue = deque()
    for s in sources:
        dist[s] = 0.0
        queue.append(s)
    while queue:
        node = queue.popleft()
        d = dist[node] + 1
        for nb in indices[indptr[node]:indptr[node + 1]]:
            if dist[nb] > d:
                dist[nb] = d
                queue.append(nb)
    return dist


def partition_clusters(adjacency: sp.csr_matrix, k: int, seed: int) -> np.ndarray:
    """Farthest-point seeding plus nearest-seed assignment.

    Built identically on every geometry, so cluster count and typical cluster size are
    matched and the dissonance index compares like with like across Experiment 1.
    """
    n = adjacency.shape[0]
    k = max(2, min(k, n))
    rng = np.random.default_rng(seed + 555)
    seeds = [int(rng.integers(0, n))]
    dmin = _bfs_distances(adjacency, seeds)
    for _ in range(k - 1):
        finite = np.where(np.isfinite(dmin), dmin, -1.0)
        nxt = int(np.argmax(finite))
        if finite[nxt] <= 0:  # graph exhausted; fall back to a random unused node
            unused = [i for i in range(n) if i not in seeds]
            if not unused:
                break
            nxt = int(rng.choice(unused))
        seeds.append(nxt)
        dmin = np.minimum(dmin, _bfs_distances(adjacency, [nxt]))

    dists = np.stack([_bfs_distances(adjacency, [s]) for s in seeds], axis=0)
    dists = np.where(np.isfinite(dists), dists, 1e9)
    return np.argmin(dists, axis=0).astype(np.int32)


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------


def build_network(cfg: NetworkConfig) -> Network:
    builders = {"tree": build_tree, "lattice": build_lattice, "hierarchical": build_hierarchical}
    if cfg.geometry not in builders:
        raise KeyError(f"unknown geometry {cfg.geometry!r}; have {sorted(builders)}")
    net = builders[cfg.geometry](cfg)
    net.shells = compute_shells(net.adjacency, cfg.n_shells, cfg.shell_cap)
    net.shells_rownorm = [_row_normalise(s) for s in net.shells]
    net.sw_adjacency = add_small_world(net.n, cfg.sw_links_per_node, cfg.seed)
    net.clusters = partition_clusters(net.adjacency, cfg.target_clusters, cfg.seed)

    counts = np.zeros(net.n, dtype=np.float64)
    for shell in net.shells:
        counts += np.asarray(shell.sum(axis=1)).ravel()
    counts += np.asarray(net.sw_adjacency.sum(axis=1)).ravel()
    net.neighbour_counts = np.maximum(counts, 1.0)
    net.meta["shell_sizes"] = [float(s.nnz) / net.n for s in net.shells]
    net.meta["n_clusters"] = int(net.clusters.max()) + 1
    return net


def matched_networks(cfg: NetworkConfig, geometries: Sequence[str]) -> Dict[str, Network]:
    """Build several geometries at a matched target node count (Experiment 1).

    Exact matching is impossible - the lattice wants a perfect square, the layered net
    wants s^2 + (s/2)^2 + ... - so `n_target` should be chosen as a perfect square and
    the residual mismatch reported rather than hidden.
    """
    nets: Dict[str, Network] = {}
    for geom in geometries:
        sub = NetworkConfig(**{**cfg.__dict__, "geometry": geom})
        nets[geom] = build_network(sub)
    return nets
