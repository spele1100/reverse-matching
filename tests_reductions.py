"""Tests for the Chapter 9 reduction layer (Lemmas 9.1-9.3).

Cross-checks the constructed instances directly against NetworkX's blossom
implementation (max_weight_matching with maxcardinality=True), per the
kill-test plan (spec sections 4.1/4.2).
"""

import random
import time

import networkx as nx
import pytest

from reductions import (
    PermNetwork, build_hk, construct_witness_hk, degree_reduce, normalize,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def nx_max_matching_size(n, edges):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    m = nx.algorithms.matching.max_weight_matching(g, maxcardinality=True)
    return len(m)


def nx_perfect(graph):
    """Perfect matching of an nx.Graph via blossom, or None."""
    m = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True)
    if len(m) * 2 == graph.number_of_nodes():
        return m
    return None


def check_perfect(vertices, edge_objs, matching_edges):
    """Verify `matching_edges` (frozensets or pairs) is a perfect matching
    of the graph given by `edge_objs` (objects with .u/.v)."""
    by_pair = {frozenset((e.u, e.v)) for e in edge_objs}
    cover = {}
    for pair in matching_edges:
        if len(pair) > 2:  # Edge namedtuple (u, v, etype, eid)
            pair = pair[:2]
        key = frozenset(pair)
        assert key in by_pair, f"edge {sorted(key)} not a constructed edge"
        for v in key:
            cover[v] = cover.get(v, 0) + 1
    assert len(matching_edges) * 2 == len(cover), "matching covers odd set"
    for v in vertices:
        assert cover.get(v, 0) == 1, f"vertex {v} covered {cover.get(v, 0)}x"


def random_edges(n, rng, p=None):
    if p is None:
        p = rng.random() * 0.5
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return normalize(n, pairs=sorted(edges))


# ---------------------------------------------------------------------------
# Lemma 9.1: permutation network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("s", [1, 2, 4, 8, 16, 64, 256])
def test_network_routable_vertex_disjoint(s):
    rng = random.Random(1234 + s)
    for _ in range(5):
        perm_list = list(range(s))
        rng.shuffle(perm_list)
        perm = dict(enumerate(perm_list))
        net = PermNetwork(s)
        arcset = net.arc_set()
        paths = net.route(perm)
        seen = set()
        for i, path in paths.items():
            assert path[0] == net.inputs[i]
            assert path[-1] == net.outputs[perm[i]]
            for p, q in zip(path, path[1:]):
                assert (p, q) in arcset, (p, q)
            for p in path:
                assert p not in seen, "paths not vertex-disjoint"
                seen.add(p)
        assert len(seen) == len(net.positions())


def test_network_size_bound():
    import math
    for s in [2 ** j for j in range(1, 15)]:
        net = PermNetwork(s)
        # positions T(s) = 2T(s/2) + 2s; arcs A(s) = 2A(s/2) + 4s
        assert len(net.positions()) <= 2 * s * (math.log2(s) + 1)
        assert len(net.arc_set()) <= 4 * s * (math.log2(s) + 1)


# ---------------------------------------------------------------------------
# Lemma 9.2: cardinality -> perfect matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_roundtrip_random(seed):
    rng = random.Random(seed)
    n = rng.randint(0, 36)
    edges = random_edges(n, rng)
    mm = nx_max_matching_size(n, edges)
    edge_set = {frozenset(e) for e in edges}
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    max_m = nx.algorithms.matching.max_weight_matching(g, maxcardinality=True)
    for _ in range(4):
        k = rng.randint(0, max(0, n // 2))
        hk = build_hk(n, edges, k)
        perfect = nx_perfect(hk.to_networkx())
        feasible = mm >= k
        assert (perfect is not None) == feasible, (
            f"n={n} m={len(edges)} k={k}: H_k perfect={perfect is not None}, "
            f"G max={mm}"
        )
        if perfect is None:
            continue
        # decode: perfect matching of H_k -> size-k matching of G
        dec = hk.decode(perfect)
        assert len(dec) == k
        covered = set()
        for (u, v) in dec:
            assert frozenset((u, v)) in edge_set
            assert u not in covered and v not in covered
            covered.update((u, v))
        # lift: any size-k submatching -> explicit perfect matching of H_k
        sub = list(max_m)[:k]
        covered_sub = set()
        for (u, v) in sub:
            covered_sub.update((u, v))
        uncovered = sorted(set(range(n)) - covered_sub)
        lifted = construct_witness_hk(hk, sub, uncovered)
        check_perfect(hk.vertices, hk.edges, lifted)


def test_hk_direct_crosscheck_complete_graphs():
    # complete graphs K_1..K_8, all k
    for n in range(1, 9):
        edges = normalize(
            n, pairs=[(u, v) for u in range(n) for v in range(u + 1, n)])
        mm = nx_max_matching_size(n, edges)
        for k in range(0, n // 2 + 1):
            hk = build_hk(n, edges, k)
            assert (nx_perfect(hk.to_networkx()) is not None) == (mm >= k), (
                n, k, mm)


def test_hk_edge_cases():
    # n = 0
    hk = build_hk(0, [], 0)
    assert nx_perfect(hk.to_networkx()) is not None

    # n = 1 (empty graph, empty matching of size 0; text: return empty list)
    hk = build_hk(1, [], 0)
    assert nx_perfect(hk.to_networkx()) is not None

    # n = 2, no edge: k=0 feasible, k=1 not
    assert nx_perfect(build_hk(2, [], 0).to_networkx()) is not None
    assert nx_perfect(build_hk(2, [], 1).to_networkx()) is None

    # n = 2, with edge: both k feasible
    edges = normalize(2, pairs=[(0, 1)])
    for k in (0, 1):
        hk = build_hk(2, edges, k)
        assert nx_perfect(hk.to_networkx()) is not None

    # odd n
    edges = normalize(5, pairs=[(0, 1), (1, 2), (2, 3), (3, 4)])
    assert nx_perfect(build_hk(5, edges, 2).to_networkx()) is not None

    # disconnected: two triangles
    edges = normalize(6, pairs=[(0, 1), (1, 2), (0, 2),
                                (3, 4), (4, 5), (3, 5)])
    mm = nx_max_matching_size(6, edges)
    assert mm == 2
    for k in range(3):
        hk = build_hk(6, edges, k)
        assert (nx_perfect(hk.to_networkx()) is not None) == (k <= mm)

    # k = n//2 on even n, path graph
    edges = normalize(8, pairs=[(i, i + 1) for i in range(7)])
    assert nx_perfect(build_hk(8, edges, 4).to_networkx()) is not None

    # k = 0 with b = n (all outputs selected when n = s)
    edges = normalize(3, pairs=[(0, 1)])
    hk = build_hk(3, edges, 0)
    assert hk.b == 3
    assert hk.s == 4
    assert len(hk.selected_outputs) == 3
    assert nx_perfect(hk.to_networkx()) is not None
    hk = build_hk(3, edges, 1)
    assert nx_perfect(hk.to_networkx()) is not None  # matching {(0,1)}


# ---------------------------------------------------------------------------
# Lemma 9.3: degree gadget
# ---------------------------------------------------------------------------


def _gadget_vertex_count(n, edges):
    deg = [0] * n
    for (u, v) in edges:
        deg[u] += 1
        deg[v] += 1
    return sum(2 * d - 1 for d in deg if d > 0)


def _max_degree(edge_objs):
    d = {}
    for e in edge_objs:
        d[e.u] = d.get(e.u, 0) + 1
        d[e.v] = d.get(e.v, 0) + 1
    return max(d.values(), default=0)


@pytest.mark.parametrize("seed", range(25))
def test_degree_gadget_random(seed):
    rng = random.Random(9000 + seed)
    n = rng.randint(1, 22)
    edges = random_edges(n, rng)
    mm = nx_max_matching_size(n, edges)
    perfect_orig = (2 * mm == n)
    g3 = degree_reduce(range(n), edges)
    if g3.isolate is not None:
        assert not perfect_orig  # an isolate really forbids a perfect matching
        return
    # size formula: sum (2 deg u - 1) = 4e - v
    assert len(g3.new_vertices) == 4 * len(edges) - n
    assert _max_degree(g3.new_edges) <= 3
    # equivalence via NetworkX
    pm3 = nx_perfect(g3.to_networkx())
    assert (pm3 is not None) == perfect_orig
    if not perfect_orig:
        return
    # decode: gadget perfect matching -> original perfect matching
    dec = g3.project(pm3)
    assert len(dec) == n // 2
    covered = set()
    for (u, v) in dec:
        assert (u, v) in [tuple(sorted(e)) for e in edges] or \
            (v, u) in [tuple(sorted(e)) for e in edges]
        assert u not in covered and v not in covered
        covered.update((u, v))
    assert len(covered) == n
    # lift: original perfect matching -> gadget perfect matching
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    max_m = nx.algorithms.matching.max_weight_matching(g, maxcardinality=True)
    lifted = g3.lift(list(max_m))
    check_perfect(g3.new_vertices, g3.new_edges, lifted)


def test_degree_gadget_complete_graphs():
    # K_t, t up to 7: perfect iff t even; K_1 is an isolate
    for t in range(1, 8):
        edges = normalize(
            t, pairs=[(u, v) for u in range(t) for v in range(u + 1, t)])
        g3 = degree_reduce(range(t), edges)
        perfect = (t % 2 == 0)
        if t == 1:
            assert g3.isolate is not None
            continue
        assert g3.isolate is None
        assert len(g3.new_vertices) == 4 * len(edges) - t
        assert _max_degree(g3.new_edges) <= 3
        assert (nx_perfect(g3.to_networkx()) is not None) == perfect


def test_degree_gadget_high_degree():
    # star K_1,10: no isolates, but no perfect matching (deficient leaves)
    edges = normalize(11, pairs=[(0, i) for i in range(1, 11)])
    g3 = degree_reduce(range(11), edges)
    assert g3.isolate is None
    assert _max_degree(g3.new_edges) <= 3
    assert nx_perfect(g3.to_networkx()) is None

    # single isolated vertex certifies absence
    assert degree_reduce(range(1), []).isolate is not None

    # two degree-5 hubs joined, leaves pairable among themselves:
    # perfect matching {(0,5),(1,2),(3,4),(6,7),(8,9)}
    edges = normalize(
        10, pairs=[(0, i) for i in range(1, 5)]
        + [(5, i) for i in range(6, 10)]
        + [(0, 5), (1, 2), (3, 4), (6, 7), (8, 9)])
    g3 = degree_reduce(range(10), edges)
    assert _max_degree(g3.new_edges) <= 3
    assert len(g3.new_vertices) == 4 * len(edges) - 10
    pm = nx_perfect(g3.to_networkx())
    assert pm is not None
    assert len(g3.project(pm)) == 5
    lifted = g3.lift([(0, 5), (1, 2), (3, 4), (6, 7), (8, 9)])
    check_perfect(g3.new_vertices, g3.new_edges, lifted)


# ---------------------------------------------------------------------------
# Composed pipeline (Eq. 9.1): H_k followed by the degree gadget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_full_pipeline(seed):
    rng = random.Random(7000 + seed)
    n = rng.randint(2, 18)
    edges = random_edges(n, rng)
    mm = nx_max_matching_size(n, edges)
    k = rng.randint(0, n // 2)
    hk = build_hk(n, edges, k)
    g3 = degree_reduce(hk.vertices, [(e.u, e.v) for e in hk.edges])
    assert g3.isolate is None  # H_k never has isolates
    assert _max_degree(g3.new_edges) <= 3
    assert (nx_perfect(g3.to_networkx()) is not None) == (mm >= k)
    assert (nx_perfect(hk.to_networkx()) is not None) == (mm >= k)


# ---------------------------------------------------------------------------
# Scale (sizes only, per kill-test plan)
# ---------------------------------------------------------------------------


def test_scale_n_10000():
    import math
    rng = random.Random(42)
    n = 10_000
    m_target = 15_000
    pairs = set()
    while len(pairs) < m_target:
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    edges = normalize(n, pairs=sorted(pairs))
    k = n // 5
    t0 = time.time()
    hk = build_hk(n, edges, k)
    t1 = time.time()
    npos = len(hk.net.positions())
    narc = len(hk.net.arc_set())
    log2s = math.log2(hk.s)
    assert npos <= 2 * hk.s * (log2s + 1)
    assert narc <= 4 * hk.s * (log2s + 1)
    assert len(hk.vertices) == n + 2 * npos + hk.b
    t2 = time.time()
    g3 = degree_reduce(hk.vertices, [(e.u, e.v) for e in hk.edges])
    t3 = time.time()
    assert g3.isolate is None
    assert _max_degree(g3.new_edges) <= 3
    assert len(g3.new_vertices) == 4 * len(hk.edges) - len(hk.vertices)
    print(f"\nscale n={n} m={len(edges)} k={k}: "
          f"positions={npos} arcs={narc} "
          f"|V(H_k)|={len(hk.vertices)} |E(H_k)|={len(hk.edges)} "
          f"build_hk={t1 - t0:.2f}s build_g3={t3 - t2:.2f}s")
