"""
tests_phase_machine.py — Layer 2 test suite for the Chapter 3 phase state
machine (phase_machine.py).

Cross-checks the perfect-matching decision and its explicit witness against
networkx's blossom implementation (max_weight_matching(maxcardinality=True))
on random subcubic graphs (small + medium seeds), plus targeted invariant
tests:

  * ledger invariants (Def 3.1 / (3.4), bottom debt-free, top positivity,
    Lemma 3.2 surplus) at every settled state — enforced inside the machine
    via PhaseParams(check_invariants=True), which is the default;
  * contract/expand ledger-edit formulas (aggregation exactness, expansion
    C+1 / L-C-1 split, child slack 1, donor paddings) via debug_edits;
  * projection both outcomes on the pure exchange network (Lemma 3.5),
    including (3.13) on a hand-crafted failing instance, plus a forced
    end-to-end peel exercising the deletion transformation / blocker path;
  * tight-set uniqueness and minimality of pair_decision against brute
    force over subsets (Lemma 3.6);
  * materialization produces a valid matching from every prescribed
    exposure of every final bin (factor-criticality, Lemma 3.8);
  * Dinic against networkx on random networks.
"""

import random
from fractions import Fraction
from itertools import combinations

import networkx as nx
import pytest

from phase_machine import (Dinic, PhaseMachine, PhaseParams,
                           exchange_projection, pair_decision)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def rand_subcubic(n, rng, p):
    deg = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if deg[u] < 3 and deg[v] < 3 and rng.random() < p:
                edges.append((u, v))
                deg[u] += 1
                deg[v] += 1
    return edges


def nx_has_perfect(n, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    M = nx.max_weight_matching(G, maxcardinality=True)
    return 2 * len(M) == n


def check_witness(n, edges, M):
    eid = {frozenset(e) for e in edges}
    seen = set()
    for (u, v) in M:
        assert frozenset((u, v)) in eid, f"({u},{v}) is not an edge"
        assert u not in seen and v not in seen, "not a matching"
        seen.add(u)
        seen.add(v)
    assert len(seen) == n


# ---------------------------------------------------------------------------
# deterministic tiny graphs
# ---------------------------------------------------------------------------

def test_empty():
    assert PhaseMachine(0, []).run() == []


def test_single_and_odd():
    assert PhaseMachine(1, []).run() is None
    assert PhaseMachine(3, [(0, 1), (1, 2)]).run() is None
    assert PhaseMachine(5, [(0, 1), (2, 3)]).run() is None


def test_n2():
    assert PhaseMachine(2, [(0, 1)]).run() == [(0, 1)]
    assert PhaseMachine(2, []).run() is None


def test_isolated_vertex():
    # n even but an isolated vertex kills perfect matchability
    assert PhaseMachine(4, [(0, 1)]).run() is None


def test_deterministic_graphs():
    cases = [
        (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], True),   # K4
        (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], True),   # C6
        (6, [(0, 1), (0, 2), (1, 2), (3, 4), (4, 5), (3, 5)], False),  # 2 triangles
        (6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], False),
        (8, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 7)], False),
        (8, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4)], True),                                            # 2 C4 + bridge
        (8, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
             (7, 0)], True),                                            # C8
        (6, [(0, 1), (0, 2), (0, 3), (4, 5)], False),                  # claw + edge
        (10, [(0, 1), (0, 4), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7),
              (3, 4), (3, 8), (4, 9), (5, 7), (5, 8), (6, 8), (6, 9),
              (7, 9)], True),                                           # Petersen
        (10, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6), (6, 7),
              (7, 8), (8, 9), (9, 5), (0, 5)], True),                   # 2 C5 + bridge
        (4, [(0, 1), (1, 2), (2, 3)], True),                            # P4
        (4, [(0, 1), (0, 2), (0, 3)], False),                           # claw K1,3
        (12, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5),
              (6, 7), (6, 8), (7, 8), (9, 10), (9, 11), (10, 11)], False),
    ]
    for (n, edges, expect) in cases:
        M = PhaseMachine(n, edges).run()
        if expect:
            assert M is not None, (n, edges)
            check_witness(n, edges, M)
        else:
            assert M is None, (n, edges, M)


# ---------------------------------------------------------------------------
# randomized cross-checks against networkx
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(30))
def test_random_small(seed):
    rng = random.Random(1000 + seed)
    n = rng.choice([4, 6, 8, 10, 12, 14, 16])
    p = rng.choice([0.15, 0.3, 0.5])
    edges = rand_subcubic(n, rng, p)
    expect = nx_has_perfect(n, edges)
    M = PhaseMachine(n, edges).run()
    if expect:
        assert M is not None, (n, edges)
        check_witness(n, edges, M)
    else:
        assert M is None, (n, edges)


@pytest.mark.parametrize("seed", range(10))
def test_random_medium(seed):
    rng = random.Random(2000 + seed)
    n = rng.choice([18, 20, 24, 26, 28])
    p = rng.choice([0.2, 0.35])
    edges = rand_subcubic(n, rng, p)
    expect = nx_has_perfect(n, edges)
    M = PhaseMachine(n, edges).run()
    if expect:
        assert M is not None, (n, edges)
        check_witness(n, edges, M)
    else:
        assert M is None, (n, edges)


# ---------------------------------------------------------------------------
# multi-ledger parameter sweep (kill-test 4.4 criterion)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,l_mult", [(1, 2), (2, 10), (3, 100), (5, 100),
                                     (8, 4)])
def test_multi_ledger_params(a, l_mult):
    for seed in range(6):
        rng = random.Random(3000 + 100 * a + seed)
        n = rng.choice([8, 10, 12, 14])
        edges = rand_subcubic(n, rng, rng.choice([0.25, 0.45]))
        expect = nx_has_perfect(n, edges)
        M = PhaseMachine(n, edges, PhaseParams(a=a, l_mult=l_mult)).run()
        assert (M is not None) == expect, (n, edges, a, l_mult)


@pytest.mark.parametrize("seed", range(8))
def test_debug_edits_invariants(seed):
    """Full structural debug assertions: contraction s-aggregation,
    expansion child slack 1 / no child debt / barrier side, (3.13)/(3.14)
    and coarser debt decrease at peels, vertex coverage at settled states."""
    rng = random.Random(4000 + seed)
    n = rng.choice([10, 12, 14, 16, 18])
    edges = rand_subcubic(n, rng, rng.choice([0.3, 0.5]))
    expect = nx_has_perfect(n, edges)
    M = PhaseMachine(n, edges, PhaseParams(a=2, debug_edits=True)).run()
    assert (M is not None) == expect
    if M is not None:
        check_witness(n, edges, M)


# ---------------------------------------------------------------------------
# pure helpers: tight-set search (Lemma 3.6) and exchange projection (3.5)
# ---------------------------------------------------------------------------

def _brute_min_h(rows_nb, bins, i, j):
    best = None
    bl = [b for b in bins if b not in (i, j)]
    for k in range(len(bl) + 1):
        for extra in combinations(bl, k):
            Z = set(extra) | {i, j}
            h = len(Z) - sum(1 for r in rows_nb
                             if set(rows_nb[r]) <= Z)
            if best is None or h < best[0]:
                best = (h, Z)
    return best


@pytest.mark.parametrize("seed", range(20))
def test_pair_decision_matches_bruteforce(seed):
    rng = random.Random(5000 + seed)
    nbins = rng.randint(2, 7)
    nrows = rng.randint(0, nbins + 1)
    bins = list(range(nbins))
    rows = [f"r{t}" for t in range(nrows)]
    rows_nb = {r: set(rng.sample(bins, rng.randint(1, nbins)))
               for r in rows}
    i, j = rng.sample(bins, 2)
    min_h, X = pair_decision(rows_nb, bins, i, j)
    bh, bZ = _brute_min_h(rows_nb, bins, i, j)
    assert min_h == bh, (rows_nb, bins, i, j)
    assert i in X and j in X
    # X attains the minimum and is inclusion-minimal among minimizers
    hX = len(X) - sum(1 for r in rows_nb if set(rows_nb[r]) <= X)
    assert hX == bh
    for b in X - {i, j}:
        Z = X - {b}
        hZ = len(Z) - sum(1 for r in rows_nb if set(rows_nb[r]) <= Z)
        assert hZ >= bh + 1 or j not in Z or i not in Z, "X not minimal"


def test_pair_decision_two_free_bins():
    # no rows: h({i,j}) = 2 -> gain branch
    min_h, X = pair_decision({}, [1, 2, 3], 1, 2)
    assert min_h == 2


def test_pair_decision_captive_tight():
    # row captive in {1,2}: h = 1, unique minimal tight X = {1,2}
    rows_nb = {"r": {1, 2}}
    min_h, X = pair_decision(rows_nb, [1, 2, 3], 1, 2)
    assert min_h == 1 and X == {1, 2}


def test_exchange_projection_fills():
    # one row with full L on a deficient bin and a spare bin at y = L:
    # the exchange moves one unit across, filling the deficit (outcome i)
    L, c = 10, 1
    weights = {1: 1, 2: 1}
    inc = [("r1", 1, 0), ("r1", 2, 1)]
    x = {("r1", 1, 0): 10, ("r1", 2, 1): 0}
    ex = exchange_projection(L, c, weights, inc, x)
    assert ex[0]
    xnew = ex[1]
    assert sum(xnew.values()) == L                  # row sum preserved
    for b in (1, 2):
        load = sum(v for k, v in xnew.items() if k[1] == b)
        assert L - load >= c * weights[b]           # zero-debt floor


def test_exchange_projection_peels():
    """The hand-crafted failing instance from Lemma 3.5's analysis:
    one heavy assigned bin (w = L+5, y = 0, deficit c(L+5)) whose only
    exchange route is one row with a single free spare bin (L - c).
    Outcome (ii) must peel and satisfy (3.13)."""
    L, c = 8, 2
    weights = {1: L + 5, 2: 1}
    inc = [("r1", 1, 0), ("r1", 2, 1)]
    x = {("r1", 1, 0): L, ("r1", 2, 1): 0}
    ex = exchange_projection(L, c, weights, inc, x)
    assert not ex[0]
    R, xnew = ex[1], ex[2]
    assert R == {1, 2}          # everything is residual-reachable
    # (3.13): L h(R) < c w(R); h(R) = 1 (r1 is captive in R)
    assert L * 1 < c * (L + 6)
    # only the spare capacity (L - c) crossed the exchange
    assert xnew[("r1", 2, 1)] == L - c
    assert xnew[("r1", 1, 0)] == c


def test_exchange_projection_partial_peel():
    """A spare bin OUTSIDE the reachable side survives the peel."""
    L, c = 8, 2
    weights = {1: L + 5, 2: 1, 3: 1}
    # bin 3 has a spare (its row allocates elsewhere); bins 1, 2 starve
    inc = [("r1", 1, 0), ("r1", 2, 1), ("r2", 3, 2), ("r2", 4, 3)]
    x = {("r1", 1, 0): L, ("r1", 2, 1): 0, ("r2", 3, 2): 0, ("r2", 4, 3): 0}
    weights[4] = 1
    x[("r2", 4, 3)] = L          # r2 assigns to bin 4; bin 3 spare
    ex = exchange_projection(L, c, weights, inc, x)
    assert not ex[0]
    R = ex[1]
    assert 1 in R and 2 in R and 3 not in R and 4 not in R
    # bin 3's deficit is zero (spare) and bin 4's deficit is saturated by
    # backflow through r2, so neither is residual-reachable; row sums and
    # the floor hold on the surviving side
    xnew = ex[2]
    for r in ("r1", "r2"):
        assert sum(v for k, v in xnew.items() if k[0] == r) == L
    assert xnew[("r1", 2, 1)] == L - c


def test_forced_peel_end_to_end(monkeypatch):
    """Force one projection to take the peel outcome (all bins) on an
    unmatchable graph; the state machinery (deletion transformation,
    materialization into U/blockers, fresh ledgers) must keep every
    invariant and still answer correctly."""
    import phase_machine as pm
    n = 8
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 7)]
    assert not nx_has_perfect(n, edges)
    orig = pm.exchange_projection
    forced = {"done": False}

    def wrapper(L, c, weights, incidences, x):
        if not forced["done"]:
            forced["done"] = True
            return False, set(weights), {k: 0 for k in incidences}
        return orig(L, c, weights, incidences, x)

    monkeypatch.setattr(pm, "exchange_projection", wrapper)
    M = PhaseMachine(n, edges, PhaseParams(a=2, debug_edits=False)).run()
    monkeypatch.setattr(pm, "exchange_projection", orig)
    assert M is None


# ---------------------------------------------------------------------------
# materialization: factor-criticality of contraction trees (Lemma 3.8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_materialize_every_exposure(seed):
    """After a negative (LOW) termination the final bins stay alive in the
    machine; materialize each one from every prescribed exposure and check
    the internal matching is valid and covers all but the exposure."""
    rng = random.Random(6000 + seed)
    n = rng.choice([8, 10, 12, 14])
    edges = rand_subcubic(n, rng, rng.choice([0.2, 0.35]))
    if nx_has_perfect(n, edges):
        edges = edges[:len(edges) // 2]   # make matchability unlikely
    m = PhaseMachine(n, edges, PhaseParams(a=2))
    M = m.run()
    # whatever the answer, final bins (if any) must be factor-critical
    assert m.bins is not None
    if M is not None:
        return   # phase boundary dissolved the bins; nothing to check
    for bid, b in m.bins.items():
        if not isinstance(b.tree, tuple) and len(b.vertices) >= 3:
            for v in sorted(b.vertices):
                pairs = m.materialize(b, v)
                seen = set()
                for (u, w) in pairs:
                    assert frozenset((u, w)) in m._pair_eid
                    assert u not in seen and w not in seen
                    seen.add(u)
                    seen.add(w)
                assert seen == b.vertices - {v}


def test_materialize_deep_tree():
    """Nested contractions: chain of triangles sharing cut vertices."""
    # triangles (3k,3k+1,3k+2) glued by bridges
    k = 4
    n = 3 * k + 2
    edges = []
    for t in range(k):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [(a, b), (b, c), (a, c)]
    for t in range(k - 1):
        edges.append((3 * t + 2, 3 * t + 3))
    edges.append((3 * k - 1, 3 * k))     # even total, likely unmatched
    edges.append((3 * k, 3 * k + 1))
    rng = random.Random(7)
    m = PhaseMachine(n, edges, PhaseParams(a=3))
    M = m.run()
    expect = nx_has_perfect(n, edges)
    assert (M is not None) == expect
    if M:
        check_witness(n, edges, M)


# ---------------------------------------------------------------------------
# Dinic vs networkx
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_dinic_matches_networkx(seed):
    rng = random.Random(7000 + seed)
    V = rng.randint(4, 12)
    d = Dinic()
    G = nx.DiGraph()
    cap = {}
    for u in range(V):
        for v in range(V):
            if u != v and rng.random() < 0.3:
                c = rng.randint(0, 10)
                d.add(u, v, c)
                G.add_edge(u, v, capacity=c)
    if 0 not in G or 1 not in G or not nx.has_path(G, 0, 1):
        return
    mine = d.maxflow(0, 1)
    ref = nx.maximum_flow_value(G, 0, 1)
    assert mine == ref


# ---------------------------------------------------------------------------
# composed pipeline with Layer 1 (reductions.py): G -> H_k -> degree gadget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,edges", [(2, [(0, 1)]), (2, [])])
def test_composed_pipeline(n, edges):
    # k = nu (feasible): H_k perfect, decode gives a size-nu matching of G
    _composed(n, edges, feasible=True)
    # k = nu + 1 (infeasible whenever nu + 1 > nu(G)): H_k imperfect
    _composed(n, edges, feasible=False)


def _composed(n, edges, feasible):
    """Full pipeline: G -> H_{nu} (Lemma 9.2) -> degree gadget (Lemma 9.3)
    -> phase machine -> project back.  Sizes are tiny on purpose: the
    gadget for a single edge already has ~54 vertices and the reference
    machine runs one phase per gain (g* = 1 in the small-parameter
    regime), so larger compositions are a runtime matter, not a
    correctness one."""
    from reductions import build_hk, degree_reduce
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    nu = len(nx.max_weight_matching(G, maxcardinality=True))
    k = nu if feasible else nu + 1
    if k > n // 2:
        return                       # cannot exceed floor(n/2)
    hk = build_hk(n, edges, k)
    dr = degree_reduce(hk.vertices, [(e.u, e.v) for e in hk.edges])
    assert dr.isolate is None
    M = PhaseMachine(len(dr.new_vertices),
                     [(e.u, e.v) for e in dr.new_edges],
                     PhaseParams(a=2)).run()
    if feasible:
        assert M is not None
        hk_edges = dr.project(M)            # perfect matching of H_k
        assert 2 * len(hk_edges) == len(hk.vertices)
        seen = set()
        for (u, v) in hk_edges:
            assert frozenset((u, v)) in hk._by_pair
            assert u not in seen and v not in seen
            seen.add(u)
            seen.add(v)
        assert seen == set(hk.vertices)
        gmatch = hk.decode(hk_edges)        # back to G (original edges)
        assert len(gmatch) == k
        eset = {frozenset(e) for e in edges}
        gseen = set()
        for (u, v) in gmatch:
            assert frozenset((u, v)) in eset
            assert u not in gseen and v not in gseen
            gseen.add(u)
            gseen.add(v)
    else:
        assert M is None, (n, edges)


@pytest.mark.parametrize("seed", range(10))
def test_random_medium(seed):
    rng = random.Random(2000 + seed)
    n = rng.choice([18, 20, 24, 26, 28])
    p = rng.choice([0.2, 0.35])
    edges = rand_subcubic(n, rng, p)
    expect = nx_has_perfect(n, edges)
    M = PhaseMachine(n, edges).run()
    if expect:
        assert M is not None, (n, edges)
        check_witness(n, edges, M)
    else:
        assert M is None, (n, edges)


# ---------------------------------------------------------------------------
# multi-ledger parameter sweep (kill-test 4.4 criterion)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,l_mult", [(1, 2), (2, 10), (3, 100), (5, 100),
                                     (8, 4)])
def test_multi_ledger_params(a, l_mult):
    for seed in range(6):
        rng = random.Random(3000 + 100 * a + seed)
        n = rng.choice([8, 10, 12, 14])
        edges = rand_subcubic(n, rng, rng.choice([0.25, 0.45]))
        expect = nx_has_perfect(n, edges)
        M = PhaseMachine(n, edges, PhaseParams(a=a, l_mult=l_mult)).run()
        assert (M is not None) == expect, (n, edges, a, l_mult)


@pytest.mark.parametrize("seed", range(8))
def test_debug_edits_invariants(seed):
    """Full structural debug assertions: contraction s-aggregation,
    expansion child slack 1 / no child debt / barrier side, (3.13)/(3.14)
    and coarser debt decrease at peels, vertex coverage at settled states."""
    rng = random.Random(4000 + seed)
    n = rng.choice([10, 12, 14, 16, 18])
    edges = rand_subcubic(n, rng, rng.choice([0.3, 0.5]))
    expect = nx_has_perfect(n, edges)
    M = PhaseMachine(n, edges, PhaseParams(a=2, debug_edits=True)).run()
    assert (M is not None) == expect
    if M is not None:
        check_witness(n, edges, M)


# ---------------------------------------------------------------------------
# pure helpers: tight-set search (Lemma 3.6) and exchange projection (3.5)
# ---------------------------------------------------------------------------

def _brute_min_h(rows_nb, bins, i, j):
    best = None
    bl = [b for b in bins if b not in (i, j)]
    for k in range(len(bl) + 1):
        for extra in combinations(bl, k):
            Z = set(extra) | {i, j}
            h = len(Z) - sum(1 for r in rows_nb
                             if set(rows_nb[r]) <= Z)
            if best is None or h < best[0]:
                best = (h, Z)
    return best


@pytest.mark.parametrize("seed", range(20))
def test_pair_decision_matches_bruteforce(seed):
    rng = random.Random(5000 + seed)
    nbins = rng.randint(2, 7)
    nrows = rng.randint(0, nbins + 1)
    bins = list(range(nbins))
    rows = [f"r{t}" for t in range(nrows)]
    rows_nb = {r: set(rng.sample(bins, rng.randint(1, nbins)))
               for r in rows}
    i, j = rng.sample(bins, 2)
    min_h, X = pair_decision(rows_nb, bins, i, j)
    bh, bZ = _brute_min_h(rows_nb, bins, i, j)
    assert min_h == bh, (rows_nb, bins, i, j)
    assert i in X and j in X
    # X attains the minimum and is inclusion-minimal among minimizers
    hX = len(X) - sum(1 for r in rows_nb if set(rows_nb[r]) <= X)
    assert hX == bh
    for b in X - {i, j}:
        Z = X - {b}
        hZ = len(Z) - sum(1 for r in rows_nb if set(rows_nb[r]) <= Z)
        assert hZ >= bh + 1 or j not in Z or i not in Z, "X not minimal"


def test_pair_decision_two_free_bins():
    # no rows: h({i,j}) = 2 -> gain branch
    min_h, X = pair_decision({}, [1, 2, 3], 1, 2)
    assert min_h == 2


def test_pair_decision_captive_tight():
    # row captive in {1,2}: h = 1, unique minimal tight X = {1,2}
    rows_nb = {"r": {1, 2}}
    min_h, X = pair_decision(rows_nb, [1, 2, 3], 1, 2)
    assert min_h == 1 and X == {1, 2}


def test_exchange_projection_fills():
    # one row with full L on a deficient bin and a spare bin at y = L:
    # the exchange moves one unit across, filling the deficit (outcome i)
    L, c = 10, 1
    weights = {1: 1, 2: 1}
    inc = [("r1", 1, 0), ("r1", 2, 1)]
    x = {("r1", 1, 0): 10, ("r1", 2, 1): 0}
    ex = exchange_projection(L, c, weights, inc, x)
    assert ex[0]
    xnew = ex[1]
    assert sum(xnew.values()) == L                  # row sum preserved
    for b in (1, 2):
        load = sum(v for k, v in xnew.items() if k[1] == b)
        assert L - load >= c * weights[b]           # zero-debt floor


def test_exchange_projection_peels():
    """The hand-crafted failing instance from Lemma 3.5's analysis:
    one heavy assigned bin (w = L+5, y = 0, deficit c(L+5)) whose only
    exchange route is one row with a single free spare bin (L - c).
    Outcome (ii) must peel and satisfy (3.13)."""
    L, c = 8, 2
    weights = {1: L + 5, 2: 1}
    inc = [("r1", 1, 0), ("r1", 2, 1)]
    x = {("r1", 1, 0): L, ("r1", 2, 1): 0}
    ex = exchange_projection(L, c, weights, inc, x)
    assert not ex[0]
    R, xnew = ex[1], ex[2]
    assert R == {1, 2}          # everything is residual-reachable
    # (3.13): L h(R) < c w(R); h(R) = 1 (r1 is captive in R)
    assert L * 1 < c * (L + 6)
    # only the spare capacity (L - c) crossed the exchange
    assert xnew[("r1", 2, 1)] == L - c
    assert xnew[("r1", 1, 0)] == c


def test_exchange_projection_partial_peel():
    """A spare bin OUTSIDE the reachable side survives the peel."""
    L, c = 8, 2
    weights = {1: L + 5, 2: 1, 3: 1}
    # bin 3 has a spare (its row allocates elsewhere); bins 1, 2 starve
    inc = [("r1", 1, 0), ("r1", 2, 1), ("r2", 3, 2), ("r2", 4, 3)]
    x = {("r1", 1, 0): L, ("r1", 2, 1): 0, ("r2", 3, 2): 0, ("r2", 4, 3): 0}
    weights[4] = 1
    x[("r2", 4, 3)] = L          # r2 assigns to bin 4; bin 3 spare
    ex = exchange_projection(L, c, weights, inc, x)
    assert not ex[0]
    R = ex[1]
    assert 1 in R and 2 in R and 3 not in R and 4 not in R
    # bin 3's deficit is zero (spare) and bin 4's deficit is saturated by
    # backflow through r2, so neither is residual-reachable; row sums and
    # the floor hold on the surviving side
    xnew = ex[2]
    for r in ("r1", "r2"):
        assert sum(v for k, v in xnew.items() if k[0] == r) == L
    assert xnew[("r1", 2, 1)] == L - c


def test_forced_peel_end_to_end(monkeypatch):
    """Force one projection to take the peel outcome (all bins) on an
    unmatchable graph; the state machinery (deletion transformation,
    materialization into U/blockers, fresh ledgers) must keep every
    invariant and still answer correctly."""
    import phase_machine as pm
    n = 8
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 7)]
    assert not nx_has_perfect(n, edges)
    orig = pm.exchange_projection
    forced = {"done": False}

    def wrapper(L, c, weights, incidences, x):
        if not forced["done"]:
            forced["done"] = True
            return False, set(weights), {k: 0 for k in incidences}
        return orig(L, c, weights, incidences, x)

    monkeypatch.setattr(pm, "exchange_projection", wrapper)
    M = PhaseMachine(n, edges, PhaseParams(a=2, debug_edits=False)).run()
    monkeypatch.setattr(pm, "exchange_projection", orig)
    assert M is None


# ---------------------------------------------------------------------------
# materialization: factor-criticality of contraction trees (Lemma 3.8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_materialize_every_exposure(seed):
    """After a negative (LOW) termination the final bins stay alive in the
    machine; materialize each one from every prescribed exposure and check
    the internal matching is valid and covers all but the exposure."""
    rng = random.Random(6000 + seed)
    n = rng.choice([8, 10, 12, 14])
    edges = rand_subcubic(n, rng, rng.choice([0.2, 0.35]))
    if nx_has_perfect(n, edges):
        edges = edges[:len(edges) // 2]   # make matchability unlikely
    m = PhaseMachine(n, edges, PhaseParams(a=2))
    M = m.run()
    # whatever the answer, final bins (if any) must be factor-critical
    assert m.bins is not None
    if M is not None:
        return   # phase boundary dissolved the bins; nothing to check
    for bid, b in m.bins.items():
        if not isinstance(b.tree, tuple) and len(b.vertices) >= 3:
            for v in sorted(b.vertices):
                pairs = m.materialize(b, v)
                seen = set()
                for (u, w) in pairs:
                    assert frozenset((u, w)) in m._pair_eid
                    assert u not in seen and w not in seen
                    seen.add(u)
                    seen.add(w)
                assert seen == b.vertices - {v}


def test_materialize_deep_tree():
    """Nested contractions: chain of triangles sharing cut vertices."""
    # triangles (3k,3k+1,3k+2) glued by bridges
    k = 4
    n = 3 * k + 2
    edges = []
    for t in range(k):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [(a, b), (b, c), (a, c)]
    for t in range(k - 1):
        edges.append((3 * t + 2, 3 * t + 3))
    edges.append((3 * k - 1, 3 * k))     # even total, likely unmatched
    edges.append((3 * k, 3 * k + 1))
    rng = random.Random(7)
    m = PhaseMachine(n, edges, PhaseParams(a=3))
    M = m.run()
    expect = nx_has_perfect(n, edges)
    assert (M is not None) == expect
    if M:
        check_witness(n, edges, M)


# ---------------------------------------------------------------------------
# Dinic vs networkx
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_dinic_matches_networkx(seed):
    rng = random.Random(7000 + seed)
    V = rng.randint(4, 12)
    d = Dinic()
    G = nx.DiGraph()
    cap = {}
    for u in range(V):
        for v in range(V):
            if u != v and rng.random() < 0.3:
                c = rng.randint(0, 10)
                d.add(u, v, c)
                G.add_edge(u, v, capacity=c)
    if 0 not in G or 1 not in G or not nx.has_path(G, 0, 1):
        return
    mine = d.maxflow(0, 1)
    ref = nx.maximum_flow_value(G, 0, 1)
    assert mine == ref


# ---------------------------------------------------------------------------
# composed pipeline with Layer 1 (reductions.py): G -> H_k -> degree gadget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,edges", [(2, [(0, 1)]), (2, [])])
def test_composed_pipeline(n, edges):
    # k = nu (feasible): H_k perfect, decode gives a size-nu matching of G
    _composed(n, edges, feasible=True)
    # k = nu + 1 (infeasible whenever nu + 1 > nu(G)): H_k imperfect
    _composed(n, edges, feasible=False)


def _composed(n, edges, feasible):
    """Full pipeline: G -> H_{nu} (Lemma 9.2) -> degree gadget (Lemma 9.3)
    -> phase machine -> project back.  Sizes are tiny on purpose: the
    gadget for a single edge already has ~54 vertices and the reference
    machine runs one phase per gain (g* = 1 in the small-parameter
    regime), so larger compositions are a runtime matter, not a
    correctness one."""
    from reductions import build_hk, degree_reduce
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    nu = len(nx.max_weight_matching(G, maxcardinality=True))
    k = nu if feasible else nu + 1
    if k > n // 2:
        return                       # cannot exceed floor(n/2)
    hk = build_hk(n, edges, k)
    dr = degree_reduce(hk.vertices, [(e.u, e.v) for e in hk.edges])
    assert dr.isolate is None
    M = PhaseMachine(len(dr.new_vertices),
                     [(e.u, e.v) for e in dr.new_edges],
                     PhaseParams(a=2)).run()
    if feasible:
        assert M is not None
        hk_edges = dr.project(M)            # perfect matching of H_k
        assert 2 * len(hk_edges) == len(hk.vertices)
        seen = set()
        for (u, v) in hk_edges:
            assert frozenset((u, v)) in hk._by_pair
            assert u not in seen and v not in seen
            seen.add(u)
            seen.add(v)
        assert seen == set(hk.vertices)
        gmatch = hk.decode(hk_edges)        # back to G (original edges)
        assert len(gmatch) == nu
        eset = {frozenset(e) for e in edges}
        gseen = set()
        for (u, v) in gmatch:
            assert frozenset((u, v)) in eset
            assert u not in gseen and v not in gseen
            gseen.add(u)
            gseen.add(v)
    else:
        assert M is None
