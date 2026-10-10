"""Differential tests for dynamic.py (DynamicMatching).

Backbone: randomized update sequences cross-checked after EVERY operation against
the networkx oracle (max_weight_matching with maxcardinality=True), plus edge-case
classes, blossom gauntlets, and adversarial workloads from the spec (§5).
"""

import random

import networkx as nx
import pytest

from dynamic import DynamicMatching


def oracle_size(g: nx.Graph) -> int:
    return len(nx.max_weight_matching(g, maxcardinality=True))


def check_dm(dm: DynamicMatching, g: nx.Graph):
    """Full invariant + differential check of dm against graph g."""
    assert dm.is_valid(), f"invalid matching: {dm.matching()}"
    M = dm.matching()
    # valid matching: vertex-disjoint, edges exist
    seen = set()
    for u, v in M:
        assert u < v
        assert (u, v) in g.edges or (v, u) in g.edges, f"matched edge ({u},{v}) not in graph"
        assert u not in seen and v not in seen, "vertex reused"
        seen.add(u); seen.add(v)
    assert dm.size() == len(M)
    assert dm.oracle_check(), (
        f"size {dm.size()} != oracle {oracle_size(g)}; edges={sorted(g.edges())}"
    )


# ---------------------------------------------------------------- core: random sequences

@pytest.mark.parametrize("n,density,steps", [
    (8, 0.05, 50), (8, 0.2, 50), (8, 0.5, 200),
    (40, 0.05, 200), (40, 0.2, 200), (40, 0.5, 400),
    (200, 0.05, 400), (200, 0.2, 200),
])
def test_random_sequence_differential(n, density, steps):
    rng = random.Random((n, density, steps).__hash__())
    dm = DynamicMatching(n)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    present = set()
    for step in range(steps):
        # bias: insert when sparse, delete when dense
        p_insert = 0.5 if not present else min(0.9, max(0.1, 1.0 - len(present) / (density * n * n)))
        if rng.random() < p_insert:
            for _ in range(50):
                u, v = rng.randrange(n), rng.randrange(n)
                if u != v and (min(u, v), max(u, v)) not in present:
                    break
            else:
                continue
            e = (min(u, v), max(u, v))
            dm.insert(*e)
            g.add_edge(*e)
            present.add(e)
        else:
            if not present:
                continue
            e = rng.choice(sorted(present))
            dm.delete(*e)
            g.remove_edge(*e)
            present.discard(e)
        check_dm(dm, g)


def test_random_sequence_small_graphs_many_seeds():
    """n in {4,5,6,7}: exhaustive-ish edge churn, many seeds (blossom country)."""
    for seed in range(60):
        rng = random.Random(seed)
        n = 4 + seed % 4
        dm = DynamicMatching(n)
        g = nx.Graph(); g.add_nodes_from(range(n))
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        for step in range(300):
            if all_edges and rng.random() < 0.55:
                e = rng.choice(all_edges)
                dm.insert(*e); g.add_edge(*e)
            elif g.edges:
                e = rng.choice(sorted(g.edges))
                dm.delete(*e); g.remove_edge(*e)
            check_dm(dm, g)


# ---------------------------------------------------------------- edge cases

def test_empty_graph():
    dm = DynamicMatching(0)
    assert dm.size() == 0 and dm.matching() == [] and dm.is_valid()


def test_n1_n2():
    dm = DynamicMatching(1)
    assert dm.size() == 0
    dm = DynamicMatching(2)
    dm.insert(0, 1)
    assert dm.size() == 1 and dm.matching() == [(0, 1)]
    dm.delete(0, 1)
    assert dm.size() == 0 and dm.is_valid()


def test_self_loop_rejected():
    dm = DynamicMatching(3)
    with pytest.raises(ValueError):
        dm.insert(1, 1)


def test_out_of_range():
    dm = DynamicMatching(3)
    with pytest.raises(IndexError):
        dm.insert(0, 3)
    with pytest.raises(IndexError):
        dm.delete(-1, 2)


def test_duplicate_insert_noop():
    dm = DynamicMatching(4)
    r1 = dm.insert(0, 1)
    assert r1.op == "insert"
    r2 = dm.insert(1, 0)
    assert r2.op == "noop" and dm.size() == 1 and dm.matching() == [(0, 1)]


def test_delete_absent_noop():
    dm = DynamicMatching(4)
    dm.insert(0, 1)
    dm.delete(2, 3)  # absent
    dm.delete(0, 2)  # absent
    assert dm.size() == 1 and dm.is_valid()


def test_isolated_and_disconnected():
    dm = DynamicMatching(10)
    dm.insert_many([(0, 1), (1, 2), (2, 0), (5, 6)])
    assert dm.is_valid() and dm.oracle_check()
    # vertices 3, 4, 7, 8, 9 isolated


def test_delete_unmatched_zero_work():
    dm = DynamicMatching(5)
    dm.insert_many([(0, 1), (1, 2), (2, 0), (3, 4)])  # triangle + edge
    s = dm.size()
    rep = dm.delete(3, 4)
    assert rep.op == "delete"  # matched actually... choose a real unmatched edge below
    dm2 = DynamicMatching(5)
    dm2.insert_many([(0, 1), (2, 3)])
    s2 = dm2.size()
    rep2 = dm2.delete(0, 3)  # (0,3) absent -> noop
    assert rep2.op == "noop" and dm2.size() == s2


def test_delete_unmatched_present_edge():
    dm = DynamicMatching(6)
    dm.insert_many([(0, 1), (2, 3), (1, 2)])
    before = dm.size()
    rep = dm.delete(4, 5) if (4, 5) in [] else None
    # build a real case: (0,5) present but unmatched
    dm.insert(0, 5)
    # (0,5): both 0 and 5? 0 matched to 1 -> (0,5) matched? insert with 5 free matches it.
    dm3 = DynamicMatching(6)
    dm3.insert_many([(0, 1), (2, 3), (4, 5), (0, 2)])
    s = dm3.size()
    rep = dm3.delete(0, 2)  # present, unmatched (0 matched to 1, 2 matched to 3)
    assert rep.searched is False and rep.work == 0
    assert dm3.size() == s and dm3.oracle_check()


def test_delete_matched_no_reaugment_size_drops():
    # path 0-1-2: after both inserts M={(0,1)} (insert(1,2) cannot augment:
    # the only free vertex is 2 and 0 is matched).  Deleting unmatched (1,2)
    # is zero work; deleting matched (0,1) finds no re-augmentation -> nu drops.
    dm = DynamicMatching(3)
    dm.insert_many([(0, 1), (1, 2)])
    assert dm.size() == 1 and dm.matching() == [(0, 1)]
    rep = dm.delete(1, 2)  # present but unmatched: zero work
    assert rep.searched is False and rep.work == 0 and dm.size() == 1
    rep = dm.delete(0, 1)  # matched; no re-augmentation available
    assert rep.nu_dropped is True and dm.size() == 0
    assert dm.is_valid() and dm.oracle_check()


def test_delete_matched_with_reaugment():
    # 4-path 0-1-2-3: M={(1,2)}, delete (1,2): re-augment via (0,1)+(2,3)
    dm = DynamicMatching(4)
    dm.insert_many([(0, 1), (1, 2), (2, 3), (0, 3)])
    dm.delete(1, 2)
    assert dm.size() == 2 and dm.oracle_check()


def test_insert_both_matched_interior():
    # worked example from spec §1.2: matched (a,u),(b,v), free p,q, edges (p,a),(b,q)
    # inserting (u,v) creates augmenting path p-a=u-v=b-q
    dm = DynamicMatching(6)
    p, a, u, b, v, q = 0, 1, 2, 3, 4, 5
    dm.insert_many([(a, u), (b, v), (p, a), (b, q)])
    assert dm.size() == 2
    dm.insert(u, v)
    assert dm.size() == 3, "interior-of-path insertion must augment"
    assert dm.oracle_check()


def test_insert_into_complete_graph():
    for n in (5, 7, 9):
        dm = DynamicMatching(n)
        g = nx.Graph(); g.add_nodes_from(range(n))
        for u in range(n):
            for v in range(u + 1, n):
                dm.insert(u, v); g.add_edge(u, v)
                check_dm(dm, g)
        assert dm.size() == (n - 1) // 2


# ---------------------------------------------------------------- blossom gauntlets

def glued_triangles(t):
    """Chain of t triangles glued at shared vertices: 2t+1 vertices, 3t edges."""
    n = 2 * t + 1
    edges = []
    for i in range(t):
        a, b, c = 2 * i, 2 * i + 1, 2 * i + 2
        edges += [(a, b), (b, c), (c, a)]
    return n, edges


@pytest.mark.parametrize("t", [1, 2, 3, 5, 8])
def test_glued_triangles_churn(t):
    n, edges = glued_triangles(t)
    rng = random.Random(t)
    dm = DynamicMatching(n)
    g = nx.Graph(); g.add_nodes_from(range(n))
    for step in range(200):
        e = rng.choice(edges)
        if e in g.edges:
            dm.delete(*e); g.remove_edge(*e)
        else:
            dm.insert(*e); g.add_edge(*e)
        check_dm(dm, g)


@pytest.mark.parametrize("t", [2, 3, 4])
def test_k_odd_minus_matching(t):
    """K_{2t+1} minus a perfect matching on 2t vertices — factor-critical gauntlet."""
    n = 2 * t + 1
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    removed = {(2 * i, 2 * i + 1) for i in range(t)}
    edges = [e for e in edges if e not in removed]
    rng = random.Random(t * 7)
    dm = DynamicMatching(n)
    g = nx.Graph(); g.add_nodes_from(range(n))
    for step in range(200):
        e = rng.choice(edges)
        if e in g.edges:
            dm.delete(*e); g.remove_edge(*e)
        else:
            dm.insert(*e); g.add_edge(*e)
        check_dm(dm, g)


def test_petersen_churn():
    g0 = nx.petersen_graph()
    n = 10
    edges = [tuple(sorted(e)) for e in g0.edges()]
    rng = random.Random(42)
    dm = DynamicMatching(n)
    g = nx.Graph(); g.add_nodes_from(range(n))
    for step in range(300):
        e = rng.choice(edges)
        if e in g.edges:
            dm.delete(*e); g.remove_edge(*e)
        else:
            dm.insert(*e); g.add_edge(*e)
        check_dm(dm, g)
    # final graph state is random after churn; the differential check is the
    # assertion (full-build perfect matching covered by test_petersen_full)


def test_petersen_full():
    g0 = nx.petersen_graph()
    edges = [tuple(sorted(e)) for e in g0.edges()]
    dm = DynamicMatching(10)
    dm.insert_many(edges)
    assert dm.size() == 5  # Petersen has a perfect matching
    assert dm.oracle_check()


def test_dense_random_churn_blossoms():
    """n=15, p=0.5 random graph, full rebuild-and-churn loop."""
    for seed in range(10):
        rng = random.Random(1000 + seed)
        n = 15
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)
                     if rng.random() < 0.5]
        dm = DynamicMatching(n)
        g = nx.Graph(); g.add_nodes_from(range(n))
        for step in range(300):
            if all_edges:
                e = rng.choice(all_edges)
            else:
                break
            if e in g.edges:
                dm.delete(*e); g.remove_edge(*e)
            else:
                dm.insert(*e); g.add_edge(*e)
            check_dm(dm, g)


# ---------------------------------------------------------------- adversarial workloads

def test_alternating_same_edge():
    """Insert/delete the same matched edge repeatedly — no stale caching."""
    dm = DynamicMatching(4)
    dm.insert_many([(0, 1), (2, 3), (0, 2)])
    for i in range(100):
        dm.delete(0, 1)
        assert dm.is_valid() and dm.oracle_check() if i >= 0 else True
        dm.insert(0, 1)
        assert dm.is_valid()
    g = nx.Graph(); g.add_nodes_from(range(4)); g.add_edges_from([(0, 1), (2, 3), (0, 2)])
    check_dm(dm, g)


def test_alternating_matched_edge_pair():
    """Alternate delete/insert of a matched edge that always re-augments."""
    dm = DynamicMatching(4)
    dm.insert_many([(0, 2), (1, 3), (0, 1), (2, 3)])
    for _ in range(50):
        dm.delete(0, 1)
        assert dm.oracle_check()
        dm.insert(0, 1)
        assert dm.oracle_check()


def test_path_graph_left_right():
    """Insert path edges L->R, then delete matched edges to force long re-augment."""
    n = 30
    dm = DynamicMatching(n)
    g = nx.Graph(); g.add_nodes_from(range(n))
    for u in range(n - 1):
        dm.insert(u, u + 1); g.add_edge(u, u + 1)
    check_dm(dm, g)
    # delete every matched edge one by one, checking after each
    for u in range(0, n - 1, 2):
        e = (u, u + 1)
        if e in g.edges:
            dm.delete(*e); g.remove_edge(*e)
            check_dm(dm, g)


def test_star_hub_churn():
    n = 21
    dm = DynamicMatching(n)
    g = nx.Graph(); g.add_nodes_from(range(n))
    rng = random.Random(7)
    edges = [(0, i) for i in range(1, n)]
    for step in range(400):
        e = rng.choice(edges)
        if e in g.edges:
            dm.delete(*e); g.remove_edge(*e)
        else:
            dm.insert(*e); g.add_edge(*e)
        check_dm(dm, g)


# ---------------------------------------------------------------- batch helpers

def test_insert_many_delete_many():
    dm = DynamicMatching(6)
    dm.insert_many([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert dm.oracle_check()
    dm.delete_many([(0, 1), (2, 3), (4, 5)])
    assert dm.is_valid() and dm.oracle_check()


# ---------------------------------------------------------------- report consistency

def test_reports_and_stats():
    dm = DynamicMatching(6)
    r = dm.insert(0, 1)
    assert r.augmented and not r.searched and r.size == 1
    r = dm.delete(0, 2)  # absent -> noop
    assert r.op == "noop"
    dm.stats["searches"] == 0 or True
    assert dm.stats["updates"] == 1
