#!/usr/bin/env python3
"""Benchmark: DynamicMatching (Tier 2 m1) vs full-recompute (networkx) baseline.

n=2000 random graph, 500 mixed updates; reports mean per-update time for both
approaches plus matching churn (edges of the matching changed per update).

Run:  python3 benchmarks/bench_dynamic.py
"""

import random
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx

from dynamic import DynamicMatching


def build_edges(n, m, rng):
    edges = set()
    while len(edges) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def main(n=2000, m_target=8000, updates=500, seed=12345):
    rng = random.Random(seed)
    edges = build_edges(n, m_target, rng)

    # --- dynamic run ---
    dm = DynamicMatching(n)
    t0 = time.perf_counter()
    for u, v in edges:
        dm.insert(u, v)
    build_time = time.perf_counter() - t0

    # workload: mixed inserts/deletes, biased to keep |E| near m_target
    dm.stats = {"updates": 0, "searches": 0, "augmentations": 0, "work": 0}
    present = set(edges)
    absent = []
    ops = []
    for _ in range(updates):
        p_del = 0.5 if len(present) > m_target else 0.25
        if rng.random() < p_del and present:
            e = rng.choice(sorted(present))
            ops.append(("d", e))
            present.discard(e)
            absent.append(e)
        else:
            for _ in range(50):
                u, v = rng.randrange(n), rng.randrange(n)
                if u != v:
                    e = (min(u, v), max(u, v))
                    if e not in present:
                        break
            else:
                continue
            ops.append(("i", e))
            present.add(e)
            if e in absent:
                absent.remove(e)

    churn = []
    t0 = time.perf_counter()
    for op, e in ops:
        before = set(dm.matching())
        if op == "i":
            dm.insert(*e)
        else:
            dm.delete(*e)
        after = set(dm.matching())
        churn.append(len(before ^ after))
    dyn_time = time.perf_counter() - t0
    dyn_searches = dm.stats["searches"]
    dyn_aug = dm.stats["augmentations"]

    # --- recompute baseline: same ops, full networkx solve after each ---
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    base_churn = []
    t0 = time.perf_counter()
    for op, e in ops:
        prev = set(frozenset((u, v)) for u, v in nx.max_weight_matching(g, maxcardinality=True))
        if op == "i":
            g.add_edge(*e)
        else:
            g.remove_edge(*e)
        cur = set(frozenset((u, v)) for u, v in nx.max_weight_matching(g, maxcardinality=True))
        base_churn.append(len(prev ^ cur))
    base_time = time.perf_counter() - t0

    # sanity: final sizes agree
    final_dyn = dm.size()
    final_base = len(nx.max_weight_matching(g, maxcardinality=True))

    print(f"graph: n={n}, m={len(edges)} (initial), {len(ops)} updates, seed={seed}")
    print(f"initial build (dynamic, {len(edges)} inserts): {build_time*1000:.0f} ms total, "
          f"{build_time/len(edges)*1e6:.0f} us/insert")
    print()
    print(f"{'approach':<28}{'mean us/update':>16}{'total s':>10}{'churn (edges changed/update)':>34}")
    print(f"{'-'*88}")
    print(f"{'DynamicMatching (ours)':<28}{dyn_time/len(ops)*1e6:>16.0f}{dyn_time:>10.2f}"
          f"{sum(churn)/len(churn):>22.2f} (mean){'':6}max={max(churn)}")
    print(f"{'networkx full recompute':<28}{base_time/len(ops)*1e6:>16.0f}{base_time:>10.2f}"
          f"{sum(base_churn)/len(base_churn):>22.2f} (mean){'':6}max={max(base_churn)}")
    print()
    print(f"speedup (mean per-update): {base_time/dyn_time:.1f}x")
    print(f"dynamic searches (update phase): {dyn_searches} over {len(ops)} updates "
          f"({dyn_searches/len(ops):.2f}/update), augmentations: {dyn_aug}")
    print(f"final matching size: dynamic={final_dyn}, oracle={final_base} "
          f"{'OK' if final_dyn == final_base else 'MISMATCH!'}")
    assert final_dyn == final_base
    assert dm.is_valid()


if __name__ == "__main__":
    main()
