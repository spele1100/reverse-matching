#!/usr/bin/env python3
"""Benchmark: DynamicMatching (Tier 2) vs full-recompute (networkx) baseline.

Builds a random graph, applies a mixed insert/delete workload biased to keep
|E| near target, and reports mean per-update time for both approaches plus
matching churn (edges of the matching changed per update) — the stability
headline: recompute rewrites O(n) matched edges per update, repair touches <1.

Run:
    python -m benchmarks.bench_dynamic                  # defaults n=2000
    python -m benchmarks.bench_dynamic --n 500 --updates 200
    python -m benchmarks.bench_dynamic --markdown       # markdown table output
"""

import argparse
import os
import random
import sys
import time

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


def make_workload(n, edges, m_target, updates, rng):
    """Mixed insert/delete ops biased to keep |E| near m_target."""
    present = set(edges)
    ops = []
    for _ in range(updates):
        p_del = 0.5 if len(present) > m_target else 0.25
        if rng.random() < p_del and present:
            e = rng.choice(sorted(present))
            ops.append(("d", e))
            present.discard(e)
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
    return ops


def run_benchmark(n, m_target, updates, seed):
    rng = random.Random(seed)
    edges = build_edges(n, m_target, rng)
    ops = make_workload(n, edges, m_target, updates, rng)

    # --- dynamic run ---
    dm = DynamicMatching(n)
    t0 = time.perf_counter()
    for u, v in edges:
        dm.insert(u, v)
    build_time = time.perf_counter() - t0

    churn = []
    t0 = time.perf_counter()
    stats0 = dict(dm.stats)
    for op, e in ops:
        before = set(dm.matching())
        if op == "i":
            dm.insert(*e)
        else:
            dm.delete(*e)
        after = set(dm.matching())
        churn.append(len(before ^ after))
    dyn_time = time.perf_counter() - t0
    dyn_searches = dm.stats["searches"] - stats0.get("searches", 0)
    dyn_aug = dm.stats["augmentations"] - stats0.get("augmentations", 0)

    # --- recompute baseline: same ops, full networkx solve after each ---
    # (solves before AND after each update to measure recompute churn fairly)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    base_churn = []
    t0 = time.perf_counter()
    for op, e in ops:
        prev = set(frozenset((u, v)) for u, v in
                   nx.max_weight_matching(g, maxcardinality=True))
        if op == "i":
            g.add_edge(*e)
        else:
            g.remove_edge(*e)
        cur = set(frozenset((u, v)) for u, v in
                  nx.max_weight_matching(g, maxcardinality=True))
        base_churn.append(len(prev ^ cur))
    base_time = time.perf_counter() - t0

    # sanity: final sizes agree
    final_dyn = dm.size()
    final_base = len(nx.max_weight_matching(g, maxcardinality=True))

    return {
        "n": n, "m": len(edges), "updates": len(ops), "seed": seed,
        "build_time": build_time,
        "dyn_time": dyn_time, "dyn_us": dyn_time / len(ops) * 1e6,
        "base_time": base_time, "base_us": base_time / len(ops) * 1e6,
        "churn_mean": sum(churn) / len(churn), "churn_max": max(churn),
        "base_churn_mean": sum(base_churn) / len(base_churn),
        "base_churn_max": max(base_churn),
        "searches": dyn_searches, "augmentations": dyn_aug,
        "final_dyn": final_dyn, "final_base": final_base,
        "speedup": base_time / dyn_time,
    }


def print_text(r):
    print(f"graph: n={r['n']}, m={r['m']} (initial), {r['updates']} updates, "
          f"seed={r['seed']}")
    print(f"initial build (dynamic, {r['m']} inserts): "
          f"{r['build_time']*1000:.0f} ms total, "
          f"{r['build_time']/r['m']*1e6:.0f} us/insert")
    print()
    hdr = (f"{'approach':<28}{'mean us/update':>16}{'total s':>10}"
           f"{'churn (edges changed/update)':>34}")
    print(hdr)
    print("-" * 88)
    print(f"{'DynamicMatching (ours)':<28}{r['dyn_us']:>16.0f}"
          f"{r['dyn_time']:>10.2f}"
          f"{r['churn_mean']:>22.2f} (mean){'':6}max={r['churn_max']}")
    print(f"{'networkx full recompute':<28}{r['base_us']:>16.0f}"
          f"{r['base_time']:>10.2f}"
          f"{r['base_churn_mean']:>22.2f} (mean){'':6}max={r['base_churn_max']}")
    print()
    print(f"speedup (mean per-update): {r['speedup']:.1f}x")
    print(f"dynamic searches (update phase): {r['searches']} over "
          f"{r['updates']} updates "
          f"({r['searches']/r['updates']:.2f}/update), "
          f"augmentations: {r['augmentations']}")
    print(f"final matching size: dynamic={r['final_dyn']}, "
          f"oracle={r['final_base']} "
          f"{'OK' if r['final_dyn'] == r['final_base'] else 'MISMATCH!'}")


def print_markdown(r):
    print(f"n={r['n']}, m={r['m']}, {r['updates']} updates, seed={r['seed']} "
          f"(pure Python vs pure Python)")
    print()
    print("| approach | mean µs/update | total s | churn mean | churn max |")
    print("|---|---|---|---|---|")
    print(f"| **DynamicMatching (ours)** | **{r['dyn_us']:.0f}** | "
          f"{r['dyn_time']:.2f} | **{r['churn_mean']:.2f}** | {r['churn_max']} |")
    print(f"| networkx full recompute | {r['base_us']:,.0f} | "
          f"{r['base_time']:.2f} | {r['base_churn_mean']:.2f} | "
          f"{r['base_churn_max']} |")
    print()
    print(f"speedup ≈ {r['speedup']:,.0f}×; {r['searches']} searches / "
          f"{r['updates']} updates ({r['searches']/r['updates']:.2f}/update); "
          f"final size dynamic={r['final_dyn']} oracle={r['final_base']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=2000, help="vertices (default 2000)")
    ap.add_argument("--m", type=int, default=None,
                    help="target edges (default 4*n)")
    ap.add_argument("--updates", type=int, default=500,
                    help="update operations (default 500)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--markdown", action="store_true",
                    help="print results as a markdown table")
    args = ap.parse_args(argv)

    m = args.m if args.m is not None else 4 * args.n
    r = run_benchmark(args.n, m, args.updates, args.seed)
    assert r["final_dyn"] == r["final_base"]
    if args.markdown:
        print_markdown(r)
    else:
        print_text(r)


if __name__ == "__main__":
    main()
