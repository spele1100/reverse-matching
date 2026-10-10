#!/usr/bin/env python3
"""maxmatch — CLI for the reverse-matching reference implementation.

Reads an edge-list graph (or generates a random one) and computes a
maximum-cardinality matching, printing a witness that is validated before
return (no false positives, ever).

Edge-list format:
    # comments allowed anywhere
    n m          <- optional header: node count, edge count
    u v          <- one edge per line, 0-indexed vertices
    ...

Default mode is "edmonds" (fast). Modes "pm" and "dovetail" are the paper's
research machinery — exact but very slow; use only on tiny graphs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from assembly import max_cardinality_matching

SLOW_MODES = {"pm", "dovetail"}


def parse_edge_list(path):
    """Parse an edge-list file. Returns (n, edges).

    n is max(header n if present, largest vertex + 1).
    """
    edges = []
    header_n = None
    header_m = None
    first = True
    edge_count = 0
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise SystemExit(f"{path}:{lineno}: expected 'u v', got {raw!r}")
            try:
                a, b = int(parts[0]), int(parts[1])
            except ValueError:
                raise SystemExit(f"{path}:{lineno}: non-integer vertex in {raw!r}")
            if a < 0 or b < 0:
                raise SystemExit(f"{path}:{lineno}: negative vertex in {raw!r}")
            # Heuristic header: the first data line "n m" with n >= m is a
            # header (an edge "u v" with u >= v on the first line is unusual,
            # and u == v is invalid anyway); edges "u v" with u < v are never
            # mistaken for a header.
            if first and a >= b:
                header_n, header_m = a, b
                first = False
                continue
            first = False
            if a == b:
                raise SystemExit(f"{path}:{lineno}: self-loop {a}-{a} not allowed")
            edges.append((a, b))
            edge_count += 1
    max_v = max((max(u, v) for u, v in edges), default=-1)
    n = max(header_n or 0, max_v + 1)
    if header_m is not None and header_m != edge_count:
        print(f"warning: header says m={header_m} but {edge_count} edges read",
              file=sys.stderr)
    return n, edges


def random_graph(n, p, seed=None):
    rng = random.Random(seed)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.add((u, v))
    return n, sorted(edges)


def validate_matching(n, edges, matching):
    seen = set()
    eset = {frozenset(e) for e in edges}
    for u, v in matching:
        if not (0 <= u < n and 0 <= v < n):
            return False, f"vertex out of range: {u}-{v}"
        if u == v:
            return False, f"self-loop in matching: {u}"
        if u in seen or v in seen:
            return False, f"vertex reused: {u}-{v}"
        if frozenset((u, v)) not in eset:
            return False, f"edge not in graph: {u}-{v}"
        seen.add(u)
        seen.add(v)
    return True, "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="maxmatch",
        description="Maximum-cardinality matching (validated witness) via the "
                    "reverse-matching reference implementation.",
    )
    ap.add_argument("graph", nargs="?", help="edge-list file (see docstring format)")
    ap.add_argument("--random", nargs=2, metavar=("N", "P"), type=float,
                    help="generate a random graph: N nodes, Erdős–Rényi edge prob P")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for --random")
    ap.add_argument("--mode", choices=["edmonds", "pm", "dovetail"],
                    default="edmonds",
                    help="algorithm mode (default: edmonds — fast; pm/dovetail "
                         "are slow research modes, tiny graphs only)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if args.random is not None:
        n, p = int(args.random[0]), args.random[1]
        if not (0 <= p <= 1):
            ap.error("--random P must be in [0, 1]")
        n, edges = random_graph(n, p, args.seed)
    elif args.graph:
        n, edges = parse_edge_list(args.graph)
    else:
        ap.error("provide a graph file or --random N P")

    if args.mode in SLOW_MODES and (n > 12 or len(edges) > 30):
        print(f"warning: mode '{args.mode}' is a slow research mode; "
              f"n={n}, m={len(edges)} may take a very long time. "
              f"Consider --mode edmonds.", file=sys.stderr)

    k, matching = max_cardinality_matching(n, edges, mode=args.mode)
    ok, why = validate_matching(n, edges, matching)
    if not ok:
        raise SystemExit(f"internal error: invalid matching ({why})")

    if args.json:
        print(json.dumps({
            "mode": args.mode, "nodes": n, "edges": len(edges),
            "matching_size": k, "matching": [list(e) for e in matching],
            "valid": ok,
        }))
    else:
        print(f"mode:            {args.mode}")
        print(f"nodes:           {n}")
        print(f"edges:           {len(edges)}")
        print(f"matching size k: {k}")
        print(f"matching edges:  {', '.join(f'{u}-{v}' for u, v in matching)}")
        print(f"validated:       yes (witness-checked, no false positives)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
