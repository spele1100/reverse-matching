# Benchmarks

`bench_dynamic.py` measures **DynamicMatching update cost vs the
NetworkX full-recompute baseline** — what a Python user without this library
would do (`networkx.max_weight_matching(G, maxcardinality=True)` re-run per
update).

## Run

```bash
python -m benchmarks.bench_dynamic                          # n=2000, 500 updates
python -m benchmarks.bench_dynamic --n 500 --updates 200    # smaller config
python -m benchmarks.bench_dynamic --markdown               # markdown table
```

Options: `--n` (vertices), `--m` (target edges, default 4n), `--updates`,
`--seed`, `--markdown`.

## What it measures

- Builds a random graph (n vertices, m edges), then applies a mixed
  insert/delete workload biased to keep |E| ≈ m.
- **Dynamic run**: apply each update to `DynamicMatching`; record per-update
  wall time and *churn* (symmetric difference of the matching before/after).
- **Baseline run**: same ops on a `networkx.Graph`, full re-solve after each
  update (solved before and after each op so churn is measurable on both
  sides). This makes the baseline per-update cost ~2 solves.
- Sanity check: final matching sizes must agree with the oracle.

## Reading the output

- **mean µs/update** — the headline speedup (repair search vs full Edmonds).
- **churn** — edges of the matching rewritten per update. Repair flips one
  short augmenting path (<1 edge changed on average); recompute routinely
  rewrites hundreds. If downstream consumers depend on assignment stability,
  recompute churn is a correctness-adjacent cost no amount of speed fixes.

Reference numbers (pure Python vs pure Python) are in the top-level
`README.md` benchmark table.
