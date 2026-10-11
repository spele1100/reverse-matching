# Reverse-Matching: A Reference Implementation of Almost-Linear-Time Max-Cardinality Matching

**AI-assisted paper reproduction, paper-to-verified-code in one day.**

This repository is a *correctness-only* pure-Python reference implementation of the
matching machinery in *"Almost-Linear-Time Maximum-Cardinality Matching in General
Graphs"* (OpenAI, 2026) — the paper claiming the first essentially-linear-time
algorithm for maximum matching in general graphs, breaking a 60-year plateau set by
Edmonds' blossom algorithm.

> **What this is:** proof that the paper's constructions actually run — decision +
> witness verified against an oracle across 290+ tests.
> **What this is not:** a fast implementation. The paper's asymptotic running-time
> claims are *not* reproduced (see [Limitations](#limitations)).

## Install & Quickstart

Requires Python 3.10+ and `networkx`.

```bash
pip install -e .          # from a clone of this repo
```

Then use the CLI:

```bash
maxmatch --random 10 0.3   # random graph, fast mode, human-readable output
maxmatch examples/path_graph.txt
maxmatch --help
```

Or the Python API:

```python
from assembly import max_cardinality_matching

k, matching = max_cardinality_matching(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
                                       mode="edmonds")
# k == 3, matching is a validated witness (no false positives, ever)
```

`mode="edmonds"` is the fast default (the paper's built-in fallback branch);
`"pm"` and `"dovetail"` are the paper's research machinery — exact but very
slow, only for tiny graphs. See `examples/` and `maxmatch --help`.

## Why

Theory papers rarely ship runnable artifacts. We wanted to know: *does it actually
work?* So we took the chapter texts (Ch. 3 state machine, Ch. 9 assembly), wrote an
implementation spec with explicit judgment calls, and built it in three verified
layers.

The interesting part is the **process**: the whole thing — spec, three layers of
implementation, ~250 tests cross-checked against NetworkX's blossom oracle — was
built in a single day, driven spec-first: write a detailed implementation spec
up front, implement layer by layer, and keep every prior layer's tests green
after each step. See [Workflow](#workflow).

## Architecture

Three layers, each independently verified:

| Layer | File | Paper | What it does | Tests |
|---|---|---|---|---|
| 1 | `reductions.py` | Ch. 9 | Lém. 9.1 Beneš-style permutation network; Lém. 9.2 cardinality→perfect-matching reduction (`build_hk`/`decode`/witness lift); Lém. 9.3 Dahlhaus–Karpinski degree-reduction gadget + isolate certificates | 71 |
| 2 | `phase_machine.py` | Ch. 3 | The phase state machine: bins, rows, µ, blockers; a+1 parallel ledgers (Def. 3.1); trichotomy main loop with exact ledger edits; projection via exact integral max-flow; deterministic tight-set search; blossom materialization | 105 |
| 3 | `assembly.py` | Ch. 9 | Full pipeline: binary threshold search over k (per-branch error 2⁻⁽ᵗ⁺²⁰⁾), dovetailing schedule with branch 0 = deterministic Edmonds fallback (guaranteed halt), total error < 1/3. Every result is witness-checked before return (no false positives) | 75 |

```python
from assembly import max_cardinality_matching

k, matching = max_cardinality_matching(n, edges)   # decision + witness
```

**Full suite: 295 passed, 1 skipped** (~27 min), with every decision cross-checked against
`networkx.max_weight_matching(maxcardinality=True)` — on G and on the constructed
graphs H_k, plus witness validity (edges exist, no vertex reused).

## Dynamic matching (Tier 2)

`dynamic.py` adds a **DynamicMatching** class that maintains an *exact maximum
matching of a general graph under edge insertions and deletions* — the operation
no practical open-source library currently supports without full recomputation
(NetworkX, LEMON, Boost all re-solve per call; DynMatch keeps only a 2-approximation).

```python
from dynamic import DynamicMatching

dm = DynamicMatching(n=6)
dm.insert(0, 1)          # both free -> matched instantly, no search
dm.insert(1, 2)          # 1 free endpoint search if needed
dm.insert(2, 4)          # arbitrary: one blossom search over free roots
dm.delete(0, 1)          # unmatched delete: zero work; matched delete: repair search
dm.size()                # |M| now
dm.matching()            # sorted list of (u, v) pairs — always a VALID matching
dm.is_valid()            # invariant self-check (mate symmetric, edges exist)
dm.oracle_check()        # test-only: differential check vs networkx
```

**Guarantees** (proofs in the module docstring, spec in `papers/`):

- The matching is **maximum after every update** (eager, exact — not approximate).
- Deleting an unmatched edge costs O(1) (Lemma A: the matching stays maximum).
- Every other update performs **at most one blossom search**, O(m) worst case,
  versus O(n·m) for a full Edmonds recompute — an O(n) factor.
- **Stability**: repairs flip one short augmenting path; full recomputation can
  churn the entire matching (see churn numbers below).
- Insert of a present edge / delete of an absent edge are no-ops; self-loops raise
  `ValueError`; parallel edges are deduplicated.

The repair correctness is the classical pair: ν changes by ≤ 1 per single edge
update (Lemma A), and by Berge's theorem one augmenting-path search restores
maximality (Lemma B) — a matched-edge deletion may legitimately shrink ν when no
re-augmentation exists; the search decides. The search kernel is a faithful port
of the battle-tested e-maxx blossom implementation, extended to a multi-root
alternating forest (an edge joining even vertices of *different* trees closes an
augmenting path; even–odd cross edges are provably inert).

**Correctness certification:** `tests_dynamic.py` is a differential harness —
after *every* operation of hundreds of randomized update sequences (n up to 200,
densities 0.05–0.5, plus glued-triangle / K₂ₜ₊₁-minus-matching / Petersen blossom
gauntlets and adversarial churn), the matching size must equal NetworkX's oracle
and the matching must be valid. 39 tests, ~every-op oracle checks.

**Benchmarks** (`python -m benchmarks.bench_dynamic`, pure Python vs pure
Python, seed 12345; the baseline harness re-solves before *and* after each
update to measure churn on both sides — so per-update speedup against a
single recompute is ≈ 2× the figure shown):

| n | m | approach | mean µs/update | churn mean | churn max |
|---|---|---|---|---|---|
| 2000 | 8000 | **DynamicMatching (ours)** | **973** | **0.46** | 12 |
| 2000 | 8000 | networkx full recompute | 3,707,425 | 236.02 | 884 |
| 500 | 2000 | **DynamicMatching (ours)** | **1037** | **0.47** | 14 |
| 500 | 2000 | networkx full recompute | 269,097 | 60.12 | 292 |

- Speedup ≈ **3,810× per update** at n=2000 (≈1,900× against a single
  recompute), ≈ 260× at n=500. See `benchmarks/README.md`.
- Search cost: 0.58 blossom searches per update on average (290 searches
  over 500 updates at n=2000), 27 of which augmented.
- Initial build: 8,000 inserts ≈ 16 ms/insert (each insert runs at most one
  O(m) search; a cold build is intentionally not optimized — use it for
  workloads that change over time, not as a static solver).
- Churn is the headline: recompute rewrites ~236 of ~1,000 matched edges per
  update on average (up to 884); repair touches <1. If downstream consumers
  depend on assignment stability, that churn is a cost no amount of speed
  fixes.


## Honest engineering notes

- **Pricing is the hard part.** The paper's pricing primitive (Lemma 5.1) sits atop
  ~50 pages of dynamic min-ratio-cycle machinery (Ch. 5–8) that we deliberately did
  *not* rebuild. Exact-rational substitutes (exact Dinic max-flow; an exact-Fraction
  simplex LP for the steepest direction) preserve correctness. Cheaper substitutes
  we tried first — Farkas/potential directions, negative-cycle detection — *failed*
  (non-termination; degeneracy on trivial 2-cycles). The LP was necessary.
- **Projection never peels end-to-end** in the small-parameter regimes we test
  (the "fill" outcome always wins). Peel correctness is covered by unit tests of
  the exchange network and a forced-peel end-to-end test.
- **Composed runs are cubic-ish.** The pipeline blows up composed graph sizes
  (n=3 pipeline ≈ 20 min), so end-to-end PM-mode tests stay tiny; `mode="edmonds"`
  handles larger inputs quickly.
- 6 documented deviations from the text (ε-grid snapping omitted, cached registers
  elided, K-accounting fix, …) — all in `README` per-layer sections and file headers.
  Chapter text wins over the spec wherever they diverged.

## Limitations

- **Running time is not reproduced.** With exact substitutes standing in for the
  Ch. 5–8 engine, the paper's p^{1+o(1)} claims are unverifiable here. This
  artifact verifies *combinatorial correctness*, not the analytic envelope.
- No word-RAM instruction-level accounting, no S-scaled ε/δ arithmetic (exact
  Fractions instead), randomized searches replaced by deterministic equivalents.

## Workflow

1. Read chapters 3 + 9; write a 32 KB implementation spec with explicit judgment
   calls (what to substitute, what to keep exact).
2. Implement one layer at a time: implement → write its own test suite → run the
   full suite including all prior layers (everything must stay green).
3. Review, re-run verification, and report.

## Requirements

Python 3.10+, `networkx` (oracle + max-flow), `pytest`.

```bash
python -m pytest tests_reductions.py -q                 # layer 1, ~90 s
python -m pytest tests_reductions.py tests_phase_machine.py -q   # +layer 2, ~4 min
python -m pytest -q                                     # everything, ~27 min
```

## License

MIT — see [LICENSE](LICENSE).
