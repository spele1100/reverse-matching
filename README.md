# Reverse-Matching: A Reference Implementation of Almost-Linear-Time Max-Cardinality Matching

**AI-assisted paper reproduction, paper-to-verified-code in one day.**

This repository is a *correctness-only* pure-Python reference implementation of the
matching machinery in *"Almost-Linear-Time Maximum-Cardinality Matching in General
Graphs"* (OpenAI, 2026) — the paper claiming the first essentially-linear-time
algorithm for maximum matching in general graphs, breaking a 60-year plateau set by
Edmonds' blossom algorithm.

> **What this is:** proof that the paper's constructions actually run — decision +
> witness verified against an oracle across 250+ tests.
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
produced by orchestrating AI sub-agents in a single day. Human effort: reading the
spec, reviewing results. See [Workflow](#workflow).

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

**Full suite: 251/251 green** (~27 min), with every decision cross-checked against
`networkx.max_weight_matching(maxcardinality=True)` — on G and on the constructed
graphs H_k, plus witness validity (edges exist, no vertex reused).

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
2. Spawn one agent per layer, each: implement → write its own test suite → run the
   full suite including all prior layers (everything must stay green).
3. Parent reviews, re-runs verification, and reports.

Failure modes we hit and survived: two agents concurrently writing the same file
(Layer 1 — merged after audit); a gateway restart killing a 4-hour run (no loss,
thanks to write-early discipline added to every subsequent brief).

## Requirements

Python 3.10+, `networkx` (oracle + max-flow), `pytest`.

```bash
python -m pytest tests_reductions.py -q                 # layer 1, ~90 s
python -m pytest tests_reductions.py tests_phase_machine.py -q   # +layer 2, ~4 min
python -m pytest -q                                     # everything, ~27 min
```

## License

MIT — see [LICENSE](LICENSE).
