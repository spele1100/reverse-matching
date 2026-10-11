# Announcement — reverse-matching

**Title:** OpenAI published 719 math manuscripts with no code. I reverse-engineered one into a working library.

**Alt title:** No open-source library could update an exact maximum matching. So I built one.

**The origin story (lead):** Late last year OpenAI released `/math` — a dump of 719
research-level mathematics manuscripts, ~42% formalized in Lean, covering
breakthroughs like almost-linear-time maximum matching in general graphs.
No code, no proofs of implementation — just prose that reads like a program
transcribed to paper (ledgers, cached registers, instruction-level schedules).
So I set out to test a hypothesis: can you implement the thing directly from
the prose and get a correct, working artifact? The matching paper became my
test case — and it worked. That project (this repo's original 3-layer
implementation) is what led me here.

**The gap I found along the way:** while working with matchings on graphs whose
edges keep arriving or disappearing, I hit a quiet hole in the open-source
toolbox: every option either recomputes the maximum matching from scratch
(NetworkX, LEMON, Boost BGL) or gives up exactness and maintains only a
2-approximation (DynMatch). Exact *dynamic* matching exists in papers
(Sankowski-style algebraic updates), but nothing practical you can pip
install.

So I shipped the boring-but-correct version: after any single edge
insert/delete, matching cardinality changes by at most 1 (Lemma A), and by
Berge's theorem a single blossom-aware augmenting-path search restores
maximality (Lemma B). That's one O(m) repair per update instead of an O(n·m)
Edmonds recompute — and, arguably more valuable, the repair flips one short
augmenting path, so assignments are *stable*: recompute can rewrite hundreds
of matched pairs per update (I measured ~236 of ~1000 on n=2000), repair
changes <1 on average.

The implementation is a pure-Python reference (MIT): an e-maxx-derived
blossom search extended to a multi-root alternating forest, certified by a
differential harness that re-checks against NetworkX's oracle after *every*
operation of hundreds of randomized update sequences, plus blossom-gauntlet
graphs (glued triangles, odd cliques, Petersen) and adversarial churn.
295 tests.

Numbers (n=2000, m≈8000, 500 mixed updates, pure Python vs pure Python):
~1 ms/update vs ~3.7 s/update for the recompute baseline — roughly 3,800×
per update (≈1,900× against a single recompute), with 0.46 edges churned
per update vs 236.

Honest caveats: worst case per update is still O(m) (the conditional lower
bounds for exact dynamic matching mean nobody gets subpolynomial worst-case);
the cold build is not optimized — this is for workloads where the graph
*changes*, not a static solver replacement.

Repo: https://github.com/spele1100/reverse-matching
(The original 3-layer reverse-engineering of the OpenAI paper — phase machine,
reductions, assembly — lives in the same repo; the dynamic matcher above is
the practical offshoot.)

