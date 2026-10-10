"""
Layer 3 assembly — Chapter 9 of the OpenAI matching paper: the full
maximum-cardinality-matching pipeline.

Wires the verified Layer 1 reductions (reductions.py, Lemmas 9.1-9.3) and
the Layer 2 phase machine (phase_machine.py, Chapter 3) into the final
decision procedure of Section 9.2:

  threshold test at k:
      G  --Lemma 9.2 (build_hk)-->       H_k        (cardinality -> perfect)
      H_k --Lemma 9.3 (degree_reduce)--> gadget     (subcubic; isolates
                                                    certified absent)
      gadget --Chapter 3 machine-->      perfect matching or negative
      negative projections: gadget.project -> H_k perfect matching,
      hk.decode -> size-k matching of G, checked by identifier + one-time
      coverage before acceptance (Section 9.1's no-false-positives rule).

  threshold search (Section 9.2): integer binary search on k in
  [0, floor(n/2)]; an accepted test stores its decoded witness and raises
  the lower endpoint, a negative test lowers the exclusive upper endpoint.
  Exact-size feasibility is monotone ("edges may be dropped"), so with no
  false negatives the final stored witness is maximum.

  dovetailing (Section 9.4): rounds j = 0, 1, 2, ...; in round j branch
  t <= j receives a budget of 2^(j-t) steps (a step = one threshold test —
  the paper's word-RAM instruction granularity is coarsened to tests for
  this reference implementation, which the spec explicitly allows).
  Branch t = 0 is the deterministic Edmonds fallback (Lemma 2.4) and always
  completes within O(log(n+2)) tests, so the schedule halts on every path.
  Branch t >= 1 runs the phase machine with parameters a = 1000 + t,
  sigma = t + 20 in the paper (Eq. 9.3); here a is scaled down to the
  small-parameter regime (a = t + 1), which the paper calls "suspended"
  territory and the spec designates acceptable for a correctness reference.
  Error bookkeeping (Eq. 9.4/9.5): branch t is wrong with probability at
  most 2^-(t+20); branch 0 is exact; sum over t >= 1 of 2^-(t+20) = 2^-20
  < 1/3.  With Layer 2's deterministic substitutes there are no false
  negatives at all, so the realized error is 0 — the budget is kept as
  explicit bookkeeping, not relied upon.

Public API:
  max_cardinality_matching(n, edges, mode="dovetail", pm_branches=2,
                           params=None, max_rounds=None)
      -> (k, witness)   with witness a list of k vertex pairs, always
         cross-checkable against networkx.max_weight_matching(
         maxcardinality=True).

Only this file (plus its tests) is new; reductions.py and phase_machine.py
are consumed unmodified.
"""

from fractions import Fraction

from reductions import normalize, build_hk, degree_reduce
from phase_machine import perfect_matching, PhaseParams

__all__ = [
    "check_matching", "decide_reduced", "pm_test", "threshold_search",
    "Branch", "EdmondsBranch", "PMBranch", "run_dovetail", "total_error",
    "max_cardinality_matching",
]

# Paper Eq. 9.5: sum over branches t >= 1 of 2^-(t+20) is 2^-20 < 1/3.
TOTAL_ERROR_BOUND = Fraction(1, 3)
BRANCH0_ERROR = Fraction(0)


def branch_error(t):
    """Per-branch error allowance (Eq. 9.4): 0 for the exact fallback,
    2^-(t+20) for randomized branch t >= 1."""
    return BRANCH0_ERROR if t == 0 else Fraction(1, 2 ** (t + 20))


def total_error(max_branch=None):
    """Total error of the dovetail schedule over branches 1..max_branch
    (all of them if None): strictly less than 1/3 (Eq. 9.5)."""
    if max_branch is None:
        # geometric sum 2^-(t+20) over t>=1 equals 2 * 2^-21 = 2^-20
        return 2 * branch_error(1)
    return sum((branch_error(t) for t in range(1, max_branch + 1)),
               Fraction(0))


# ---------------------------------------------------------------------------
# Witness verification (Section 9.1: "check every proposed perfect matching
# by edge identifier and one-time coverage")
# ---------------------------------------------------------------------------

def check_matching(n, edges, matching, k=None):
    """Valid matching of the n-vertex graph with edge set `edges`: every
    pair is a genuine edge, each vertex covered at most once, and size k if
    given."""
    eset = {frozenset(e) for e in normalize(n, pairs=edges)}
    seen = set()
    for (u, v) in matching:
        if frozenset((u, v)) not in eset:
            return False
        if u in seen or v in seen or not (0 <= u < n and 0 <= v < n):
            return False
        seen.add(u)
        seen.add(v)
    if k is not None and len(matching) != k:
        return False
    return True


def _verify_perfect_hk(hk, hk_match):
    """Identifier + one-time-coverage check of a proposed perfect matching
    of H_k (Section 9.1).  Raises AssertionError on failure — with the
    no-false-positives rule a failing check is a bug, never an answer."""
    seen = set()
    for pair in hk_match:
        key = frozenset(pair)
        if key not in hk._by_pair:
            raise AssertionError(f"proposed edge {pair} is not an edge of H_k")
        u, v = pair
        if u in seen or v in seen:
            raise AssertionError(f"vertex covered twice in {pair}")
        seen.add(u)
        seen.add(v)
    if seen != set(hk.vertices):
        missing = set(hk.vertices) - seen
        raise AssertionError(f"H_k vertices uncovered: {sorted(missing)[:8]}")


# ---------------------------------------------------------------------------
# One threshold test through the full pipeline
# ---------------------------------------------------------------------------

def decide_reduced(vertices, edge_pairs, params=None):
    """Perfect-matching decision on an arbitrary loopless graph via Lemma
    9.3 + the Chapter 3 machine.  Returns a perfect matching as a list of
    original vertex pairs, or None.  An isolate certificate (Lemma 9.3:
    'check for isolates first') is a certified negative answer."""
    params = params if params is not None else PhaseParams(a=2)
    gadget = degree_reduce(vertices, edge_pairs)
    if gadget.isolate is not None:
        return None                      # certified: no perfect matching
    M = perfect_matching(len(gadget.new_vertices),
                         [(e.u, e.v) for e in gadget.new_edges], params)
    if M is None:
        return None                      # machine's negative answer
    return gadget.project(M)             # perfect matching of the input


def pm_test(n, edges, k, params=None):
    """Threshold test at cardinality k through the full Chapter 9 pipeline
    (Lemmas 9.2 + 9.3 + Chapter 3 machine).  Returns (True, witness) with
    witness a checked size-k matching of G, or (False, None).

    Never returns a false positive: every accepted witness is verified by
    edge identifier and one-time coverage at each reduction level."""
    hk = build_hk(n, edges, k)
    hk_match = decide_reduced(hk.vertices, [(e.u, e.v) for e in hk.edges],
                              params)
    if hk_match is None:
        return False, None
    _verify_perfect_hk(hk, hk_match)
    witness = hk.decode(hk_match)
    if not check_matching(n, edges, witness, k):
        raise AssertionError("decoded witness failed the final check")
    return True, witness


# ---------------------------------------------------------------------------
# Threshold search (Section 9.2)
# ---------------------------------------------------------------------------

def threshold_search(n, edges, test_fn):
    """Integer binary search on k: lower endpoint starts feasible at 0 with
    the stored empty matching, exclusive upper endpoint at floor(n/2)+1.
    Returns (k, witness).  test_fn(k) -> (ok, witness)."""
    lo, witness = 0, []
    hi = n // 2 + 1
    while hi - lo > 1:
        k = (lo + hi) // 2
        ok, w = test_fn(k)
        if ok:
            lo, witness = k, w            # accepted: store witness, raise lo
        else:
            hi = k                        # negative: lower upper endpoint
    return lo, witness


# ---------------------------------------------------------------------------
# Branches and the dovetailing schedule (Sections 9.2 / 9.4)
# ---------------------------------------------------------------------------

class Branch:
    """One dovetail branch: its own binary search state machine, advanced
    in unit steps (one threshold test per step)."""

    def __init__(self, t, n, edges):
        self.t = t
        self.n = n
        self.edges = list(edges)
        self.lo, self.hi, self.witness = 0, n // 2 + 1, []
        self.done = False
        self.suspended = False            # a suspended branch stays inactive
        self.answer = None
        self.tests_run = 0
        self.test_log = []                # (k, ok) per completed test
        self.rounds_run = 0

    def error(self):
        return branch_error(self.t)

    def _test(self, k):
        raise NotImplementedError

    def step(self, budget):
        """Run up to `budget` unit steps (threshold tests)."""
        for _ in range(budget):
            if self.suspended or self.done:
                return
            if self.hi - self.lo <= 1:
                self._finish()
                return
            k = (self.lo + self.hi) // 2
            ok, w = self._test(k)
            self.tests_run += 1
            self.test_log.append((k, ok))
            if ok:
                self.lo, self.witness = k, w
            else:
                self.hi = k
        if self.hi - self.lo <= 1:
            self._finish()

    def _finish(self):
        self.done = True
        self.answer = (self.lo, self.witness)

    def steps_needed_upper_bound(self):
        """Tests this branch still needs, upper bound (for halt analysis)."""
        import math
        return max(0, (self.n // 2 + 1).bit_length())


class EdmondsBranch(Branch):
    """Branch 0: the deterministic Edmonds fallback (Lemma 2.4) — 'run a
    standard blossom algorithm and check the answer'.  Exact on every
    threshold, so it always completes with a maximum matching."""

    def __init__(self, n, edges):
        super().__init__(0, n, edges)
        self._nu = None
        self._max_matching = None

    def _solve(self):
        if self._nu is None:
            import networkx as nx
            g = nx.Graph()
            g.add_nodes_from(range(self.n))
            g.add_edges_from(self.edges)
            self._max_matching = [(u, v) for u, v in
                                  nx.max_weight_matching(g, maxcardinality=True)]
            self._nu = len(self._max_matching)

    def _test(self, k):
        self._solve()
        if k <= self._nu:
            # monotonicity: dropping edges from the maximum matching keeps
            # a matching, so any k <= nu is feasible with an explicit witness
            return True, self._max_matching[:k]
        return False, None


class PMBranch(Branch):
    """Branch t >= 1: the Chapter 3 phase machine with branch parameters.
    Paper Eq. 9.3 uses a = 1000 + t, sigma = t + 20; this reference scales
    a down to the small-parameter regime (a = t + 1), per the spec's key
    judgment.  With Layer 2's deterministic substitutes there are no false
    negatives and no suspensions; the error allowance 2^-(t+20) is carried
    as bookkeeping only."""

    def __init__(self, t, n, edges, params=None):
        super().__init__(t, n, edges)
        self.params = params if params is not None else PhaseParams(a=t + 1)

    def _test(self, k):
        return pm_test(self.n, self.edges, k, self.params)


def run_dovetail(branches, max_rounds=None):
    """Weighted dovetailing schedule (Section 9.4).  Round j gives branch
    t = 0..j, in order, a budget of 2^(j-t) steps, resuming each branch at
    its interruption point; suspended branches stay inactive; the first
    branch to complete wins and its answer is returned.

    Halt: branch 0 (Edmonds fallback) needs at most ceil(log2(n+2)) + 1
    tests and receives 2^j steps in round j, so it always finishes; the
    schedule halts on every path."""
    j = 0
    limit = max_rounds
    if limit is None and branches and branches[0].t == 0:
        # branch 0 needs at most (n//2 + 1).bit_length() tests; it has
        # received sum(2^i for i<j) = 2^j - 1 steps by round j, so it is
        # done by round J = bit_length(n//2 + 1) at the latest.
        limit = branches[0].steps_needed_upper_bound() + 2
    while True:
        for t in range(min(j + 1, len(branches))):
            br = branches[t]
            if br.suspended or br.done:
                continue
            br.step(2 ** (j - t))
            br.rounds_run += 1
            if br.done:
                return br
        j += 1
        if limit is not None and j > limit:
            raise RuntimeError(
                "dovetail schedule exceeded its halt bound (branch 0 "
                "should always complete; this is a bug)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def max_cardinality_matching(n, edges, mode="dovetail", pm_branches=2,
                             params=None, max_rounds=None):
    """Maximum-cardinality matching, Chapter 9 assembly.

    n, edges : the input graph (vertices 0..n-1, loopless; edges arbitrary
               (u, v) pairs, deduplicated internally)
    mode     : "dovetail" — branch 0 (Edmonds fallback) + pm_branches
                          phase-machine branches under the round schedule;
               "pm"       — one phase-machine branch alone (still exact
                          with the deterministic Layer 2 substitutes);
               "edmonds"  — branch 0 alone (the paper's built-in fallback,
                          i.e. the cross-check target itself).
    params   : PhaseParams (or dict of its kwargs) for the PM branches.

    Returns (k, witness): k = matching number, witness a checked list of k
    vertex pairs.  Never returns an unchecked (possibly false-positive)
    answer."""
    if params is not None and not isinstance(params, PhaseParams):
        params = PhaseParams(**params)
    edges = normalize(n, pairs=edges)
    if mode == "edmonds":
        branches = [EdmondsBranch(n, edges)]
    elif mode == "pm":
        branches = [PMBranch(1, n, edges, params)]
        if max_rounds is None:
            max_rounds = 8 * max(2, n)      # no branch-0 halt guarantee here
    elif mode == "dovetail":
        branches = [EdmondsBranch(n, edges)]
        branches += [PMBranch(t, n, edges, params)
                     for t in range(1, pm_branches + 1)]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    winner = run_dovetail(branches, max_rounds=max_rounds)
    k, witness = winner.answer
    if not check_matching(n, edges, witness, k):
        raise AssertionError("final witness failed the output check")
    return k, witness
