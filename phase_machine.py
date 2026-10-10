"""
phase_machine.py — Layer 2: Chapter 3 "phase state machine" of the OpenAI
matching paper (Almost-Linear-Time Maximum-Cardinality Matching in General
Graphs), correctness-only reference implementation.

Source of truth: /root/papers/ch1-3.txt (Sections 3.1-3.8).  Spec:
/root/.openclaw/workspace/papers/matching-ch3-ch9-spec.md.

Per the spec's key judgment, the research-grade interfaces of Sections 4-8
are replaced by exact polynomial-time substitutes:

  * Pricing primitive (Lemma 5.1) -> an exact priced step on the top
    objective Phi = sum_e log x_e + sum_d log s_d: the steepest
    circulation direction (max of g.h over sum_e |h_e|/f_e <= 1) is
    computed exactly by a small exact-rational simplex LP, then an exact
    rational line search (bisection on Phi'(t), strictly decreasing) is
    run along it.  LOW is reported exactly when the max ratio is
    <= 1/2 - 1/1000, matching Lemma 5.1's contract that LOW is impossible
    whenever a direction with ratio >= 1/2 exists.

  * Local projection at j>1 (Lemma 4.1) -> the same exact integral max-flow
    exchange network of Lemma 3.5, computed globally (Dinic).

  * Randomized bounded searches 1 and 3 (Section 3.7) -> deterministic
    exact substitutes: the pair action decides gain-vs-contract with an
    exact min-cut computing min_{Z >= {i,j}} h(Z) (Search 2's refinement
    network run globally), and joint freeing uses exhaustive BFS
    alternating paths in the full-neighbor reverse graph (the paper's own
    exhausted-set argument proves this deterministic criterion).

Deviations from the literal text (correctness-preserving; see README):
  * No epsilon-grid snapping: the top ledger uses exact Fractions (the grid
    serves the dynamic engine's bit complexity, not correctness).
  * No cached registers: expansion eligibility reads the exact top slack,
    so barrier (3.10) holds with cache error zero.
  * delta is a configurable exact rational (default 2^-120), not S^-1000.
  * a, the L multiplier, etc. are configurable small parameters.
"""

import sys
from collections import defaultdict, deque
from fractions import Fraction
from math import ceil, log2

sys.setrecursionlimit(1_000_000)


def _ceil_div(a, b):
    return -(-a // b)


# ---------------------------------------------------------------------------
# Parameters (Eq. 3.2 / 3.3 / 3.11 with configurable small values)
# ---------------------------------------------------------------------------

class PhaseParams:
    def __init__(self, a=3, l_mult=100, delta=None, check_invariants=True,
                 debug_edits=False, max_actions=200000,
                 max_pricing_steps=100000, line_iters=64):
        assert a >= 1, "need ledgers 0..a with floors down to c_a = 1"
        self.a = a
        self.C = a + 1
        self.l_mult = l_mult          # L = ceil(l_mult*C*N/K); paper uses 10^4
        self.delta = Fraction(1, 2 ** 120) if delta is None else Fraction(delta)
        self.check_invariants = check_invariants
        self.debug_edits = debug_edits
        self.max_actions = max_actions
        self.max_pricing_steps = max_pricing_steps
        self.line_iters = line_iters


# ---------------------------------------------------------------------------
# Exact integral max flow (Dinic) — deterministic Lemma 2.1 substitute
# ---------------------------------------------------------------------------

class Dinic:
    def __init__(self):
        self.to = []
        self.cap = []
        self.g = defaultdict(list)

    def add(self, u, v, c):
        i = len(self.to)
        self.to.append(v); self.cap.append(c); self.g[u].append(i)
        self.to.append(u); self.cap.append(0); self.g[v].append(i + 1)
        return i

    def maxflow(self, s, t):
        total = 0
        while True:
            level = {s: 0}
            q = deque([s])
            while q:
                u = q.popleft()
                for i in self.g[u]:
                    v = self.to[i]
                    if self.cap[i] > 0 and v not in level:
                        level[v] = level[u] + 1
                        q.append(v)
            if t not in level:
                return total
            it = defaultdict(int)

            def dfs(u, lim):
                if u == t:
                    return lim
                while it[u] < len(self.g[u]):
                    i = self.g[u][it[u]]
                    v = self.to[i]
                    if self.cap[i] > 0 and level.get(v) == level[u] + 1:
                        d = dfs(v, min(lim, self.cap[i]))
                        if d > 0:
                            self.cap[i] -= d
                            self.cap[i ^ 1] += d
                            return d
                    it[u] += 1
                level[u] = -1
                return 0

            while True:
                f = dfs(s, float("inf"))
                if f == 0:
                    break
                total += f

    def residual_reachable(self, s):
        seen = {s}
        q = deque([s])
        while q:
            u = q.popleft()
            for i in self.g[u]:
                v = self.to[i]
                if self.cap[i] > 0 and v not in seen:
                    seen.add(v)
                    q.append(v)
        return seen

    def flow_on(self, i):
        return self.cap[i ^ 1]


# ---------------------------------------------------------------------------
# Pure combinatorial helpers (unit-testable)
# ---------------------------------------------------------------------------

def pair_decision(rows_nb, bins, i, j):
    """Tight-set alternative, deterministic (Lemma 3.6 + Search 2 globally).

    rows_nb : row -> iterable of adjacent bins (N_B(r))
    bins    : iterable of bin ids
    Returns (min_h, X): min_h = min over Z containing {i,j} of
    h(Z) = |Z| - |{r : N_B(r) subset of Z}|; X is the source side of the
    smallest minimum cut (the unique minimal minimizer, by submodularity).
    """
    bins = list(bins)
    rows = list(rows_nb)
    A = len(rows)
    BIG = A + len(bins) + 10
    d = Dinic()
    S, T = ("S",), ("T",)
    for r in rows:
        d.add(S, ("r", r), 1)
    seen = set()
    for r in rows:
        for b in rows_nb[r]:
            if (r, b) in seen:
                continue
            seen.add((r, b))
            d.add(("r", r), ("b", b), BIG)
    for b in bins:
        d.add(("b", b), T, 1)
    d.add(S, ("b", i), BIG)
    d.add(S, ("b", j), BIG)
    val = d.maxflow(S, T)
    min_h = val - A
    reached = d.residual_reachable(S)
    X = {b for b in bins if ("b", b) in reached}
    return min_h, X


def exchange_projection(L, c, weights, incidences, x):
    """Lemma 3.5 exact exchange projection (pure).

    L          : per-row allocation amount
    c          : requested floor constant (c_j)
    weights    : bid -> w_d
    incidences : iterable of keys (r, bid, eid) of the incidence graph B
    x          : dict key -> parent allocation (ints; missing = 0)

    Returns (True, xnew) if all deficits fill at value beta (outcome i), or
    (False, R, xnew) with R the residual-reachable peel set (outcome ii).
    xnew covers the whole state; callers restrict it to survivors.
    """
    bids = list(weights)
    incidences = list(incidences)
    load = defaultdict(int)
    for k in incidences:
        load[k[1]] += x.get(k, 0)
    y = {b: L - load[b] for b in bids}
    b_def = {b: max(0, c * weights[b] - y[b]) for b in bids}
    u_spare = {b: max(0, y[b] - c * weights[b]) for b in bids}
    beta = sum(b_def.values())
    d = Dinic()
    S, T = ("S",), ("T",)
    for b in bids:
        d.add(S, ("b", b), b_def[b])
        d.add(("b", b), T, u_spare[b])
    arc_dr = {}
    arc_rd = {}
    for k in incidences:
        r, b, _eid = k
        arc_dr[k] = d.add(("b", b), ("r", r), x.get(k, 0))
        arc_rd[k] = d.add(("r", r), ("b", b), L + 1)
    val = d.maxflow(S, T)
    xnew = {k: x.get(k, 0) - d.flow_on(arc_dr[k]) + d.flow_on(arc_rd[k])
            for k in incidences}
    if val == beta:
        return True, xnew
    reached = d.residual_reachable(S)
    R = {b for b in bids if ("b", b) in reached}
    return False, R, xnew


# ---------------------------------------------------------------------------
# Phase state
# ---------------------------------------------------------------------------

class CNode:
    """Contraction-tree node (Lemma 3.8): children are the contracted bins
    and the swallowed captive rows."""

    __slots__ = ("children", "ctype", "cobj", "cverts", "join_eid",
                 "cap_assign", "unused", "row_nb")

    def __init__(self):
        self.children = []       # ids: bid or row vertex
        self.ctype = {}          # id -> 'bin' | 'row'
        self.cobj = {}           # bid -> Bin object (retained for expansion)
        self.cverts = {}         # id -> frozenset of physical vertices
        self.join_eid = None
        self.cap_assign = {}     # captive row -> child bid (mu at contraction)
        self.unused = None       # child bid with no captive
        self.row_nb = {}         # captive row -> sorted adjacent child bids

    def child_of(self, v):
        for cid in self.children:
            if v in self.cverts[cid]:
                return cid
        raise KeyError(f"vertex {v} not in this contraction node")


class Bin:
    __slots__ = ("bid", "vertices", "weight", "exposure", "assigned", "tree")

    def __init__(self, bid, vertices, weight, exposure, assigned, tree):
        self.bid = bid
        self.vertices = vertices          # frozenset
        self.weight = weight              # (|d|+1)/2
        self.exposure = exposure          # current free exposure (when free)
        self.assigned = assigned          # (row, eid) or None
        self.tree = tree                  # ('leaf', v) or CNode


class Ledger:
    __slots__ = ("level", "c", "x", "p")

    def __init__(self, level, c):
        self.level = level
        self.c = c
        self.x = {}    # incidence key (r, bid, eid) -> value (0s may be absent)
        self.p = {}    # bid -> debt



# ---------------------------------------------------------------------------
# Exact rational simplex (tableau, Bland's rule) for the max-ratio direction
# LP:  maximize sum_e g_e h_e  s.t.  h is a circulation and
#       sum_e w_e |h_e| <= 1,   with h split as h = u - v, u,v >= 0.
# Returns (value, h) with value = max ratio and h the steepest direction.
# ---------------------------------------------------------------------------

def max_ratio_direction(nodes, arcs, g, w):
    """nodes: list of node ids; arcs: list of (u, v) node pairs;
    g, w: per-arc Fractions (w > 0).  Returns (value, h) exactly.

    Tableau simplex on:  maximize sum g_e (u_e - v_e) subject to node
    conservation, sum w_e (u_e + v_e) + slack = 1, all variables >= 0.
    Objective row maintained incrementally; Dantzig pricing with a Bland
    fallback once the iteration budget is exhausted (guarantees finite
    termination)."""
    m = len(arcs)
    nv = 2 * m + 1
    rows = len(nodes) + 1
    node_idx = {n: i for i, n in enumerate(nodes)}
    T = [[Fraction(0)] * nv for _ in range(rows)]
    for j, (uu, vv) in enumerate(arcs):
        T[node_idx[uu]][j] = Fraction(1)
        T[node_idx[vv]][j] = Fraction(-1)
        T[node_idx[uu]][m + j] = Fraction(-1)
        T[node_idx[vv]][m + j] = Fraction(1)
        T[rows - 1][j] = w[j]
        T[rows - 1][m + j] = w[j]
    T[rows - 1][nv - 1] = Fraction(1)
    # all-artificial starting basis (b = 0 on conservation rows, 1 on the
    # norm row, whose artificial leaves in phase 1 via the slack column)
    ART = rows
    total = nv + ART
    for i in range(rows):
        T[i] = T[i] + [Fraction(0)] * ART
        T[i][nv + i] = Fraction(1)
    b = [Fraction(0)] * (rows - 1) + [Fraction(1)]
    basis = [nv + i for i in range(rows)]

    def obj_row(cost):
        cb = [cost[basis[i]] for i in range(rows)]
        z = []
        for j in range(total):
            z.append(cost[j] - sum(cb[i] * T[i][j] for i in range(rows)))
        return z

    def pivot(pr, pc, z):
        pv = T[pr][pc]
        T[pr] = [x / pv for x in T[pr]]
        b[pr] = b[pr] / pv
        for i in range(rows):
            if i != pr and T[i][pc] != 0:
                f = T[i][pc]
                T[i] = [a - f * bb for a, bb in zip(T[i], T[pr])]
                b[i] = b[i] - f * b[pr]
        if z[pc] != 0:
            f = z[pc]
            z[:] = [a - f * bb for a, bb in zip(z, T[pr])]
        basis[pr] = pc

    def solve(cost, allow_art, z):
        it = 0
        bland_after = 6 * (rows + total)
        while True:
            it += 1
            assert it < 200000, "simplex iteration cap"
            entering = None
            if it < bland_after:
                # Dantzig: most negative reduced cost
                best = Fraction(0)
                for j in range(total):
                    if not allow_art and j >= nv:
                        continue
                    if z[j] < best:
                        best = z[j]
                        entering = j
            else:
                for j in range(total):
                    if not allow_art and j >= nv:
                        continue
                    if z[j] < 0:
                        entering = j
                        break
            if entering is None:
                return
            leaving = None
            best = None
            for i in range(rows):
                if T[i][entering] > 0:
                    ratio = b[i] / T[i][entering]
                    if (best is None or ratio < best
                            or (ratio == best and basis[i] < basis[leaving])):
                        best = ratio
                        leaving = i
            assert leaving is not None, "unbounded LP (impossible here)"
            pivot(leaving, entering, z)

    # phase 1: drive artificials to zero
    cost1 = [Fraction(0)] * nv + [Fraction(1)] * ART
    solve(cost1, True, obj_row(cost1))
    p1 = sum(b[i] for i in range(rows) if basis[i] >= nv)
    assert p1 == 0, "phase 1 infeasible"
    # phase 2: minimize -(g.u - g.v)
    cost2 = [-x for x in g] + [x for x in g] + [Fraction(0)] * (1 + ART)
    solve(cost2, False, obj_row(cost2))
    h = [Fraction(0)] * m
    for i in range(rows):
        j = basis[i]
        if j < m:
            h[j] += b[i]
        elif m <= j < 2 * m:
            h[j - m] -= b[i]
    val = sum(gi * hi for gi, hi in zip(g, h))
    return val, h


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------

class PhaseMachine:
    def __init__(self, n, edges, params=None, trace=None):
        self.params = params or PhaseParams()
        self.trace = trace
        self.n = n
        self.edges = {}
        self.adj = [[] for _ in range(n)]
        for (u, v) in edges:
            if u == v:
                raise ValueError("loop")
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError("endpoint out of range")
            eid = len(self.edges)
            self.edges[eid] = (u, v)
            self.adj[u].append((v, eid))
            self.adj[v].append((u, eid))
        if max((len(a) for a in self.adj), default=0) > 3:
            raise ValueError("phase machine is subcubic (max degree 3)")
        self._pair_eid = {frozenset(e): k for k, e in self.edges.items()}
        self.bins = {}
        self.rows = set()
        self.mu = {}
        self.U = {}
        self.blockers = []
        self.where = {}
        self.ledgers = []
        self.tick = 0
        self.gains = 0
        self.g_star = 0
        self.L = 0
        self.K = 0
        self.b_per = {}
        self.actions = 0
        self.pricing_steps = 0
        self._inc_row = defaultdict(set)
        self._inc_bin = defaultdict(set)
        self._next_bid = 0

    # -- helpers ------------------------------------------------------------

    def _tr(self, ev):
        if self.trace is not None:
            self.trace.append(ev)

    def _contact(self, row, bid):
        best = None
        for (nbr, eid) in self.adj[row]:
            if self.where[nbr] == ("bin", bid):
                if best is None or eid < best[1]:
                    best = (nbr, eid)
        if best is None:
            raise AssertionError(f"row {row} not adjacent to bin {bid}")
        return best[0]

    def _add_incidence(self, key):
        self._inc_row[key[0]].add(key)
        self._inc_bin[key[1]].add(key)

    def _del_incidence(self, key):
        self._inc_row[key[0]].discard(key)
        self._inc_bin[key[1]].discard(key)

    def _y(self, lev, bid):
        return self.L - sum(self.ledgers[lev].x.get(k, 0)
                            for k in self._inc_bin[bid])

    def _p(self, lev, bid):
        return self.ledgers[lev].p.get(bid, 0)

    def _slack(self, bid):
        return (self._y(0, bid) + self._p(0, bid)
                - self.params.C * self.bins[bid].weight)

    def _captive_rows(self, X):
        Xs = set(X)
        out = []
        for r in self.rows:
            nbs = set()
            for (nbr, _e) in self.adj[r]:
                loc = self.where[nbr]
                if loc[0] == "bin":
                    nbs.add(loc[1])
            if nbs <= Xs:
                out.append(r)
        return out

    def _h(self, X):
        return len(set(X)) - len(self._captive_rows(X))

    def _row_nb(self, r):
        return {loc[1] for (nbr, _e) in self.adj[r]
                if (loc := self.where[nbr])[0] == "bin"}

    # -- run ------------------------------------------------------------------

    def run(self):
        if self.n == 0:
            return []
        if self.n % 2 == 1:
            return None
        M = []
        while True:
            K = self.n // 2 - len(M)
            if K == 0:
                return self._checked_matching(M)
            res = self._run_phase(M, K)
            if res is None:
                return None
            M = res

    def _checked_matching(self, M):
        seen = set()
        for (u, v) in M:
            assert frozenset((u, v)) in self._pair_eid, f"({u},{v}) not an edge"
            assert u not in seen and v not in seen, "not a matching"
            seen.add(u)
            seen.add(v)
        return M

    # -- phase ------------------------------------------------------------------

    def _init_phase(self, M, K):
        p = self.params
        self.K = K
        self.g_star = max(1, K // (100 * p.C))
        self.L = _ceil_div(p.l_mult * p.C * self.n, K)
        s = max(1, ceil(log2(max(2, self.n))))
        B = 2 * _ceil_div(s, p.a)
        self.b_per = {j: B ** (p.a - j) for j in range(1, p.a + 1)}
        self.tick = 0
        self.gains = 0
        self.actions = 0
        self.pricing_steps = 0
        self.bins = {}
        self.rows = set()
        self.mu = {}
        self.U = {}
        self.blockers = []
        self.where = {}
        self._inc_row = defaultdict(set)
        self._inc_bin = defaultdict(set)
        self._next_bid = 0
        for (u, v) in M:
            self.U[u] = v
            self.U[v] = u
            self.where[u] = ("u",)
            self.where[v] = ("u",)
        for v in range(self.n):
            if self.where.get(v) is None:
                bid = self._next_bid
                self._next_bid += 1
                self.bins[bid] = Bin(bid, frozenset([v]), 1, v, None,
                                     ("leaf", v))
                self.where[v] = ("bin", bid)
        self.ledgers = [Ledger(j, p.C - j) for j in range(p.a + 1)]
        self._tr(("phase", K, self.g_star, self.L))

    def _run_phase(self, M, K):
        p = self.params
        self._init_phase(M, K)
        while True:
            if p.check_invariants:
                self._check_settled()
            if (self.actions > p.max_actions
                    or self.pricing_steps > p.max_pricing_steps):
                raise RuntimeError("phase budget exceeded (reference cap)")
            act = self._find_action()
            if act[0] == "pair":
                t0 = self.tick
                self.actions += 1
                ended = self._pair_action(act)
                if ended:
                    return self._materialize_all()
                self._project(t0)
            elif act[0] == "expand":
                t0 = self.tick
                self.actions += 1
                self._expand_action(act)
                self._project(t0)
            else:
                self.pricing_steps += 1
                if not self._pricing_step():
                    self._tr(("low",))
                    return None

    def _find_action(self):
        C = self.params.C
        for eid, (u, v) in self.edges.items():
            lu, lv = self.where[u], self.where[v]
            if lu[0] == "bin" and lv[0] == "bin" and lu[1] != lv[1]:
                return ("pair", lu[1], lv[1], eid, u, v)
        for eid, (u, v) in self.edges.items():
            lu, lv = self.where[u], self.where[v]
            if lu[0] == "bin" and lv[0] == "u":
                if self._slack(lu[1]) >= 4 * C:
                    return ("expand", lu[1], v, eid)
            elif lv[0] == "bin" and lu[0] == "u":
                if self._slack(lv[1]) >= 4 * C:
                    return ("expand", lv[1], u, eid)
        return ("price",)

    # -- settled-state invariant check ------------------------------------------

    def _check_settled(self):
        p = self.params
        targets = [self.mu[r][0] for r in self.rows]
        assert len(targets) == len(set(targets)), "mu not injective"
        for r in self.rows:
            bid, eid = self.mu[r]
            assert self.where[r] == ("row",)
            assert self.bins[bid].assigned == (r, eid)
            u, v = self.edges[eid]
            assert ((u == r and self.where[v] == ("bin", bid))
                    or (v == r and self.where[u] == ("bin", bid))), \
                "assigned edge must be an incidence"
        for bid, b in self.bins.items():
            for v in b.vertices:
                assert self.where[v] == ("bin", bid)
            if b.assigned is None:
                assert b.exposure in b.vertices
        for v, m in self.U.items():
            assert self.where[v] == ("u",) and self.U[m] == v
        for r in self.rows:
            for (nbr, eid) in self.adj[r]:
                loc = self.where[nbr]
                if loc[0] == "bin":
                    assert (r, loc[1], eid) in self._inc_row[r], \
                        "incidence bookkeeping"
        for lev, led in enumerate(self.ledgers):
            for r in self.rows:
                tot = sum(led.x.get(k, 0) for k in self._inc_row[r])
                assert tot == self.L, f"row sum {tot} != L at level {lev}"
            for bid, b in self.bins.items():
                y = self._y(lev, bid)
                pp = self._p(lev, bid)
                assert pp >= 0, "negative debt"
                assert y + pp >= led.c * b.weight, \
                    f"invariant (3.4) fails at level {lev} bin {bid}"
        assert all(self._p(p.a, bid) == 0 for bid in self.bins), "bottom debt"
        all_keys = [k for r in self.rows for k in self._inc_row[r]]
        for k in all_keys:
            assert self.ledgers[0].x.get(k, 0) > 0, "top x positivity"
        assert len(all_keys) == len(set(all_keys))
        for bid in self.bins:
            assert self._slack(bid) > 0, "top slack positivity"
        # full vertex coverage
        counts = [0] * self.n
        for bid, b in self.bins.items():
            for v in b.vertices:
                counts[v] += 1
        for v in self.U:
            counts[v] += 1
        for v in self.blockers:
            counts[v] += 1
        for r in self.rows:
            counts[r] += 1
        assert all(c == 1 for c in counts), \
            f"vertex coverage broken: {[i for i,c in enumerate(counts) if c!=1]}"
        K_rep = (len(self.blockers) + sum(
            1 for b in self.bins.values() if b.assigned is None)) // 2
        assert 2 * K_rep == len(self.blockers) + sum(
            1 for b in self.bins.values() if b.assigned is None), \
            "unmatched count must be even"
        assert K_rep == self.K, f"represented deficit {K_rep} != {self.K}"

    # -- action 1: pair -----------------------------------------------------------

    def _pair_action(self, act):
        """Returns True iff the phase ended (g*-th gain)."""
        _tag, i, j, eid, ue, ve = act
        min_h, X = self._pair_mincut(i, j)
        if min_h >= 2:
            return self._do_gain(i, j, eid, ue, ve)
        assert min_h == 1, f"min_h = {min_h}; h >= 1 at settled states"
        assert i in X and j in X and len(X) >= 2
        self._do_contract(X, eid)
        return False

    def _pair_mincut(self, i, j):
        rows_nb = {r: self._row_nb(r) for r in self.rows}
        return pair_decision(rows_nb, list(self.bins), i, j)

    def _do_gain(self, i, j, eid, ue, ve):
        self._free_bin(i, avoid=())
        self._free_bin(j, avoid=(i,))
        assert self.bins[i].assigned is None
        assert self.bins[j].assigned is None
        pairs_i = self.materialize(self.bins[i], ue)
        pairs_j = self.materialize(self.bins[j], ve)
        for (u, v) in pairs_i + pairs_j:
            self._add_U(u, v)
        self._add_U(ue, ve)
        self.K -= 1
        self.gains += 1
        self._tr(("gain", i, j, eid))
        if self.gains == self.g_star:
            # phase ends without ledger repair; ledgers are discarded, so
            # just drop the state structures of the two dissolved bins
            for bid in (i, j):
                for k in list(self._inc_bin[bid]):
                    self._del_incidence(k)
                self._remove_bin_state(bid)
            return True
        # deletion transformation (Lemma 3.3) on every ledger, R = {i, j}
        self._deletion_transform({i, j})
        self._remove_bin_state(i)
        self._remove_bin_state(j)
        self.tick += self.L
        return False

    def _add_U(self, u, v):
        assert self.where[u] != ("u",) and self.where[v] != ("u",)
        assert frozenset((u, v)) in self._pair_eid, "U pairs must be edges"
        self.U[u] = v
        self.U[v] = u
        self.where[u] = ("u",)
        self.where[v] = ("u",)

    def _remove_bin_state(self, bid):
        """Drop bin from state maps (ledger/incidence edits done by caller)."""
        self.bins.pop(bid)
        assert not self._inc_bin[bid], "incidences left behind"

    def _free_bin(self, b0, avoid=()):
        """Rearrange mu so b0 becomes free: deterministic BFS alternating
        path in the full-neighbor reverse graph (Section 3.7 substitute)."""
        if self.bins[b0].assigned is None:
            return
        parent = {}
        seen = {b0}
        q = deque([b0])
        target = None
        while q and target is None:
            d = q.popleft()
            r, _aeid = self.bins[d].assigned
            for (nbr, eid) in self.adj[r]:
                loc = self.where[nbr]
                if loc[0] != "bin":
                    continue
                d2 = loc[1]
                if d2 == d or d2 in seen or d2 in avoid:
                    continue
                seen.add(d2)
                parent[d2] = (d, r, eid)
                if self.bins[d2].assigned is None:
                    target = d2
                    break
                q.append(d2)
        assert target is not None, \
            "freeing walk exhausted (contradicts the Hall surplus)"
        path = [target]
        while path[0] != b0:
            path.insert(0, parent[path[0]][0])
        # rows currently assigned along the path (captured before any edit)
        rows = [self.bins[d].assigned for d in path[:-1]]
        assert all(a is not None for a in rows)
        # b0 loses its row; each later bin takes the previous bin's row
        r0, e0 = rows[0]
        self.bins[b0].assigned = None
        self.bins[b0].exposure = self._contact(r0, b0)
        for t in range(1, len(path)):
            _pd, r, eid = parent[path[t]]
            u, v = self.edges[eid]
            assert r == rows[t - 1][0]
            contact = u if self.where[u] == ("bin", path[t]) else v
            b = self.bins[path[t]]
            b.assigned = (r, eid)
            b.exposure = contact
            self.mu[r] = (path[t], eid)
        assert self.bins[b0].assigned is None

    def _do_contract(self, X, join_eid):
        p = self.params
        X = set(X)
        assert self._h(X) == 1, "contraction set must be tight"
        captives = self._captive_rows(X)
        assert len(captives) == len(X) - 1
        capt_set = set(captives)
        exterior = [r for r in self.rows
                    if self.mu[r][0] in X and r not in capt_set]
        assert len(exterior) <= 1, "at most one exterior assignment enters X"

        if p.debug_edits:
            ssum = sum(self._slack(b) for b in X)

        node = CNode()
        node.join_eid = join_eid
        new_vertices = set()
        weight = 0
        for bid in sorted(X):
            b = self.bins[bid]
            cid = ("bin", bid)
            node.children.append(cid)
            node.ctype[cid] = "bin"
            node.cobj[cid] = b
            node.cverts[cid] = b.vertices
            new_vertices |= b.vertices
            weight += b.weight
        for r in captives:
            cid = ("row", r)
            node.children.append(cid)
            node.ctype[cid] = "row"
            node.cverts[cid] = frozenset([r])
            new_vertices.add(r)
            node.cap_assign[cid] = ("bin", self.mu[r][0])
            node.row_nb[cid] = [("bin", bb) for bb in sorted(self._row_nb(r) & X)]
        unused_set = {("bin", b) for b in X} - set(node.cap_assign.values())
        assert len(unused_set) == 1
        node.unused = next(iter(unused_set))

        if exterior:
            er = exterior[0]
            eeid = self.mu[er][1]
            u, v = self.edges[eeid]
            expo = u if u != er else v
            assert expo in new_vertices
            assigned = (er, eeid)
        else:
            expo = node.cobj[node.unused].exposure
            assigned = None
        assert len(new_vertices) % 2 == 1
        new_bid = self._next_bid
        self._next_bid += 1
        self.bins[new_bid] = Bin(new_bid, frozenset(new_vertices), weight,
                                 expo, assigned, node)

        # ---- edit plan (before any structure change) ----
        retarget = []
        captive_keys = []
        for bid in X:
            for k in list(self._inc_bin[bid]):
                if k[0] in capt_set:
                    captive_keys.append(k)
                else:
                    retarget.append((k, (k[0], new_bid, k[2])))
        swallowed_keys = []
        for r in list(self.rows):
            if r in capt_set:
                continue
            for (nbr, eid) in self.adj[r]:
                if nbr in capt_set:
                    swallowed_keys.append((r, new_bid, eid))

        # ---- ledger edits, each level ----
        for lev in range(p.a + 1):
            led = self.ledgers[lev]
            for k in captive_keys:
                led.x.pop(k, None)
            for (ok, nk) in retarget:
                val = led.x.pop(ok, 0)
                if val:
                    led.x[nk] = led.x.get(nk, 0) + val
            agg = sum(led.p.pop(bid, 0) for bid in X)
            if agg:
                led.p[new_bid] = led.p.get(new_bid, 0) + agg

        # ---- structure edits (once) ----
        for k in captive_keys:
            self._del_incidence(k)
        for (ok, nk) in retarget:
            self._del_incidence(ok)
            self._add_incidence(nk)
        for nk in swallowed_keys:
            self._add_incidence(nk)

        # ---- top positivity: pad the new zero incidences ----
        top = self.ledgers[0]
        for nk in swallowed_keys:
            self._donor_pad(nk)
        for k in list(top.x):
            if k[1] == new_bid:
                assert top.x[k] > 0, "top incidence at contracted bin must stay positive"

        # ---- state bookkeeping ----
        for bid in X:
            self.bins.pop(bid)
        for r in captives:
            self.rows.discard(r)
            self.mu.pop(r, None)
        for r in exterior:
            self.mu[r] = (new_bid, self.mu[r][1])
        for v in new_vertices:
            self.where[v] = ("bin", new_bid)

        if p.debug_edits:
            # exact aggregation plus Lemma 3.4 paddings: 4-delta transfers
            # from donors inside X raise the aggregate; none lower it
            assert self._slack(new_bid) >= ssum, "top s aggregation broken"
        self.tick += 1
        self._tr(("contract", frozenset(X), new_bid))

    def _donor_pad(self, key):
        """Lemma 3.4: a new/low positive allocation receives 4-delta from a
        large donor in its row; 4-delta debt is added at the destination."""
        p = self.params
        d4 = 4 * p.delta
        r = key[0]
        entries = [(self.ledgers[0].x.get(k, 0), k)
                   for k in self._inc_row[r] if k != key]
        assert entries, "no donor available in row"
        donor_val, donor_key = max(entries)
        assert donor_val >= Fraction(self.L, 3), "donor must be large (>= L/3)"
        assert donor_val - d4 > 3 * p.delta
        top = self.ledgers[0]
        top.x[donor_key] = donor_val - d4
        top.x[key] = top.x.get(key, 0) + d4
        top.p[key[1]] = self._p(0, key[1]) + d4

    # -- action 2: expansion --------------------------------------------------------

    def _expand_action(self, act):
        _tag, d, v, eid = act
        p = self.params
        w = self.U[v]
        assert self.where[w] == ("u",)
        old_eid = None
        for (nbr, e2) in self.adj[v]:
            if nbr == w:
                old_eid = e2 if old_eid is None else min(old_eid, e2)
        assert old_eid is not None
        assert self._slack(d) >= 3 * p.C, "barrier (3.10) actual side"
        del self.U[v]
        del self.U[w]
        wbid = self._next_bid
        self._next_bid += 1
        self.bins[wbid] = Bin(wbid, frozenset([w]), 1, w, (v, old_eid),
                              ("leaf", w))
        self.where[w] = ("bin", wbid)
        self.rows.add(v)
        self.mu[v] = (wbid, old_eid)
        self.where[v] = ("row",)
        for (nbr, e2) in self.adj[v]:
            loc = self.where[nbr]
            if loc[0] == "bin":
                self._add_incidence((v, loc[1], e2))
        # w's edges to OTHER rows also become incidences (w is a bin now)
        w_new_incs = []
        for (nbr, e2) in self.adj[w]:
            if nbr != v and self.where[nbr] == ("row",):
                w_new_incs.append((nbr, wbid, e2))
        for k in w_new_incs:
            self._add_incidence(k)
        # integral ledgers: whole L on the assigned child, debt c_j (j >= 1)
        for lev in range(1, p.a + 1):
            led = self.ledgers[lev]
            led.x[(v, wbid, old_eid)] = self.L
            led.p[wbid] = led.c
        # top ledger: C+1 on the trigger incidence, L-C-1 on the child
        top = self.ledgers[0]
        top.x[(v, d, eid)] = p.C + 1
        top.x[(v, wbid, old_eid)] = self.L - p.C - 1
        if p.debug_edits:
            assert self._slack(wbid) == 1, "child top slack must be 1"
            assert self._p(0, wbid) == 0, "no initial top child debt"
            assert self._slack(d) > 0, "parent stays positive after paying"
        for k in list(self._inc_row[v]):
            if k in ((v, d, eid), (v, wbid, old_eid)):
                continue
            if top.x.get(k, 0) == 0:
                self._donor_pad(k)
        for k in w_new_incs:
            self._donor_pad(k)   # 4-delta padding from a donor in k's row
        self.tick += 1
        self._tr(("expand", d, v, wbid))

    # -- deletion transformation (Lemma 3.3) -------------------------------------------

    def _deletion_transform(self, R):
        """Remove rows assigned into R; redirect each surviving row's
        allocation into R to its current assigned edge, adding equal debt at
        the recipient bin.  Ledger-only + row removal; callers remove bins."""
        R = set(R)
        p = self.params
        removed = [r for r in self.rows if self.mu[r][0] in R]
        rem_set = set(removed)
        doomed = []
        for bid in R:
            for k in list(self._inc_bin[bid]):
                doomed.append((k, k[0] in rem_set))
        extra_removed = [k for r in removed for k in self._inc_row[r]
                         if k[1] not in R]
        # ledger edits per level
        for lev in range(p.a + 1):
            led = self.ledgers[lev]
            for k in extra_removed:
                led.x.pop(k, None)
            for (k, is_rem) in doomed:
                amt = led.x.pop(k, 0)
                if amt and not is_rem:
                    r = k[0]
                    tk = (r, self.mu[r][0], self.mu[r][1])
                    led.x[tk] = led.x.get(tk, 0) + amt
                    led.p[self.mu[r][0]] = self._p(lev, self.mu[r][0]) + amt
            for bid in R:
                led.p.pop(bid, None)
        # structure edits (once)
        for (k, _is_rem) in doomed:
            self._del_incidence(k)
        for k in extra_removed:
            self._del_incidence(k)
        for r in removed:
            self.rows.discard(r)
            self.mu.pop(r, None)
            self.where[r] = ("gone",)
        self._tr(("deletion", frozenset(R), len(removed)))

    # -- projection (Section 3.5) --------------------------------------------------------

    def _project(self, t0):
        p = self.params
        t1 = self.tick
        js = [j for j in range(1, p.a + 1)
              if t1 // self.b_per[j] > t0 // self.b_per[j]]
        assert js, "a reset level exists since b_a = 1"
        j = min(js)
        c = p.C - j
        if j == 1:
            parent_x = {}
            for r in self.rows:
                bid, eid = self.mu[r]
                parent_x[(r, bid, eid)] = self.L
        else:
            parent_x = dict(self.ledgers[j - 1].x)
        all_keys = [k for r in self.rows for k in self._inc_row[r]]
        weights = {bid: b.weight for bid, b in self.bins.items()}
        ex = exchange_projection(self.L, c, weights, all_keys, parent_x)
        if ex[0]:
            _ok, xnew = ex
            fresh = Ledger(j, c)
            fresh.x = {k: v for k, v in xnew.items() if v}
            for jj in range(j, p.a + 1):
                cp = Ledger(jj, p.C - jj)
                cp.x = dict(fresh.x)
                self.ledgers[jj] = cp
            if p.check_invariants:
                for bid in self.bins:
                    y = self.L - sum(self.ledgers[j].x.get(k, 0)
                                     for k in self._inc_bin[bid])
                    assert y >= c * weights[bid], "outcome (i) floor"
            self._tr(("project", j, None))
        else:
            _ok, R, xnew = ex
            self._peel(set(R), j, xnew)

    def _peel(self, R, j, xnew):
        p = self.params
        if p.debug_edits:
            hR = self._h(R)
            wR = sum(self.bins[bid].weight for bid in R)
            assert self.L * hR < (p.C - j) * wR, "(3.13) violated"
            f = sum(1 for bid in R if self.bins[bid].assigned is None)
            assert f <= hR, "(3.14) violated"
            psum_before = [sum(self.ledgers[l].p.values()) for l in range(j)]
            wR_dbg = wR
        self._deletion_transform(R)
        for bid in sorted(R):
            b = self.bins[bid]
            if b.assigned is not None:
                r, aeid = b.assigned
                u, v = self.edges[aeid]
                contact = u if u in b.vertices else v
                for (x, y2) in self.materialize(b, contact):
                    self._add_U(x, y2)
                self._add_U(contact, r)
            else:
                for (x, y2) in self.materialize(b, b.exposure):
                    self._add_U(x, y2)
                self.blockers.append(b.exposure)
                self.where[b.exposure] = ("blk",)
            self._remove_bin_state(bid)
        c = p.C - j
        fresh = Ledger(j, c)
        fresh.x = {k: v for k, v in xnew.items()
                   if v and k[0] in self.rows and k[1] in self.bins}
        for jj in range(j, p.a + 1):
            cp = Ledger(jj, p.C - jj)
            cp.x = dict(fresh.x)
            self.ledgers[jj] = cp
        if p.debug_edits:
            for l in range(j):
                after = sum(self.ledgers[l].p.values())
                assert psum_before[l] - after >= wR_dbg, \
                    f"coarser level {l} must lose >= w(R) net debt"
        if p.check_invariants:
            for bid in self.bins:
                y = self.L - sum(self.ledgers[j].x.get(k, 0)
                                 for k in self._inc_bin[bid])
                assert y >= c * self.bins[bid].weight, "post-peel floor"
        self._tr(("peel", frozenset(R), j))

    # -- pricing (Section 3.2, exact substitute) --------------------------------------

    def _top_arcs(self):
        arcs = []
        for r in self.rows:
            for k in self._inc_row[r]:
                arcs.append((("r", r), ("b", k[1]), "x", k))
        for bid in self.bins:
            arcs.append((("aux",), ("b", bid), "s", bid))
        return arcs

    def _coord(self, kind, key):
        if kind == "x":
            return self.ledgers[0].x.get(key, 0)
        return self._slack(key)

    def _pricing_step(self):
        """Exact priced step (Lemma 5.1 interface, exact substitute).

        Computes the exact steepest circulation direction (max ratio
        g.h / sum_e |h_e|/f_e) by an exact rational LP.  Lemma 5.1's LOW
        criterion: LOW is impossible whenever a direction with ratio
        >= 1/2 exists, so we report LOW exactly when the max ratio is
        <= 1/2 - gamma (gamma = 1/1000 guards the closed/open boundary).
        Otherwise take the steepest direction and run an exact rational
        line search on Phi along it.
        """
        p = self.params
        arcs = self._top_arcs()
        if not arcs:
            return False   # no coordinates: trivially optimal -> LOW
        vals = [self._coord(a[2], a[3]) for a in arcs]
        assert all(v > 0 for v in vals), "pricing needs positive coordinates"
        nodes = sorted({x for a in arcs for x in a[:2]})
        g = [Fraction(1, 1) / v for v in vals]
        w = [Fraction(1, 1) / v for v in vals]
        value, h = max_ratio_direction(nodes, [(a[0], a[1]) for a in arcs],
                                       g, w)
        lam = Fraction(1, 2) - Fraction(1, 1000)
        if value <= lam:
            return False   # LOW: no direction with ratio >= 1/2 - gamma
        coords = [(arcs[i][2], arcs[i][3], vals[i], hv)
                  for i, hv in enumerate(h) if hv]
        assert coords
        slope = sum((Fraction(1, 1) / v) * hv for (_k, _key, v, hv) in coords)
        assert slope > 0, "LP direction must ascend"
        tmax = None
        for (_k, _key, v, hv) in coords:
            if hv < 0:
                t = (-v) / hv
                if tmax is None or t < tmax:
                    tmax = t
        assert tmax is not None and tmax > 0

        def dphi(t):
            s = Fraction(0)
            for (_k, _key, v, hv) in coords:
                s += hv / (v + t * hv)
            return s

        assert dphi(Fraction(0)) > 0
        lo, hi = Fraction(0), tmax
        for _ in range(p.line_iters):
            mid = (lo + hi) / 2
            if mid == lo or mid == hi:
                break
            if dphi(mid) > 0:
                lo = mid
            else:
                hi = mid
        t = lo
        assert dphi(t) >= 0, "line search must not decrease Phi"
        top = self.ledgers[0]
        for (kind, key, v, hv) in coords:
            if kind == "x":
                top.x[key] = v + t * hv
        d4 = 4 * p.delta
        thr = 3 * p.delta
        for k in list(top.x):
            xv = top.x[k]
            assert xv > 0, "nonpositive top allocation after pricing"
            if xv <= thr:
                self._donor_pad(k)
        for bid in list(self.bins):
            sv = self._slack(bid)
            assert sv > 0, "nonpositive top slack after pricing"
            if sv <= thr:
                top.p[bid] = self._p(0, bid) + d4
        self._tr(("price", float(value)))
        return True

    # -- materialization (Section 3.8) ------------------------------------------------

    def materialize(self, b, exposure):
        """Internal matching (list of vertex pairs) covering b's vertices
        except `exposure`.  Pure: does not mutate the state."""
        assert exposure in b.vertices
        return self._materialize(b, exposure)

    def _materialize(self, b, exposure):
        tree = b.tree
        if isinstance(tree, tuple):
            assert tree[0] == "leaf" and exposure == tree[1]
            return []
        node = tree
        cid = node.child_of(exposure)
        # rows to match and forbidden child bins (Lemma 3.8):
        #  - exposure in child bin d: forbid d; d's captive row loses its
        #    assignment but stays available (at most one augmentation);
        #  - exposure is swallowed row r: delete r; reserve i, j (endpoints
        #    of the stored joining edge); their captives stay available
        #    (at most two augmentations; inequalities (3.16)).
        if node.ctype[cid] == "bin":
            forbidden = {cid}
            rows = list(node.cap_assign)
        else:
            u, v = self.edges[node.join_eid]
            ci = node.child_of(u)
            cj = node.child_of(v)
            forbidden = {ci, cj}
            rows = [r for r in node.cap_assign if r != cid]
        targets = {c for c in node.children
                   if node.ctype[c] == "bin" and c not in forbidden}
        assert len(rows) == len(targets), "row/bin count mismatch"
        nb = {r: [bb for bb in node.row_nb[r] if bb in targets]
              for r in rows}
        match = {r: bb for r, bb in node.cap_assign.items()
                 if r in rows and bb not in forbidden}
        bmatch = {bb: r for r, bb in match.items()}
        assert len(match) == len(bmatch)

        def try_aug(r, seen):
            for bb in nb[r]:
                if bb in seen:
                    continue
                seen.add(bb)
                if bb not in bmatch or try_aug(bmatch[bb], seen):
                    bmatch[bb] = r
                    match[r] = bb
                    return True
            return False

        for r in rows:
            if r not in match:
                assert try_aug(r, set()), "Hall failure in materialization"
        assert len(match) == len(rows), "rows unmatched"
        assert set(match.values()) == targets, "child bins uncovered"

        pairs = []
        for r, bb in sorted(match.items()):
            contact = self._node_contact(node, r, bb)
            pairs.append((contact, r[1]))          # row pair (assigned edge)
            pairs.extend(self._materialize(node.cobj[bb], contact))
        if node.ctype[cid] == "bin":
            pairs.extend(self._materialize(node.cobj[cid], exposure))
        else:
            u, v = self.edges[node.join_eid]
            ci = node.child_of(u)
            cj = node.child_of(v)
            pairs.append((u, v))                    # the stored joining edge
            pairs.extend(self._materialize(node.cobj[ci], u))
            pairs.extend(self._materialize(node.cobj[cj], v))
        return pairs

    def _node_contact(self, node, rid, bid):
        r = rid[1]
        vs = node.cverts[bid]
        best = None
        for (nbr, eid) in self.adj[r]:
            if nbr in vs and (best is None or eid < best[1]):
                best = (nbr, eid)
        assert best is not None, "captive row not adjacent to its child bin"
        return best[0]

    def _materialize_all(self):
        pairs = []
        for v, m in self.U.items():
            if v < m:
                pairs.append((v, m))
        for bid in sorted(self.bins):
            b = self.bins[bid]
            if b.assigned is not None:
                r, aeid = b.assigned
                u, v = self.edges[aeid]
                contact = u if u in b.vertices else v
                pairs.extend(self.materialize(b, contact))
                pairs.append((contact, r))
            else:
                pairs.extend(self.materialize(b, b.exposure))
        nfree = sum(1 for b in self.bins.values() if b.assigned is None)
        assert 2 * len(pairs) + nfree + len(self.blockers) == self.n
        assert len(pairs) == self.n // 2 - self.K
        return self._checked_matching(pairs)


def perfect_matching(n, edges, params=None):
    """Decide perfect matching on a subcubic loopless graph; return an
    explicit perfect matching (list of vertex pairs) or None."""
    return PhaseMachine(n, edges, params).run()
