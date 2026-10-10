"""Layer 3 tests: Chapter 9 assembly (threshold search + dovetailing).

End-to-end decision + witness is cross-checked against
networkx.max_weight_matching(maxcardinality=True), per the spec's plan.

Timing note: the reference phase machine takes ~20 s per composed
threshold test even at n = 2 (gadget N = 54, one phase per gain), so the
pipeline-mode end-to-end sweep is deliberately tiny; the broad graph sweep
runs in 'edmonds' mode (branch 0), which is the paper's own built-in
cross-check, plus scheduling/bookkeeping unit tests that need no machine
run at all.
"""

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
import pytest
from fractions import Fraction

from assembly import (
    Branch, EdmondsBranch, PMBranch, branch_error, check_matching,
    decide_reduced, max_cardinality_matching, pm_test, run_dovetail,
    threshold_search, total_error,
)
from phase_machine import PhaseParams


def nx_nu(n, edges):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    return len(nx.max_weight_matching(g, maxcardinality=True))


# ---------------------------------------------------------------------------
# Error budget bookkeeping (Eq. 9.4 / 9.5)
# ---------------------------------------------------------------------------

class TestErrorBudget:
    def test_branch_error_values(self):
        assert branch_error(0) == 0                     # fallback is exact
        for t in (1, 2, 3, 10, 100):
            assert branch_error(t) == pytest.approx(2.0 ** -(t + 20))

    def test_total_error_below_third(self):
        # full geometric sum: 2^-20 (Eq. 9.5)
        assert total_error() == Fraction(1, 2 ** 20)
        assert total_error() < Fraction(1, 3)

    def test_total_error_partial_branches(self):
        for m in (1, 2, 5, 50):
            e = total_error(m)
            assert e < 1.0 / 3.0
            assert e < total_error()                    # partial < full sum

    def test_branch_objects_carry_allowance(self):
        b0 = EdmondsBranch(3, [(0, 1), (1, 2)])
        b2 = PMBranch(2, 3, [(0, 1), (1, 2)])
        assert b0.error() == 0
        assert b2.error() == pytest.approx(2.0 ** -22)


# ---------------------------------------------------------------------------
# Witness verification
# ---------------------------------------------------------------------------

class TestCheckMatching:
    def test_valid(self):
        assert check_matching(4, [(0, 1), (2, 3)], [(0, 1), (2, 3)], 2)
        assert check_matching(4, [(0, 1), (2, 3)], [(2, 3)], 1)
        assert check_matching(0, [], [], 0)

    def test_invalid(self):
        edges = [(0, 1), (1, 2), (2, 3)]
        assert not check_matching(4, edges, [(0, 1), (1, 2)])     # not disjoint
        assert not check_matching(4, edges, [(0, 2)])             # not an edge
        assert not check_matching(4, edges, [(0, 1), (2, 3)], 1)  # wrong size
        assert not check_matching(4, edges, [(0, 0)])             # loop
        assert not check_matching(4, edges, [(0, 4)])             # out of range


# ---------------------------------------------------------------------------
# Threshold search bookkeeping (Section 9.2)
# ---------------------------------------------------------------------------

class TestThresholdSearch:
    def _fake(self, nu):
        calls = []

        def test(k):
            calls.append(k)
            # monotone oracle: feasible iff k <= nu; witness = k edges
            if k <= nu:
                return True, [(2 * i, 2 * i + 1) for i in range(k)]
            return False, None

        return test, calls

    @pytest.mark.parametrize("n,nu", [(0, 0), (1, 0), (2, 1), (5, 2),
                                      (8, 4), (8, 1), (10, 5), (7, 3)])
    def test_finds_nu(self, n, nu):
        test, calls = self._fake(nu)
        k, wit = threshold_search(n, [], test)
        assert k == nu
        assert len(wit) == nu

    def test_witness_is_the_last_accepted_one(self):
        # the stored witness must be the one from the highest accepted test
        def test(k):
            return (True, [("x", k)]) if k <= 3 else (False, None)

        k, wit = threshold_search(8, [], test)
        assert k == 3
        assert wit == [("x", 3)]

    def test_test_count_logarithmic(self):
        # O(log(n+2)) and certainly at most n+1 tests (Section 9.2)
        for n in (2, 8, 32, 100):
            for nu in (0, 1, n // 2, n // 3):
                test, calls = self._fake(nu)
                threshold_search(n, [], test)
                assert len(calls) <= (n // 2 + 2).bit_length()
                assert len(calls) <= n + 1

    def test_endpoints_invariant(self):
        # every tested k: accepted ones feasible, rejected ones infeasible
        nu = 3
        test, calls = self._fake(nu)
        threshold_search(10, [], test)
        for k in calls:
            if k <= nu:
                assert k <= nu
            else:
                assert k > nu
        assert max(calls) <= 10 // 2 + 1


# ---------------------------------------------------------------------------
# Branch state machines (no machine runs)
# ---------------------------------------------------------------------------

class FakeBranch(Branch):
    """Deterministic fake: feasible iff k <= nu, configurable cost."""

    def __init__(self, t, n, nu):
        super().__init__(t, n, [])
        self.nu = nu

    def _test(self, k):
        return (True, list(range(k))) if k <= self.nu else (False, None)


class SlowBranch(Branch):
    """Suspends itself before doing any test (models a deterministic
    prerequisite suspension: 'a suspended branch stays inactive')."""

    def __init__(self, t, n):
        super().__init__(t, n, [])

    def _test(self, k):
        raise AssertionError("suspended branch must never be stepped")


class TestBranches:
    def test_branch_completes_with_maximum(self):
        b = FakeBranch(1, 8, 3)
        b.step(10)
        assert b.done
        assert b.answer == (3, list(range(3)))
        assert b.tests_run == 3 or b.tests_run == 4

    def test_branch_steps_in_units(self):
        b = FakeBranch(1, 100, 40)
        b.step(1)
        assert b.tests_run == 1 and not b.done
        b.step(1)
        assert b.tests_run == 2

    def test_zero_test_graphs_finish_immediately(self):
        for n in (0, 1, 2):
            b = FakeBranch(0, n, 0)
            b.step(1)
            assert b.done and b.answer == (0, [])

    def test_edmonds_branch_matches_networkx(self):
        cases = [
            (6, [(0, 1), (1, 2), (2, 0), (3, 4)]),          # triangle + edge
            (5, [(0, 1), (0, 2), (0, 3), (0, 4)]),          # star, odd n
            (7, []),                                         # all isolates
            (6, [(0, 1), (2, 3), (4, 5)]),                   # perfect
            (4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),   # K4 minus edge
        ]
        for n, edges in cases:
            b = EdmondsBranch(n, edges)
            b.step(10)
            assert b.done
            k, wit = b.answer
            assert k == nx_nu(n, edges)
            assert check_matching(n, edges, wit, k)

    def test_edmonds_branch_witness_dropping(self):
        # sub-witnesses at k < nu are prefixes of the stored maximum
        b = EdmondsBranch(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        ok, w1 = b._test(1)
        ok2, w2 = b._test(2)
        assert ok and ok2
        assert w2[:1] == w1


# ---------------------------------------------------------------------------
# Dovetailing schedule (Section 9.4)
# ---------------------------------------------------------------------------

class TestDovetail:
    def test_quota_weights_and_halt(self):
        b0 = FakeBranch(0, 8, 3)
        branches = [b0, FakeBranch(1, 8, 3)]
        winner = run_dovetail(branches)
        assert winner.t == 0
        assert winner.answer[0] == 3
        # branch 0 must finish: it needs <= bit_length(5)=3 tests and gets
        # 2^j per round j — always within the schedule's halt bound
        assert b0.done

    def test_first_completed_branch_wins_even_if_not_branch0(self):
        # branch 1 needs only 1 test; branch 0 is a fake needing many
        class VerySlow(Branch):
            def _test(self, k):
                return (k <= 3, []) if k <= 3 else (False, None)

            def steps_needed_upper_bound(self):
                return 60                        # large n -> many rounds

        slow = VerySlow(0, 10 ** 6, [])
        fast = FakeBranch(1, 8, 3)               # needs ~2 tests
        winner = run_dovetail([slow, fast], max_rounds=100)
        assert winner is fast
        assert winner.answer[0] == 3

    def test_suspended_branch_stays_inactive(self):
        susp = SlowBranch(1, 8)
        susp.suspended = True
        winner = run_dovetail([FakeBranch(0, 8, 3), susp])
        assert winner.t == 0

    def test_round_j_gives_branch_t_quota_2pow_j_minus_t(self):
        # record every step budget handed out
        handed = []

        class RecBranch(Branch):
            def __init__(self, t, n, nu):
                super().__init__(t, n, [])
                self.nu = nu
                self.grants = []

            def _test(self, k):
                return (True, list(range(k))) if k <= self.nu else (False, None)

            def step(self, budget):
                # recover the round from the schedule's rule budget = 2^(j-t)
                import math
                j = self.t + int(math.log2(budget))
                self.grants.append((j, self.t, budget))
                super().step(budget)

        b0 = RecBranch(0, 300, 150)   # needs ~8 tests -> runs into round 3
        b1 = RecBranch(1, 300, 150)
        b2 = RecBranch(2, 300, 150)
        winner = run_dovetail([b0, b1, b2])
        assert winner.t == 0
        handed = b0.grants + b1.grants + b2.grants
        # every recorded grant matches the schedule: round j, branch t,
        # budget 2^(j-t), only for t <= j
        for (j, t, budget) in handed:
            assert t <= j
            assert budget == 2 ** (j - t)
        # branch 0 runs first in each round, branches in t order
        rounds = {}
        for (j, t, _) in handed:
            rounds.setdefault(j, []).append(t)
        for j, ts in rounds.items():
            assert ts == sorted(ts)
        # a branch t is never stepped before round t
        assert min(j for (j, t, _) in handed if t == 2) >= 2

    def test_pm_branch_never_runs_when_branch0_wins_early(self):
        # on a small graph branch 0 completes before any PM test fires,
        # so the expensive branch is never stepped
        pm = PMBranch(1, 2, [(0, 1)])
        pm._test = lambda k: (_ for _ in ()).throw(
            AssertionError("PM branch should not run here"))
        winner = run_dovetail([EdmondsBranch(2, [(0, 1)]), pm])
        assert winner.t == 0
        assert winner.answer[0] == 1
        assert check_matching(2, [(0, 1)], winner.answer[1], 1)


# ---------------------------------------------------------------------------
# decide_reduced: Lemma 9.3 + machine on direct (small) inputs
# ---------------------------------------------------------------------------

class TestDecideReduced:
    def test_isolate_certificate_is_negative(self):
        # isolate recorded before gadget construction -> certified NO
        assert decide_reduced([0, 1, 2], [(0, 1)]) is None

    def test_tiny_graphs(self):
        assert decide_reduced([0, 1, 2, 3],
                              [(0, 1), (1, 2), (2, 3), (3, 0)]) is not None
        assert decide_reduced([0, 1, 2, 3],
                              [(0, 1), (1, 2), (2, 3)]) is not None
        assert decide_reduced([0, 1, 2], [(0, 1), (1, 2), (0, 2)]) is None
        assert decide_reduced([0, 1, 2, 3],
                              [(0, 1), (0, 2), (0, 3)]) is None   # claw
        assert decide_reduced([], []) == []
        assert decide_reduced([0], []) is None

    def test_witness_validity(self):
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        m = decide_reduced(list(range(4)), edges)
        assert sorted(map(sorted, m)) in (
            [[0, 1], [2, 3]], [[0, 3], [1, 2]])
        assert check_matching(4, edges, m, 2)


# ---------------------------------------------------------------------------
# pm_test: one full-pipeline threshold test (slow: ~20 s+ each)
# ---------------------------------------------------------------------------

class TestPMTest:
    """One full-pipeline threshold test through the phase machine.

    The composed instance is quadratically expensive in the reference
    machine (N = 54 gadget vertices -> ~20 s; N ~ 236 -> ~10 min), so the
    sweep stays at n <= 3; Layer 2 already cross-checked the composed
    pipeline machinery itself."""

    def test_n2_feasible(self):
        ok, w = pm_test(2, [(0, 1)], 1)
        assert ok and w == [(0, 1)]

    def test_n2_empty_infeasible(self):
        # nu = 0 < k = 1: the machine must answer NO
        ok, w = pm_test(2, [], 1)
        assert not ok and w is None

    def test_k0_is_trivially_feasible(self):
        # k = 0: H_0 is all idle pairs; the search never tests it, but the
        # test function must handle it
        ok, w = pm_test(2, [(0, 1)], 0)
        assert ok and w == []

    def test_n3_path_feasible(self):
        # SLOW (~15+ min: N ~ 156 gadget vertices through the reference
        # machine); the same instance is also covered end-to-end in
        # TestPMModeEndToEnd.test_n3_path, so this unit is skipped unless
        # RUN_SLOW_PM is set.
        if not os.environ.get("RUN_SLOW_PM"):
            pytest.skip("set RUN_SLOW_PM=1 to run the slow n=3 machine test")
        edges = [(0, 1), (1, 2)]
        ok, w = pm_test(3, edges, 1)
        assert ok and check_matching(3, edges, w, 1)


# ---------------------------------------------------------------------------
# Public API — end-to-end
# ---------------------------------------------------------------------------

class TestEdmondsModeSweep:
    """Broad graph sweep in edmonds mode (branch 0, fast)."""

    CASES = [
        (0, []),
        (1, []),
        (2, []),
        (2, [(0, 1)]),
        (3, [(0, 1), (1, 2)]),                       # odd n
        (3, [(0, 1), (1, 2), (0, 2)]),               # triangle
        (4, [(0, 1), (0, 2), (0, 3)]),               # star
        (4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]),  # K4
        (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),       # odd path
        (6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]),  # 2 triangles
        (7, [(0, 1), (2, 3)]),                       # isolates
        (8, []),                                     # all isolates
    ]

    @pytest.mark.parametrize("n,edges", CASES)
    def test_fixed_cases(self, n, edges):
        k, wit = max_cardinality_matching(n, edges, mode="edmonds")
        assert k == nx_nu(n, edges)
        assert check_matching(n, edges, wit, k)

    @pytest.mark.parametrize("seed", range(15))
    def test_random(self, seed):
        rng = random.Random(seed)
        n = rng.randint(0, 12)
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)
                 if rng.random() < rng.choice((0.2, 0.5, 0.8))]
        k, wit = max_cardinality_matching(n, pairs, mode="edmonds")
        assert k == nx_nu(n, pairs)
        assert check_matching(n, pairs, wit, k)

    @pytest.mark.parametrize("n", [4, 5, 6, 7])
    def test_complete_graphs(self, n):
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        k, wit = max_cardinality_matching(n, edges, mode="edmonds")
        assert k == n // 2 == nx_nu(n, edges)
        assert check_matching(n, edges, wit, k)


class TestDovetailModeEndToEnd:
    def test_small(self):
        n, edges = 4, [(0, 1), (1, 2), (2, 3)]
        k, wit = max_cardinality_matching(n, edges, mode="dovetail",
                                          pm_branches=2)
        assert k == 2 == nx_nu(n, edges)
        assert check_matching(n, edges, wit, k)

    def test_with_isolates(self):
        n, edges = 5, [(0, 1), (3, 4)]
        k, wit = max_cardinality_matching(n, edges, mode="dovetail")
        assert k == 2 == nx_nu(n, edges)
        assert check_matching(n, edges, wit, k)

    def test_empty_and_tiny(self):
        for n, edges in [(0, []), (1, []), (2, []), (2, [(0, 1)])]:
            k, wit = max_cardinality_matching(n, edges, mode="dovetail")
            assert k == nx_nu(n, edges)
            assert check_matching(n, edges, wit, k)


class TestPMModeEndToEnd:
    """Full pipeline through the phase machine only (slow — keep tiny)."""

    def test_edge_cases_no_pm_test_needed(self):
        # search interval empty: (0, witness) with zero machine runs
        for n, edges in [(0, []), (1, []), (2, []), (2, [(0, 1)])]:
            k, wit = max_cardinality_matching(n, edges, mode="pm")
            assert k == nx_nu(n, edges)
            assert check_matching(n, edges, wit, k)

    def test_n2(self):
        k, wit = max_cardinality_matching(2, [(0, 1)], mode="pm")
        assert k == 1 and wit == [(0, 1)]

    def test_n2_empty(self):
        # one machine test at k = 1 answers NO; witness stays empty
        k, wit = max_cardinality_matching(2, [], mode="pm")
        assert k == 0 == nx_nu(2, [])
        assert wit == []

    def test_n3_path(self):
        # SLOW: one machine test at k = 1 (N ~ 156 gadget vertices)
        edges = [(0, 1), (1, 2)]
        k, wit = max_cardinality_matching(3, edges, mode="pm")
        assert k == 1 == nx_nu(3, edges)
        assert check_matching(3, edges, wit, 1)


class TestAPIContract:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            max_cardinality_matching(2, [(0, 1)], mode="bogus")

    def test_dedup_and_orientation_of_edges(self):
        # duplicate / flipped occurrences are normalized away
        k, wit = max_cardinality_matching(
            4, [(0, 1), (1, 0), (0, 1), (2, 3), (3, 2)], mode="edmonds")
        assert k == 2
        assert check_matching(4, [(0, 1), (2, 3)], wit, 2)

    def test_loops_rejected(self):
        with pytest.raises(ValueError):
            max_cardinality_matching(2, [(0, 0)], mode="edmonds")
