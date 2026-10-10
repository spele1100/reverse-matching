"""DynamicMatching — exact single-repair dynamic maximum-cardinality matching.

Tier 2 m1 (Strategy A: exact single-repair), per papers/dynamic-matching-spec.md.

Maintains a maximum-cardinality matching of a general (non-bipartite) graph under
edge insertions and deletions, without recomputing from scratch.

Correctness engine (spec §1.2):

* Lemma A (ν-Lipschitz): any single edge insertion or deletion changes the maximum
  matching size ν by at most 1.
* Lemma B (one-repair suffices): by Berge's theorem, after one update the current
  matching M′ (|M′| ≥ |M|−1) either is maximum or admits an augmenting path; a
  single blossom-aware augmenting-path search decides which and restores maximality.

Update dispatch (exactly ONE search per update that can change ν):

* insert, both endpoints free: match the new edge directly (Lemma A: this is optimal,
  |M|+1 ≤ ν(G+e) ≤ |M|+1). No search.
* insert, exactly one endpoint free (say u): any augmenting path w.r.t. M in G+e must
  use the new edge (a path avoiding it lives in G and would contradict maximality of
  M), and a free vertex on an alternating path is necessarily an endpoint — so every
  augmenting path starts at u. One rooted search from u is complete.
* insert, both endpoints matched: one search rooted at ALL free vertices
  (multi-root alternating forest). Complete because every augmenting path has free
  endpoints, and the multi-root forest finds an augmenting path iff one exists
  (classical Edmonds search; see kernel notes below).
* delete of an unmatched edge: zero work — M is untouched and still maximum (Lemma A).
* delete of a matched edge (u,v): free both endpoints; every augmenting path w.r.t.
  the shrunk matching must end at u or v (free vertices are path endpoints, and a
  path avoiding both would have been augmenting before the deletion). One two-root
  search from {u, v} is complete.

Search kernel: multi-root Edmonds blossom search (alternating BFS forest over all
given free roots, blossom contraction with base[] labels, p[] repointing inside
contracted blossoms, augmenting flip). It is a Python port of the classical e-maxx /
cp-algorithms single-root implementation (cf. Rosetta Code "Blossom algorithm" C++
listing; original: e-maxx.ru "Matching in general graphs"), extended to multiple
roots in the standard way:

  * every forest vertex carries a tree id (its root);
  * a same-tree re-encounter of a forest vertex triggers the classical blossom
    contraction (odd cycle found via LCA in the alternating tree);
  * an edge joining two EVEN vertices of DIFFERENT trees closes an augmenting path:
    the tree path root_A … v is alternating ending in v's matched edge, the cross
    edge (v,u) is unmatched, and u … root_B alternates again to a free root — the
    concatenation is augmenting and is flipped by an explicit pair-flip;
  * an edge joining an even vertex to an ODD vertex of a different tree is ignored:
    it cannot lie on an augmenting path (it would force two consecutive unmatched
    edges at the odd endpoint) and it cannot close an odd cycle (the endpoints share
    no tree).  This mirrors the single-root rule that even→odd same-tree edges are
    inert.

Provenance: the single-root kernel structure (lca / mark_path / flip walk) is a
faithful port of the battle-tested e-maxx code, not original work; the multi-root
extension and the dynamic dispatch around it are.

Guarantees:
  * `matching()` always returns a valid matching of the current graph.
  * eager mode (the only mode here): the matching is maximum after every update.
  * per update: O(1) bookkeeping + at most ONE blossom search, O(m) worst case
    (O(n·m) pathological blossom cascades, as always for this algorithm family).

Semantics:
  * insert of an already-present edge: no-op (zero-work report).
  * delete of an absent edge: no-op.
  * insert(u, u): raises ValueError (self-loops rejected).
  * parallel edges are deduplicated: one logical edge per vertex pair; at most that
    one copy can be matched (cardinality is unaffected by multiplicity).
  * out-of-range vertices raise IndexError.

v0.1 scope: fixed vertex set 0..n-1 given at construction; no weighted mode, no
lazy mode, no vertex add/remove (see spec §2 for the fuller v0.2 API).
"""

from __future__ import annotations

from collections import deque


class RepairReport:
    """Summary of the work done by a single update (insert or delete)."""

    __slots__ = ("op", "edge", "searched", "augmented", "nu_dropped", "work", "size")

    def __init__(self, op: str, edge, searched: bool, augmented: bool,
                 nu_dropped: bool, work: int, size: int):
        self.op = op              # "insert" | "delete" | "noop"
        self.edge = edge          # (u, v) with u < v
        self.searched = searched  # any blossom search was run
        self.augmented = augmented      # |M| increased by this update
        self.nu_dropped = nu_dropped    # delete: freed edge could not be re-augmented
        self.work = work          # edges scanned by searches (0 for no-op updates)
        self.size = size          # |M| after the update

    def __repr__(self):  # pragma: no cover - debug aid
        return (f"RepairReport({self.op!r}, {self.edge}, searched={self.searched}, "
                f"augmented={self.augmented}, nu_dropped={self.nu_dropped}, "
                f"work={self.work}, size={self.size})")


class DynamicMatching:
    """Exact dynamic maximum-cardinality matching on n fixed vertices.

    Example:
        dm = DynamicMatching(4)
        dm.insert(0, 1); dm.insert(1, 2); dm.insert(2, 3)
        dm.size()      # 2
        dm.matching()  # [(0, 1), (2, 3)]
    """

    def __init__(self, n: int):
        if n < 0:
            raise ValueError("n must be >= 0")
        self.n = n
        # adjacency: per-vertex dict neighbor -> True (deduped, O(1) insert/delete)
        self.adj = [dict() for _ in range(n)]
        self.mate = [-1] * n     # mate[v] = partner vertex or -1
        self._m = 0              # number of logical edges
        self._size = 0           # |M|
        self.stats = {"updates": 0, "searches": 0, "augmentations": 0, "work": 0}

    # ------------------------------------------------------------------ helpers

    def _check(self, u: int, v: int):
        if u == v:
            raise ValueError(f"self-loop rejected: ({u}, {u})")
        if not (0 <= u < self.n) or not (0 <= v < self.n):
            raise IndexError(f"vertex out of range: ({u}, {v}) n={self.n}")

    # ------------------------------------------------------------------ updates

    def insert(self, u: int, v: int) -> RepairReport:
        """Insert edge (u, v); no-op if already present."""
        self._check(u, v)
        if v in self.adj[u]:
            return RepairReport("noop", (min(u, v), max(u, v)),
                                False, False, False, 0, self._size)
        self.adj[u][v] = True
        self.adj[v][u] = True
        self._m += 1
        self.stats["updates"] += 1
        size0 = self._size
        mu, mv = self.mate[u], self.mate[v]
        if mu == -1 and mv == -1:
            # both free: matching the new edge is always optimal (Lemma A). No search.
            self.mate[u] = v
            self.mate[v] = u
            self._size += 1
            return RepairReport("insert", (min(u, v), max(u, v)),
                                False, True, False, 0, self._size)
        if mu == -1 or mv == -1:
            # one free endpoint: every augmenting path must start there (docstring)
            root = u if mu == -1 else v
            work = self._search([root])
            return RepairReport("insert", (min(u, v), max(u, v)), True,
                                self._size > size0, False, work, self._size)
        # both matched: one multi-root search over all free vertices
        roots = [r for r in range(self.n) if self.mate[r] == -1]
        work = self._search(roots)
        return RepairReport("insert", (min(u, v), max(u, v)), True,
                            self._size > size0, False, work, self._size)

    def delete(self, u: int, v: int) -> RepairReport:
        """Delete edge (u, v); no-op if absent."""
        self._check(u, v)
        if v not in self.adj[u]:
            return RepairReport("noop", (min(u, v), max(u, v)),
                                False, False, False, 0, self._size)
        del self.adj[u][v]
        del self.adj[v][u]
        self._m -= 1
        self.stats["updates"] += 1
        size0 = self._size
        if self.mate[u] != v:
            # unmatched edge: zero work, still maximum (Lemma A)
            return RepairReport("delete", (min(u, v), max(u, v)),
                                False, False, False, 0, self._size)
        # matched edge: free both endpoints; every augmenting path must end at u or v
        self.mate[u] = self.mate[v] = -1
        self._size -= 1
        work = self._search([u, v])
        nu_dropped = self._size == size0 - 1
        return RepairReport("delete", (min(u, v), max(u, v)), True,
                            self._size > size0 - 1, nu_dropped, work, self._size)

    # ------------------------------------------------------------ batch helpers

    def insert_many(self, edges) -> list:
        """Apply a sequence of inserts; returns the RepairReports."""
        return [self.insert(u, v) for u, v in edges]

    def delete_many(self, edges) -> list:
        """Apply a sequence of deletes; returns the RepairReports."""
        return [self.delete(u, v) for u, v in edges]

    # ------------------------------------------------------------------ queries

    def matching(self):
        """Current maximum matching as a sorted list of (u, v) pairs, u < v."""
        out = []
        for v in range(self.n):
            w = self.mate[v]
            if w != -1 and v < w:
                out.append((v, w))
        out.sort()
        return out

    def size(self) -> int:
        """Number of matched edges (|M|)."""
        return self._size

    def num_edges(self) -> int:
        """Number of logical (deduplicated) edges currently present."""
        return self._m

    def is_valid(self) -> bool:
        """Invariant check: mate symmetric, every matched pair is a real edge."""
        for v in range(self.n):
            w = self.mate[v]
            if w == -1:
                continue
            if not (0 <= w < self.n) or self.mate[w] != v or w not in self.adj[v]:
                return False
        return True

    def free_vertices(self):
        return {v for v in range(self.n) if self.mate[v] == -1}

    def oracle_check(self) -> bool:
        """TEST-ONLY: compare |M| against a full networkx recompute."""
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(self.n))
        for u in range(self.n):
            for v in self.adj[u]:
                if u < v:
                    g.add_edge(u, v)
        return len(nx.max_weight_matching(g, maxcardinality=True)) == self._size

    # ------------------------------------------------------------------ kernel

    def _search(self, roots) -> int:
        """Multi-root Edmonds blossom search from free vertices `roots`.

        On success, augments self.mate along the found path (self._size += 1).
        Returns the number of edges scanned (work).

        Single-root structure is a faithful port of the classical e-maxx
        implementation; multi-root extension per the module docstring.
        """
        n = self.n
        mate = self.mate
        adj = self.adj
        p = [-1] * n
        base = list(range(n))
        used = [False] * n        # "even" (outer) forest membership
        tree = [-1] * n           # root id of the tree a vertex belongs to
        blossom = [False] * n
        q = deque()
        for r in roots:
            used[r] = True
            tree[r] = r
            q.append(r)
        work = 0
        self.stats["searches"] += 1

        def lca(a, b):
            # a and b are guaranteed to be in the same tree here
            used_path = [False] * n
            while True:
                a = base[a]
                used_path[a] = True
                if mate[a] < 0:
                    break
                a = p[mate[a]]
            while True:
                b = base[b]
                if used_path[b]:
                    return b
                b = p[mate[b]]

        def mark_path(v, b, x):
            while base[v] != b:
                blossom[base[v]] = True
                blossom[base[mate[v]]] = True
                p[v] = x
                x = mate[v]
                v = p[x]

        def walk_up(x):
            """Vertex path from even forest vertex x up to its free root.

            Uses the e-maxx invariants: even non-root -> mate (matched edge up),
            odd -> p (unmatched edge up).  Blossom repointing of p keeps this
            walk alternating inside contracted blossoms (the same walk the
            single-root flip uses).  Positional parity, not used[].
            """
            path = [x]
            cur, even = x, True
            while True:
                if even:
                    if mate[cur] == -1:
                        return path          # cur is the free root
                    cur = mate[cur]
                else:
                    cur = p[cur]
                path.append(cur)
                even = not even

        def flip_walk(u):
            """Classical e-maxx augmenting flip from free odd vertex u (p[u] set)."""
            cur = u
            while cur != -1:
                prev = p[cur]
                nxt = mate[prev] if prev >= 0 else -1
                mate[cur] = prev
                if prev >= 0:
                    mate[prev] = cur
                cur = nxt
            self._size += 1
            self.stats["augmentations"] += 1

        def cross_flip(v, u):
            """Even-even edge across trees v–u: augment root_v…v–u…root_u."""
            pv = walk_up(v)      # v … rootA
            pu = walk_up(u)      # u … rootB
            full = pv[::-1] + pu  # rootA … v, u … rootB  (odd number of edges)
            for i in range(0, len(full) - 1, 2):
                a, b = full[i], full[i + 1]
                mate[a], mate[b] = b, a
            self._size += 1
            self.stats["augmentations"] += 1

        while q:
            v = q.popleft()
            tv = tree[v]
            for u in adj[v]:
                work += 1
                if base[v] == base[u] or mate[v] == u:
                    continue
                if not used[u] and p[u] < 0:
                    # not in forest: discover as odd child of v
                    p[u] = v
                    tree[u] = tv
                    if mate[u] < 0:
                        flip_walk(u)             # single-tree augmenting path
                        self.stats["work"] += work
                        return work
                    used[mate[u]] = True
                    tree[mate[u]] = tv
                    q.append(mate[u])
                elif used[u]:
                    # u is even in some tree
                    if tree[u] != tv:
                        cross_flip(v, u)         # augmenting path across trees
                        self.stats["work"] += work
                        return work
                    if mate[u] == -1 or p[mate[u]] >= 0:
                        # same tree, even u: blossom (u==root case included)
                        curbase = lca(v, u)
                        for i in range(n):
                            blossom[i] = False
                        mark_path(v, curbase, u)
                        mark_path(u, curbase, v)
                        for i in range(n):
                            if blossom[base[i]]:
                                base[i] = curbase
                                if not used[i]:
                                    used[i] = True
                                    q.append(i)
                else:
                    # u is odd in some tree (p[u] >= 0)
                    if tree[u] == tv and mate[u] >= 0 and p[mate[u]] >= 0:
                        # same tree, repointed mate: blossom (e-maxx rule)
                        curbase = lca(v, u)
                        for i in range(n):
                            blossom[i] = False
                        mark_path(v, curbase, u)
                        mark_path(u, curbase, v)
                        for i in range(n):
                            if blossom[base[i]]:
                                base[i] = curbase
                                if not used[i]:
                                    used[i] = True
                                    q.append(i)
                    # otherwise inert: even-odd edge (same or different tree)
        self.stats["work"] += work
        return work
