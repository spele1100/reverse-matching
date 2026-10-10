"""
Layer 1 reductions for the OpenAI matching paper (Chapter 9, Lemmas 9.1-9.3).

Implements, following the paper text in /root/papers/ch9.txt:
  * Input normalization (edge records emitted at occurrence (u,v) with u<v).
  * Lemma 9.1: linearithmic permutation network (recursive Benes-style
    switching construction) + routing of an arbitrary permutation by
    vertex-disjoint paths.
  * Lemma 9.2: exact-cardinality -> perfect-matching reduction. Given G and
    0 <= k <= floor(n/2), builds H_k with b = n-2k mandatory sink vertices,
    s = least power of two >= n network positions (each split into v_in/v_out
    joined by an idle edge), arcs u->v becoming u_out-v_in, original vertex i
    joined to the input endpoint of network input i. H_k has a perfect
    matching iff G has a matching of size k. decode() recovers a size-k
    matching of G, and construct_witness_hk() builds an explicit perfect
    matching of H_k from a size-k matching of G.
  * Lemma 9.3: Dahlhaus-Karpinski degree reduction to max degree 3
    (degree-r vertex -> path on 2r-1 vertices, incidences at odd positions),
    with witness conversion in both directions.

Vertices are plain ints; edges carry a type tag and an identifier so that
witnesses can be decoded exactly as the paper prescribes ("Store each edge's
type and original identifier. Decoding simply selects the original edges.").
"""

from collections import defaultdict, namedtuple
from itertools import count

Edge = namedtuple("Edge", ["u", "v", "etype", "eid"])
# etype in:
#   'original' : an edge of the input graph G (eid = normalized edge id)
#   'idle'     : network split-position idle edge v_in - v_out
#   'arc'      : network transition edge (arc u->v became u_out - v_in)
#   'attach'   : original vertex i - input endpoint of network input i
#   'sink'     : selected output endpoint - mandatory sink vertex
#   'gpath'    : degree gadget internal path edge
#   'gext'     : degree gadget external edge (eid = original edge id)


# ---------------------------------------------------------------------------
# Input normalization (Section 9.1 preamble)
# ---------------------------------------------------------------------------

def normalize(n, pairs=None, adjacency=None):
    """Read raw edge occurrences once; emit an edge record only at the
    occurrence (u, v) with u < v.  Returns a list of (u, v) with u < v,
    deduplicated, with no assumption on list order.

    Accepts either an iterable of endpoint pairs or an adjacency list
    (list of neighbor lists, one per vertex).
    """
    if adjacency is not None:
        pairs = ((u, w) for u in range(n) for w in adjacency[u])
    seen = set()
    out = []
    for u, v in pairs:
        if u == v:
            raise ValueError(f"loop at vertex {u}: input must be loopless")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"endpoint out of range: {(u, v)} for n={n}")
        if u > v:
            u, v = v, u
        if (u, v) not in seen:
            seen.add((u, v))
            out.append((u, v))
    return out


# ---------------------------------------------------------------------------
# Lemma 9.1: linearithmic permutation network
# ---------------------------------------------------------------------------

class PermNetwork:
    """Recursive switching network (Benes construction, Lemma 9.1).

    s positions per side (s a power of two, s >= 1).  For s = 1 a single
    position is both an input and an output and there are no arcs.  For s > 1:
    s new input positions, s new output positions, disjoint upper/lower
    sub-networks on s/2 positions; each input in pair {2i, 2i+1} connects to
    input i of both copies, and output j of either copy connects to both
    positions of output pair {2j, 2j+1}.  Every permutation of inputs to
    outputs is routable by vertex-disjoint directed paths (2-color the even
    cycles of the pair multigraph, route inductively).
    """

    def __init__(self, s, ids=None):
        if s < 1 or (s & (s - 1)) != 0:
            raise ValueError("s must be a power of two >= 1")
        if ids is None:
            ids = count()
        self.s = s
        if s == 1:
            p = next(ids)
            self.inputs = [p]
            self.outputs = [p]  # position is both input and output when s=1
            self.arcs = []
            self.up = None
            self.low = None
            return
        self.inputs = [next(ids) for _ in range(s)]
        self.outputs = [next(ids) for _ in range(s)]
        self.up = PermNetwork(s // 2, ids)
        self.low = PermNetwork(s // 2, ids)
        arcs = []
        for i in range(s // 2):
            arcs.append((self.inputs[2 * i], self.up.inputs[i]))
            arcs.append((self.inputs[2 * i + 1], self.up.inputs[i]))
            arcs.append((self.inputs[2 * i], self.low.inputs[i]))
            arcs.append((self.inputs[2 * i + 1], self.low.inputs[i]))
        for j in range(s // 2):
            arcs.append((self.up.outputs[j], self.outputs[2 * j]))
            arcs.append((self.up.outputs[j], self.outputs[2 * j + 1]))
            arcs.append((self.low.outputs[j], self.outputs[2 * j]))
            arcs.append((self.low.outputs[j], self.outputs[2 * j + 1]))
        self.arcs = arcs

    def positions(self):
        """All distinct position ids used anywhere in the network."""
        if self.s == 1:
            return list(self.inputs)
        out = list(self.inputs) + list(self.outputs)
        out.extend(self.up.positions())
        out.extend(self.low.positions())
        return out

    def arc_set(self):
        if self.s == 1:
            return set()
        s = set(self.arcs)
        s |= self.up.arc_set()
        s |= self.low.arc_set()
        return s

    # -- routing ------------------------------------------------------------

    def _split_by_color(self, perm):
        """Color the pair multigraph alternately up/low along its even
        cycles.  Returns (color, up_map, low_map): color[i] tells which copy
        routes input i; up_map/low_map are the induced sub-permutations.
        """
        s = self.s
        edges = [(i // 2, perm[i] // 2) for i in range(s)]
        adj = defaultdict(list)
        for idx, (a, b) in enumerate(edges):
            adj[("in", a)].append(idx)
            adj[("out", b)].append(idx)
        color = [None] * s
        unused = set(range(s))
        while unused:
            e0 = unused.pop()
            node = ("in", edges[e0][0])
            cur = e0
            c = "up"
            while True:
                color[cur] = c
                a, b = edges[cur]
                nxt_node = ("out", b) if node == ("in", a) else ("in", a)
                cand = [x for x in adj[nxt_node] if x != cur]
                if len(cand) != 1:
                    raise AssertionError("pair multigraph not 2-regular")
                nxt = cand[0]
                if nxt == e0:
                    break
                unused.discard(nxt)
                cur, node = nxt, nxt_node
                c = "low" if c == "up" else "up"
        up_map, low_map = {}, {}
        for i in range(s):
            if color[i] == "up":
                up_map[i // 2] = perm[i] // 2
            else:
                low_map[i // 2] = perm[i] // 2
        return color, up_map, low_map

    def route(self, perm):
        """Route permutation perm (dict input index -> output index, a
        bijection on range(s)) by vertex-disjoint directed paths.

        Returns dict i -> list of position ids tracing the path from
        self.inputs[i] to self.outputs[perm[i]]; consecutive entries are
        joined by genuine arcs of the network.
        """
        s = self.s
        if sorted(perm.keys()) != list(range(s)):
            raise ValueError("perm must be a bijection on range(s)")
        if sorted(perm.values()) != list(range(s)):
            raise ValueError("perm must be a bijection on range(s)")
        if s == 1:
            return {0: [self.inputs[0]]}
        color, up_map, low_map = self._split_by_color(perm)
        up_paths = self.up.route(up_map)
        low_paths = self.low.route(low_map)
        paths = {}
        for i in range(s):
            if color[i] == "up":
                subpath = up_paths[i // 2]
            else:
                subpath = low_paths[i // 2]
            paths[i] = [self.inputs[i]] + subpath + [self.outputs[perm[i]]]
        return paths


# ---------------------------------------------------------------------------
# Lemma 9.2: exact cardinality -> perfect matching
# ---------------------------------------------------------------------------

class HK:
    """Constructed instance H_k of Lemma 9.2."""

    def __init__(self, n, k, edges, net, pos_in, pos_out, selected, sinks,
                 attach, edge_list):
        self.n = n
        self.k = k
        self.b = n - 2 * k
        self.s = net.s
        self.net = net
        self.pos_in = pos_in        # position id -> v_in vertex id
        self.pos_out = pos_out      # position id -> v_out vertex id
        self.selected_outputs = selected  # b selected output positions
        self.sinks = sinks          # b mandatory sink vertex ids
        self.attach = attach        # original vertex -> attached input position
        self.edges = edge_list      # list[Edge]
        self._by_pair = {frozenset((e.u, e.v)): e for e in edge_list}

    @property
    def vertices(self):
        vs = list(range(self.n))
        for pos in self.net.positions():
            vs.append(self.pos_in[pos])
            vs.append(self.pos_out[pos])
        vs.extend(self.sinks)
        return vs

    def to_networkx(self):
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(self.vertices)
        g.add_edges_from((e.u, e.v) for e in self.edges)
        return g

    def decode(self, matching):
        """Select the original edges from a returned perfect matching.
        `matching` is any iterable of vertex pairs.  Returns list of (u, v).
        """
        out = []
        for pair in matching:
            e = self._by_pair[frozenset(pair)]
            if e is None:
                raise KeyError(f"edge {pair} not present in H_k")
            if e.etype == "original":
                out.append((e.u, e.v))
        return out


def build_hk(n, edges, k):
    """Construct H_k (Lemma 9.2).

    n    : number of vertices of G (labeled 0..n-1)
    edges: iterable of (u, v) pairs of G (any orientation; normalized here)
    k    : target cardinality, 0 <= k <= n//2
    """
    if not (0 <= k <= n // 2):
        raise ValueError(f"k={k} out of range for n={n}")
    edges = normalize(n, pairs=edges)
    b = n - 2 * k
    s = 1
    while s < n:
        s *= 2
    ids = count(start=n)  # network position ids
    net = PermNetwork(s, ids)
    positions = net.positions()
    # Split every position into v_in / v_out joined by an idle edge.
    vid = count(start=next(ids))
    pos_in, pos_out = {}, {}
    out_edges = []
    for pos in positions:
        vi, vo = next(vid), next(vid)
        pos_in[pos], pos_out[pos] = vi, vo
        out_edges.append(Edge(vi, vo, "idle", ("idle", pos)))
    # Every arc u->v of the network becomes the edge u_out v_in.
    for (a, c) in sorted(net.arc_set()):
        out_edges.append(Edge(pos_out[a], pos_in[c], "arc", ("arc", a, c)))
    # Retain all original vertices and edges.
    for eid, (u, v) in enumerate(edges):
        out_edges.append(Edge(u, v, "original", eid))
    # Join original vertex i to the input endpoint of network input i.
    attach = {}
    for i in range(n):
        pos = net.inputs[i]
        attach[i] = pos
        out_edges.append(Edge(i, pos_in[pos], "attach", ("attach", i)))
    # One mandatory sink vertex adjacent to the output endpoint of each of
    # b distinct selected outputs (we select outputs 0..b-1).
    sink_base = next(vid)
    sinks = list(range(sink_base, sink_base + b))
    selected = net.outputs[:b]
    for t, opos in enumerate(selected):
        out_edges.append(Edge(pos_out[opos], sinks[t], "sink", ("sink", t)))
    return HK(n, k, edges, net, pos_in, pos_out, selected, sinks, attach,
              out_edges)


def construct_witness_hk(hk, g_matching, uncovered):
    """Converse direction of Lemma 9.2: given a size-k matching of G and the
    set of b = n-2k vertices it leaves uncovered, build an explicit perfect
    matching of H_k by routing a permutation through the network.

    Returns list[Edge].
    """
    b = hk.b
    uncovered = list(uncovered)
    if len(uncovered) != b:
        raise ValueError("need exactly b = n-2k uncovered vertices")
    if len(g_matching) != hk.k:
        raise ValueError("matching must have exactly k edges")
    in_index = {pos: i for i, pos in enumerate(hk.net.inputs)}
    out_index = {pos: j for j, pos in enumerate(hk.net.outputs)}
    # Map the uncovered vertices bijectively to the selected outputs and
    # extend to a permutation of all s positions (identity on the rest).
    perm = {}
    for u, opos in zip(sorted(uncovered), hk.selected_outputs):
        perm[in_index[hk.attach[u]]] = out_index[opos]
    free_in = [i for i in range(hk.s) if i not in perm]
    free_out = [j for j in range(hk.s) if j not in set(perm.values())]
    for i, j in zip(free_in, free_out):
        perm[i] = j
    paths = hk.net.route(perm)
    by_pair = hk._by_pair
    chosen = {}

    def add(u, v):
        key = frozenset((u, v))
        if key in chosen:
            raise AssertionError(f"conflict: edge {key} chosen twice")
        chosen[key] = by_pair[key]

    # Retained paths: attachment edge, all transition edges, sink edge.
    for u in uncovered:
        p = paths[in_index[hk.attach[u]]]
        add(u, hk.pos_in[p[0]])                       # original-to-input edge
        for a, c in zip(p, p[1:]):                    # transition edges
            add(hk.pos_out[a], hk.pos_in[c])
        sink = hk.sinks[hk.selected_outputs.index(p[-1])]
        add(hk.pos_out[p[-1]], sink)                  # edge to mandatory sink
    # Idle edges on every other position.
    active = set()
    for u in uncovered:
        active.update(paths[in_index[hk.attach[u]]])
    for pos in hk.net.positions():
        if pos not in active:
            add(hk.pos_in[pos], hk.pos_out[pos])
    # The original matching on its covered vertices.
    for (u, v) in g_matching:
        add(u, v)
    return list(chosen.values())


# ---------------------------------------------------------------------------
# Lemma 9.3: degree reduction to max degree 3 (Dahlhaus-Karpinski)
# ---------------------------------------------------------------------------

class DegreeReduced:
    """Result of degree_reduce().  If `.isolate` is not None, the original
    graph has an isolated vertex certifying absence of a perfect matching
    and no gadget exists for it (check isolates first, as the paper says)."""

    def __init__(self, vertices, edges, new_vertices, new_edges, gadgets,
                 isolate, att_of_edge):
        self.orig_vertices = list(vertices)
        self.orig_edges = list(edges)     # list of (u, v)
        self.new_vertices = new_vertices
        self.new_edges = new_edges        # list[Edge]
        self.gadgets = gadgets            # v -> list of attachment vertices
        self.isolate = isolate
        self._att_of_edge = att_of_edge   # (v, eid) -> attachment vertex
        self._by_pair = {frozenset((e.u, e.v)): e for e in new_edges}

    def to_networkx(self):
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(self.new_vertices)
        g.add_edges_from((e.u, e.v) for e in self.new_edges)
        return g

    def project(self, matching):
        """Gadget perfect matching -> original perfect matching: select the
        external edges (they carry the original identifiers at both ends).
        Returns list of (u, v) of the original graph."""
        out = []
        for pair in matching:
            e = self._by_pair[frozenset(pair)]
            if e.etype == "gext":
                out.append(self.orig_edges[e.eid])
        return out

    def lift(self, orig_matching):
        """Original perfect matching -> gadget perfect matching.  The odd
        attachment of each selected edge leaves two even-length paths on its
        sides; match those internally.  Returns list[Edge]."""
        eid_of = {frozenset(e): i for i, e in enumerate(self.orig_edges)}
        chosen = []
        selected_att = {}  # gadget v -> attachment vertex of its chosen edge
        for m in orig_matching:
            eid = eid_of[frozenset(m)]
            u, v = self.orig_edges[eid]
            au = self._att_of_edge[(u, eid)]
            av = self._att_of_edge[(v, eid)]
            if u in selected_att or v in selected_att:
                raise AssertionError("not a matching")
            selected_att[u] = au
            selected_att[v] = av
            chosen.append(self._by_pair[frozenset((au, av))])
        for v in self.orig_vertices:
            if v not in self.gadgets:
                continue  # isolated in the original graph
            full = self._path_order(v)  # 2r-1 vertices in path order
            t = full.index(selected_att[v])  # even position of chosen incidence
            for side in (range(0, t), range(t + 1, len(full))):
                idx = list(side)
                for a, c in zip(idx[0::2], idx[1::2]):
                    chosen.append(self._by_pair[frozenset((full[a], full[c]))])
        return chosen

    def _path_order(self, v):
        """Attachment gadget path of v in order (2r-1 vertices).  Between two
        consecutive odd attachments path[2j], path[2j+2] sits the unique even
        vertex path[2j+1], found as their common gpath neighbor."""
        atts = self.gadgets[v]
        nb = _path_neighbors(self.new_edges)
        order = []
        for j in range(len(atts)):
            order.append(atts[j])
            if j + 1 < len(atts):
                order.append(next(iter(nb[atts[j]] & nb[atts[j + 1]])))
        return order


def _path_neighbors(new_edges):
    """map vertex -> set of its neighbors along gpath edges."""
    nb = defaultdict(set)
    for e in new_edges:
        if e.etype == "gpath":
            nb[e.u].add(e.v)
            nb[e.v].add(e.u)
    return nb


def degree_reduce(vertices, edges):
    """Lemma 9.3.  vertices: iterable of vertex ids; edges: list of (u, v).

    Replaces a vertex of degree r >= 1 by a path on 2r-1 vertices, attaching
    its r old incidences to successive odd positions.  Returns DegreeReduced.
    """
    edges = list(edges)
    incidences = defaultdict(list)  # v -> list of edge ids, in scan order
    for i, (u, v) in enumerate(edges):
        incidences[u].append(i)
        incidences[v].append(i)
    isolate = None
    new_vertices = []
    new_edges = []
    gadgets = {}
    att_of_edge = {}
    ids = count()
    for v in vertices:
        inc = incidences.get(v, [])
        r = len(inc)
        if r == 0:
            isolate = v
            continue
        path = [next(ids) for _ in range(2 * r - 1)]
        new_vertices.extend(path)
        for a, c in zip(path, path[1:]):
            new_edges.append(Edge(a, c, "gpath", ("gpath", v, a)))
        atts = [path[2 * j] for j in range(r)]  # incidences at odd positions
        gadgets[v] = atts
        for j, eid in enumerate(inc):
            att_of_edge[(v, eid)] = atts[j]
    for eid, (u, v) in enumerate(edges):
        # edges incident to an isolated vertex cannot exist (isolates have
        # no incidences), so both attachments are always defined.
        au = att_of_edge[(u, eid)]
        av = att_of_edge[(v, eid)]
        new_edges.append(Edge(au, av, "gext", eid))
    return DegreeReduced(vertices, edges, new_vertices, new_edges, gadgets,
                         isolate, att_of_edge)
