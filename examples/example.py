"""Demo of the reverse-matching Python API.

Run:  python examples/example.py
"""

from assembly import max_cardinality_matching

# Path graph 0-1-2-3-4-5
n = 6
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

# Fast mode (Edmonds fallback branch of the paper's assembly).
k, matching = max_cardinality_matching(n, edges, mode="edmonds")
print(f"path graph: k={k}, matching={matching}")
assert k == 3

# A triangle plus an isolated vertex.
n = 4
edges = [(0, 1), (1, 2), (2, 0)]
k, matching = max_cardinality_matching(n, edges, mode="edmonds")
print(f"triangle + isolate: k={k}, matching={matching}")
assert k == 1

# Research modes ("pm", "dovetail") are exact but *very* slow — the README
# quotes ~20 min for an n=3 pipeline in pm mode. Stick to "edmonds" unless
# you are deliberately exercising the paper's machinery on tiny graphs.

print("all good")
