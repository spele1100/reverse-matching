"""Tests for the maxmatch CLI (cli.py)."""

import json
import os
import subprocess
import sys

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cli  # noqa: E402


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, os.path.join(HERE, "cli.py"), *argv],
        capture_output=True, text=True, timeout=300,
    )


def test_parse_file_path_graph():
    r = run_cli(os.path.join(HERE, "examples", "path_graph.txt"))
    assert r.returncode == 0, r.stderr
    assert "matching size k: 3" in r.stdout
    assert "validated:       yes" in r.stdout


def test_parse_file_with_header_and_isolates():
    r = run_cli(os.path.join(HERE, "examples", "isolates.txt"))
    assert r.returncode == 0, r.stderr
    assert "nodes:           8" in r.stdout
    assert "matching size k: 3" in r.stdout


def test_random_json_output():
    r = run_cli("--random", "10", "0.3", "--seed", "42", "--json")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["nodes"] == 10
    assert out["mode"] == "edmonds"
    assert out["valid"] is True
    assert out["matching_size"] == len(out["matching"])
    verts = [v for e in out["matching"] for v in e]
    assert len(verts) == len(set(verts))


def test_random_cross_check_vs_networkx():
    n, p = 12, 0.35
    n, edges = cli.random_graph(n, p, seed=7)
    assert edges, "empty random graph"
    r = run_cli("--random", str(n), str(p), "--seed", "7", "--json")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    k_nx = len(nx.max_weight_matching(g, maxcardinality=True))
    assert out["matching_size"] == k_nx, (out["matching_size"], k_nx)
    ok, why = cli.validate_matching(n, edges, [tuple(e) for e in out["matching"]])
    assert ok, why


def test_self_loop_rejected():
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("0 1\n1 1\n")
        path = f.name
    r = run_cli(path)
    os.unlink(path)
    assert r.returncode != 0
    assert "self-loop" in r.stderr


def test_help():
    r = run_cli("--help")
    assert r.returncode == 0
    assert "--random" in r.stdout and "--mode" in r.stdout
