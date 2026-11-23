"""Narrow exemptions for frozen files (round K, 2026-11-23).

A pre-registration freezes whole files by SHA-256. When a frozen file later gains code a
scoring does not use (a new problem added to a list, new functions appended), the
scoring may continue only if every function it DOES use is unchanged. This compares the
source of named functions with their source at the pre-registration commit (git), so the
exemption is checked, not asserted.
"""
import ast
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "Practical_Examples/omar_pfem/verification_agent"


def _functions(src):
    tree = ast.parse(src)
    return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


def functions_unchanged(filename, commit, names):
    """True iff every function in `names` has the same AST now and at `commit`."""
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True,
                          text=True, check=True).stdout.strip()
    old = subprocess.run(["git", "show", f"{commit}:{REL}/{filename}"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    new = open(os.path.join(HERE, filename)).read()
    fo, fn = _functions(old), _functions(new)
    return all(n in fo and n in fn and fo[n] == fn[n] for n in names)


def _strip_switch(tree, env_name):
    """Remove every `if os.environ.get(env_name) == ...:` block (no else) from the AST."""
    class T(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            if env_name in ast.unparse(node.test) and not node.orelse:
                return None
            return node
    return T().visit(tree)


def unchanged_except_switch(filename, commit, env_name):
    """True iff the whole module equals its version at `commit` once every block guarded by
    the post hoc switch `env_name` is removed (round M: VERIF_POSTHOC_DETF in run_check).
    With the switch off such a block never runs, so the scoring code path is unchanged."""
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True,
                          text=True, check=True).stdout.strip()
    old = subprocess.run(["git", "show", f"{commit}:{REL}/{filename}"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    new = open(os.path.join(HERE, filename)).read()
    a = ast.dump(_strip_switch(ast.parse(old), env_name))
    b = ast.dump(_strip_switch(ast.parse(new), env_name))
    return a == b and os.environ.get(env_name) != "1"
