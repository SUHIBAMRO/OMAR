"""Sanity-checks every notebook in the repo before it is handed to anyone.

This exists because a notebook shipped with every line of every cell missing
its trailing newline. nbformat stores a cell's source as a LIST OF LINES, and
Jupyter simply concatenates that list -- so lines without "\n" arrive glued
together and the first cell dies with `SyntaxError: invalid syntax` on
something like `from google.colab import drivedrive.mount(...)`. The file is
still valid JSON and still opens fine, so nothing catches it until someone
tries to run it.

Run this after regenerating notebooks:
    python zeroshot_notebooks/check_notebooks.py
"""
import ast
import sys
import json
import glob
import os


def check(path):
    problems = []
    try:
        nb = json.load(open(path))
    except Exception as e:
        return [f"not valid JSON: {e}"]

    for i, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", [])
        if isinstance(src, list):
            # Every line but the last must end in a newline, or Jupyter's
            # concatenation silently welds it to the line below.
            missing = [j for j, line in enumerate(src[:-1])
                       if not line.endswith("\n")]
            if missing:
                problems.append(
                    f"cell {i} ({cell['cell_type']}): {len(missing)} of "
                    f"{len(src) - 1} lines missing their trailing newline "
                    f"-- this cell will be concatenated into one line")
            text = "".join(src)
        else:
            text = src

        # A code cell with no IPython magics is plain Python and must parse.
        # Cells using ! or % can't be parsed by ast, so they're skipped.
        if cell["cell_type"] == "code" and text.strip():
            if not any(l.lstrip().startswith(("!", "%")) for l in text.split("\n")):
                try:
                    ast.parse(text)
                except SyntaxError as e:
                    problems.append(f"cell {i}: SyntaxError: {e}")
    return problems


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(root, "**", "*.ipynb"), recursive=True))
    if not paths:
        print("no notebooks found under", root)
        return 0

    failed = 0
    for path in paths:
        problems = check(path)
        rel = os.path.relpath(path, root)
        if problems:
            failed += 1
            print(f"FAIL  {rel}")
            for p in problems:
                print(f"        {p}")
        else:
            print(f"ok    {rel}")

    print(f"\n{len(paths) - failed}/{len(paths)} notebooks OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
