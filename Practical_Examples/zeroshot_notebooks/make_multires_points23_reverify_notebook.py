"""Builds ONE notebook, two cells: re-verifies points 2 (max feasible
batch size/throughput) and 3 (profiling/torch.compile/TF32) against the
NEW multi-res checkpoint, since the draft's point 1 now recommends that
checkpoint but points 2/3 were only ever verified against the old
(correct-but-narrower) N=21,33 checkpoint. Timing should be checkpoint-
independent (same architecture/parameter count) -- this project verifies
that instead of assuming it. See the two cell files' own docstrings.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code_from_file(path):
    with open(path) as f:
        src = f.read()
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in src.splitlines()]}


def build():
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "Round6_MultiRes_Points23_Reverify.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# Points 2 and 3, re-verified against the multi-res checkpoint",
                "",
                "The draft's point 1 now recommends the multi-res-retrained checkpoint "
                "(trained on N=21,33,101,201), but points 2 (max feasible batch size/"
                "throughput) and 3 (profiling/torch.compile/TF32) were only ever "
                "verified against the OLD checkpoint (N=21,33 only, correct but "
                "narrower). Timing should be checkpoint-independent -- same "
                "architecture, same parameter count, only the weights differ -- but "
                "this project verifies that claim instead of assuming it, exactly as "
                "was already done once for the wrong-vs-correct-checkpoint question.",
                "",
                "Two cells below, each prints an OLD-vs-NEW comparison at the end. "
                "Run cell 1 (point 2, ~10 min) then cell 2 (point 3, ~15-20 min).",
            ),
            code_from_file(f"{HERE}/cell_max_feasible_batch_size_multires.py"),
            code_from_file(f"{HERE}/cell_no_inference_torch_compile_multires.py"),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round6_MultiRes_Points23_Reverify.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
