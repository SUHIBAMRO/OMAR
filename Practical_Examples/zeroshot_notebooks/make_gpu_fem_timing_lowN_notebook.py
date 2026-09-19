"""Builds ONE notebook: GPU-native FEM solver timing across the FULL
LOW_N sweep, all six cases (Timon's newest email, item 1). See
cell_gpu_fem_timing_lowN_all_cases.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_gpu_fem_timing_lowN_all_cases.py"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def build():
    with open(CELL_FILE) as f:
        cell_src = f.read()
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "GPU_FEM_Timing_LowN_AllCases.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# توقيت GPU-FEM عبر كل الـLOW_N (طلب تيمون الجديد، نقطة 1)",
                "",
                "لحد هلق `gpu_fem_benchmark.py` انقاس مرة وحدة بس عند N=11 "
                "لحالة وحدة (لجدول break-even الأصلي). تيمون طلب جدول جديد: "
                "لكل QoI (L2, H1/Energy, Reaction, Cauchy بالمنطقة) ولكل "
                "مستوى دقة (1%/2%/5% مثلًا)، شو أصغر N لازم من FEM يحقق هاد "
                "المستوى، ووين نقطة التعادل (break-even) مع الـoperator. "
                "هاد يحتاج كلفة FEM الحقيقية عند **كل** N بالـLOW_N، مش بس "
                "N=11.",
                "",
                "هاي الخلية بتعيد استخدام دوال `gpu_fem_benchmark.py` "
                "المتحقق منها أصلًا (`build_batch_b1/b2` ونفس أسلوب "
                "warm-up/timing) مباشرة بدون subprocess لكل N -- هيك ما "
                "منضيع وقت بإعادة تشغيل CUDA context 96 مرة. تكرار واحد "
                "warm-up + 3 تكرارات مقاسة لكل (حالة, N)، batch_size=1 "
                "(نفس القياس المستخدم بكل حسابات break-even بهاد المشروع).",
                "",
                "**الكلفة المتوقعة**: 6 حالات × 16 N × 4 حلول = 384 حل، كلهم "
                "بشبكات صغيرة (N<=49) -- توقع بضع دقائق إجمالًا، مش ساعات GPU.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/GPU_FEM_Timing_LowN_AllCases.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
