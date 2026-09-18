"""Builds ONE notebook: an independent, from-scratch re-verification of the
B2xNeo-Hookean classical-QoI staleness finding (2026-09-18). Omar asked for
this specifically because cross-referencing existing Drive files (the
2026-09-17 post-retrain sanity sweep, which already agreed with today's
Remaining5_LowN_Accuracy.ipynb run) was not convincing enough on its own --
he wants a fresh computation he can watch happen. See
cell_verify_b2_neo_hookean_independent.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_verify_b2_neo_hookean_independent.py"


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
            "colab": {"provenance": [], "name": "Verify_B2_NeoHookean_Independent.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# تحقق مستقل من جديد: خطأ B2×Neo-Hookean عند N=13",
                "",
                "عمر مش قانع بالتحقق عن طريق مقارنة ملفات موجودة مسبقًا على Drive "
                "(حتى لو كانت مستقلة ومتوافقة) -- بده حساب من الصفر يشوفه يصير "
                "بعينه.",
                "",
                "**هاي النوتة بتكتب لملف جديد كليًا** "
                "(`no_accuracy_degradation_sweep_B2_neo_hookean_VERIFY.json`) ما "
                "كان موجود قبل هيك، فآلية الـresume ما رح تلاقي شي تتخطاه -- "
                "الـ16 resolution كلهم رح يتحسبوا من جديد فعليًا هالمرة.",
                "",
                "بتحمّل نفس الـcheckpoint النهائي بالضبط "
                "(`zeroshot_B2_neo_hookean_multires/model_best.pt`) الي كل خلية "
                "تانية بجولة 12 بتستخدمه، وبتطبع الـfingerprint (sha256) قبل ما "
                "تحسب أي شي، وبالآخر بتقارن رقم N=13 الجديد مباشرة مع الرقمين "
                "المتنازع عليهم (12.71% القديم مقابل 53.33% الجديد).",
                "",
                "**الكلفة المتوقعة**: دقيقة إلى دقيقتين فعليًا (حالة واحدة بس، "
                "مو خمسة متل نوتة Remaining5).",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Verify_B2_NeoHookean_Independent.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
