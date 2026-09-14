"""Builds ONE notebook: extends the N=1401 accuracy/QoI/accuracy-matched
analysis (Timon round-11 point 2) to B1 x Mooney-Rivlin and B1 x
Arruda-Boyce, using their EXISTING zero-shot checkpoints (N=21,33 jointly
trained) -- no retraining. See cell_n1401_b1_other_materials.py for the
full rationale, the prerequisite material-parameter generalization fix,
and what is deliberately NOT included (resolution-matched break-even for
these two, and B2's three cases -- both flagged as separate follow-ups).
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_n1401_b1_other_materials.py"


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
            "colab": {"provenance": [], "name": "Round6_N1401_B1_OtherMaterials.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# N=1401 accuracy/QoI للحالتين الباقيتين من B1 (Mooney-Rivlin و Arruda-Boyce)",
                "",
                "**طلب تيمون (round-11, نقطة 2)**: وسّع تحليل N=1401 لكل الحالات الست، "
                "ووضّح أي كمية (L2/H1/الطاقة/القوى/الإجهاد) بالضبط بتحدد \"أرخص FEM بنفس الدقة\".",
                "",
                "**هاي الخلية بتغطي B1×Mooney-Rivlin و B1×Arruda-Boyce فقط** (باستخدام "
                "الـcheckpoints الموجودة أصلاً، المدربة على N=21,33 -- بدون إعادة تدريب)، "
                "بنفس الطريقة المستخدمة أصلاً لـB1×Neo-Hookean.",
                "",
                "**حالات B2 الثلاث لسا مش مشمولة** -- بتحتاج مسار GPU سريع منفصل لتوليد "
                "الحل المرجعي (fast ground truth) لهندسة B2، غير موجود لسا.",
                "",
                "**تصليح لازم قبل هاي الخلية (منجز فعلاً، commit 2f96fa9)**: الحل السريع "
                "(`solve_b1_fast_gpu`) كان مبرمج بس لـNeo-Hookean (خطأ حقيقي كان بيطلع "
                "\"too many values to unpack\" لأي مادة تانية) -- تصلح وتحقق منه على الـCPU "
                "لكل الثلاث مواد قبل ما تستعمل هون.",
                "",
                "**الوقت المتوقع**: كل مادة ~16 دقة (نفس list الـNeo-Hookean)، كل وحدة "
                "توليد بيانات مرجعية سريع (ثواني لكل عينة حتى عند N=1401) + inference سريع "
                "-- الكل المتوقع أقل من ساعة على GPU حقيقي.",
                "",
                "**مش مشمول هون (متروك كمتابعة منفصلة)**: مقارنة break-even المطابقة-بالدقة "
                "(NO وFEM الاثنين على N=1401) لهاتين المادتين -- بتحتاج قياس torch-fem "
                "حقيقي عند N=1401 لكل مادة، ما انقاس أبداً لغير Neo-Hookean.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round6_N1401_B1_OtherMaterials.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
