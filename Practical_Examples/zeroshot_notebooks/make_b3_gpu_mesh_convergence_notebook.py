"""Builds ONE notebook: B3's mesh-convergence study extended to GPU
resolutions, per Omar's own explicit instruction (2026-09-21) to keep
refining until the region-Cauchy-stress QoI plateaus before designating
a final fine reference or trusting the 5%/2%/1% required-resolution
table. See cell_b3_gpu_mesh_convergence.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_gpu_mesh_convergence.py"


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
            "colab": {"provenance": [], "name": "B3_GPU_MeshConvergence.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3 (bushing بالـgroove): تقارب الشبكة على GPU لدقات أعلى",
                "",
                "على الـCPU وصلنا لحد 3,240 عنصر، وإجهاد Cauchy بالمنطقة "
                "(region_avg_sigma_xx) **كان لسا طالع**: 1.56 → 2.30 → 2.53 → "
                "2.75 — يعني المرجع الدقيق (fine reference) يلي استخدمناه "
                "(9,464 عنصر) **مش مؤكد إنو كافي** لهاد الـQoI المحلي بالذات، "
                "حتى لو كان كافي لـL2/H1 (عمر، 2026-09-21).",
                "",
                "هاي الخلية بتمد نفس دراسة التقارب (بنفس الكود المتحقق منه "
                "أصلًا -- `mesh_convergence_B3.py`، ما تغير شي بالفيزياء أو "
                "الهندسة أو الـBCs) لدقات أعلى بكتير عالـGPU (لحد 243,360 "
                "عنصر كمرجع نهائي)، وبتطبع بوضوح **نسبة التغيّر بإجهاد "
                "المنطقة بين كل دقتين متتاليتين** -- إذا ما استقرت لنسبة "
                "صغيرة (كام بالمية) حتى بآخر صف، هاد لسا مش كافي وبدنا نمد "
                "أكتر.",
                "",
                "بعدها بتحسب **جدول الدقة المطلوبة (5%/2%/1%)** لكل الـQoIs "
                "السبعة (L2, H1, الطاقة, قوة رد الفعل, العزم, متوسط/p99 "
                "إجهاد Cauchy بالمنطقة) -- بس مع تنبيه واضح إذا الجدول لسا "
                "مؤقت لحد ما يثبت استقرار إجهاد المنطقة تحديدًا.",
                "",
                "**الكلفة المتوقعة**: 8 حلول (7 دقات + المرجع النهائي)، "
                "أكبرهم ~243 ألف عنصر -- على A100 المفروض تاخد دقائق مش "
                "ساعات (المشروع سبق حل شبكات 2D أكبر بكتير على نفس الـGPU).",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_GPU_MeshConvergence.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
