"""Builds ONE notebook: the B3 geometry figure Timon's round-14 reply
asked for directly ("Can you please show also a geometry of the
problem"). Runnable by Omar himself in Colab -- no GPU or torch-fem
install needed, pure numpy/matplotlib, seconds to run. See
cell_b3_geometry_figure.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_geometry_figure.py"


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
            "colab": {"provenance": [], "name": "B3_Geometry_Figure.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            md(
                "# B3 (bushing بالـgroove): صورة الهندسة",
                "",
                "تيمون طلب صراحة بردّه على راوند 14: \"Can you please show "
                "also a geometry of the problem\" -- يعني بدو **صورة**، مش "
                "شرح إضافي.",
                "",
                "هاي الخلية بتبني الصورة باستخدام **نفس كود الهندسة "
                "الحقيقي** (`data_generate_B3.py`) و**نفس الأرقام الفيزيائية "
                "بالضبط** يلي استخدمناها بدراسة تقارب الشبكة (R_in0=0.5, "
                "R_out=1.0, Lz=1.0, عمق الأخدود=0.05, نصف عرضه=0.15, زاوية "
                "الدوران phi=0.05 راديان -- نفس الحالة الوحيدة الحتمية "
                "يلي دراسة التقارب كلها استخدمتها). الشي الوحيد المختلف هو "
                "**دقة الشبكة** (25,10,23) -- أخف من دقات دراسة التقارب "
                "الحقيقية، بس بس لغرض إنو الصورة تبين واضحة (شبكة دقيقة "
                "كتير بتصير نقط مزدحمة ما بتنشاف بصورة ثابتة). الهندسة "
                "نفسها (نصف القطر، شكل الأخدود) **حقيقية 100%**، مش تقريب.",
                "",
                "ثلاث لوحات: (1) الشكل غير المشوّه، (2) الشكل بعد الدوران "
                "الحقيقي (rigid-body rotation)، (3) مقطع محوري مُقرَّب "
                "بيبين الأخدود بوضوح.",
                "",
                "**ما بتحتاج GPU ولا تثبيت torch-fem** -- كود الهندسة "
                "numpy بس، بتاخد ثواني.",
                "",
                "بالنسبة لقاعدة عمر الدائمة (2026-09-21): هاي الخلية بتحفظ "
                "الصورة على Drive **وكمان** بتعرضها مباشرة بآخر الخلية.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Geometry_Figure.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
