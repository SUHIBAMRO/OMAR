"""Builds ONE notebook: Option A (B3 with a 4x sharper groove, depth=0.20)
mesh-convergence study extended to GPU resolutions into and above the
10^5-10^6-element range the advisor asked about. Same corrected
discipline as the B8 GPU study: no "converged reference" claims until
a reference-to-reference comparison actually supports it, no using a
rising true peak stress as evidence of difficulty, region-Cauchy field
error as the primary local QoI, fixed stress-evaluation region. See
cell_b3_groove_gpu_mesh_convergence.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_groove_gpu_mesh_convergence.py"


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
            "colab": {"provenance": [], "name": "B3_Groove_Sharp_GPU_MeshConvergence.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# Option A (B3 بأخدود أحدّ 4x، depth=0.20): تقارب الشبكة "
                "على GPU -- داخل نطاق 10^5-10^6 عنصر",
                "",
                "دراسة CPU لقت مشكلتين حقيقيتين وصلحتهم (مش تحايل): خطوات "
                "التحميل الافتراضية (11) فشلت بالتقارب عند 20% بس من "
                "الحمل -- صلحناها لـ21؛ وr_grading=2.5 عمل مشكلة ill-"
                "conditioning حقيقية بالحلّال -- صلحناها بـr_grading=1.5. "
                "بعد الإصلاح، خطأ الإزاحة الكلي (disp_L2) بيتقارب بنفس "
                "سرعة B3 الأصلي تقريبًا.",
                "",
                "**تعليمات صريحة انلتزم فيها هون** (نفس الانضباط يلي "
                "انطبق على B8):",
                "1. ما منسمي أي شبكة \"مرجع متقارب\" لحد ما فحص المرجع-"
                "مقابل-المرجع يثبت هيك فعليًا.",
                "2. ما منستخدم ارتفاع true_max كدليل صعوبة -- بيبقى "
                "تشخيصي بس.",
                "3. منطقة قياس الإجهاد (عند أعمق نقطة بالأخدود، نصف "
                "قطرها = 2× نصف قطر انحناء الأخدود) **ثابتة** لأنها "
                "مرتبطة بهندسة الأخدود الثابتة نفسها (depth=0.20, "
                "half_width=0.15)، مش بدقة الشبكة.",
                "4. خطأ حقل Cauchy الإقليمي هو الـQoI المحلي الأساسي.",
                "",
                "**السلّم**: دقات الـCPU يلي سبق تقريرها (336 لـ18,200 "
                "عنصر) + 3 دقات GPU جديدة توصل لنطاق 10^5-10^6 (79k, "
                "202k, 480k عنصر)، مقارنة بمرجع أدق (~2.21 مليون عنصر)، "
                "المرجع نفسه محقق مقابل مرجع أقدم (~1.07 مليون).",
                "",
                "**الهدف المحدد**: هل خطأ الـQoI الإقليمي بيضل تقريبًا "
                "5-10% داخل نطاق 10^5-10^6 عنصر؟",
                "",
                "بالنسبة لقاعدة عمر الدائمة (2026-09-21): هاي الخلية "
                "بتولّد صورتين، بتحفظهم على Drive **وكمان** بتعرضهم "
                "مباشرة بآخر الخلية.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Groove_Sharp_GPU_MeshConvergence.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
