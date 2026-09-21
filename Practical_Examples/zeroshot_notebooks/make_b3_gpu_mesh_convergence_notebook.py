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
                "# B3 (bushing بالـgroove): تقارب الشبكة على GPU -- مرجع أدق "
                "(مراجعة عمر الـ11 نقطة، 2026-09-21)",
                "",
                "الـrun الأول على GPU وصل لمرجع 243,360 عنصر (81,40,79)، بس "
                "آخر تغيّر بإجهاد المنطقة (region_avg) كان لسا ~0.87% "
                "(و~1.17% للـp99) -- قريب كتير من حد الـ1% يلي عم نّدعيه. "
                "عمر طلب صراحة: **مرجع إضافي أدق (~424,128 عنصر، تقريبًا "
                "97,48,95) للتأكد** -- إذا التغيّر نزل تحت 0.5-0.7% بهاد "
                "المرجع الجديد، ادعاء الـ1% بيصير أقوى بكتير؛ إذا لأ، "
                "بنكتب صراحة \"1% مش محقق / مؤقت\".",
                "",
                "هاي الخلية معدّلة كمان لتستخدم `mesh_convergence_B3.py` "
                "المُعاد كتابته بالكامل (خطأ حقل Cauchy الكامل بدل sigma_xx "
                "بس، عيّنات مرجّحة بالحجم على نقاط Gauss بدل مراكز العناصر، "
                "قوة رد الفعل موثقة إنها مش صفر فعليًا فالعزم هو الـQoI "
                "الأساسي، وتسمية \"total strain energy\" بدل \"tangent "
                "energy\" الخاطئة).",
                "",
                "بتحل **مرجعين**: القديم (81,40,79) والجديد الأدق "
                "(97,48,95)، بتقارنهم مباشرة (region_avg, p99 إذا موثوق, "
                "خطأ حقل Cauchy الكامل) **قبل أي شي تاني** -- وبعدين بتستخدم "
                "الأدق كمرجع نهائي لجدول الدقة المطلوبة (5%/2%/1%) لكل "
                "الـQoIs الثمانية.",
                "",
                "**الكلفة المتوقعة**: 9 حلول (7 دقات + مرجعين)، أكبرهم "
                "~424 ألف عنصر -- على A100 المفروض تاخد دقائق (المشروع سبق "
                "حل شبكات 2D أكبر بكتير على نفس الـGPU)، بس ممكن تاخد وقت "
                "أطول شوي من الـrun الأول بسبب المرجع الإضافي الأكبر.",
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
