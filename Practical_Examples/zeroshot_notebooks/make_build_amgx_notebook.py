"""Builds ONE notebook: an experimental from-source build of NVIDIA AmgX
in Colab, so torch-fem's CUDA AMG preconditioner path becomes available,
followed by a real retest of the exact 105,456-element B8-final case
that failed on GPU/Jacobi (ConvergenceError: CG did not reach 1e-08
within the iteration limit; Newton-Raphson did not converge in
increment 4 after 10 cutbacks). See cell_build_amgx.py for the full
build logic and rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_build_amgx.py"


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
            "colab": {"provenance": [], "name": "AmgX_Build_And_Test.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# بناء AmgX من المصدر + اختبار AMG الحقيقي على GPU (تجريبي)",
                "",
                "**السياق**: B8-final فشل بخطأ حقيقي وواضح عند 105,456 عنصر "
                "(ConvergenceError: CG ما وصل لـ1e-08 حتى بعد حد التكرارات "
                "الأقصى، Newton-Raphson ما تقارب بالخطوة 4 بعد 10 محاولات "
                "تصغير). السبب المرجح: التباين الهائل بصلابة الفولاذ عن "
                "المطاط (~294,000:1) بيصعّب التقارب على حلّال Jacobi+CG "
                "العادي عند هاد الحجم.",
                "",
                "**الحل المقترح**: AMG (Algebraic Multigrid) -- نوع حلّال "
                "أقوى بكتير مصمم خصيصًا لهيك حالات. على الـCPU متوفر "
                "جاهز (pyamg)، بس اختبار أولي بحجم أصغر (30,576 عنصر) طلع "
                "غير حاسم (نفس عدد تكرارات Jacobi، وأبطأ). على الـGPU، "
                "torch-fem بيحتاج مكتبة AmgX من NVIDIA -- وهاي **ما إلها "
                "تحميل جاهز (pip)**، لازم تُبنى من المصدر.",
                "",
                "**هاي الخلية بتعمل**:",
                "1. تنزل مصدر AmgX من GitHub الرسمي (NVIDIA/AMGX).",
                "2. تبنيها (compile) باستخدام cmake + الـCUDA الموجودة "
                "أصلاً بـColab، مع قراءة قدرة الـGPU الحقيقية (compute "
                "capability) تلقائيًا بدل افتراض رقم ثابت.",
                "3. إذا فشل البناء بمشكلة معروفة (ربط OpenMP)، بتعيد "
                "المحاولة بإعدادات ربط إضافية.",
                "4. تدور على الملف الناتج (`libamgxsh.so`) فعليًا (مو "
                "افتراض مكانه)، وتربطها مع torch-fem.",
                "5. تتحقق إنه torch-fem فعلاً عم يشوف AmgX متوفرة.",
                "6. **بعدين بس** تجرب نفس الحالة يلي فشلت بالضبط (105,456 "
                "عنصر) -- **بدون أي تغيير** بالهندسة، المواد، منطقة "
                "القياس، الدقة (FP64)، مسار التحميل، أو حدود التقارب -- "
                "الفرق الوحيد: AMG بدل Jacobi، وعلى GPU هالمرة (مو CPU).",
                "",
                "⚠️ **تحذير صادق**: هاد بناء حقيقي من الصفر لمكتبة معقدة، "
                "**ما تم اختباره من قبل بهاي الجلسة** (ما في وصول لـGPU "
                "بجلسة العمل الحالية). فيه احتمال حقيقي يفشل بخطوة ما -- "
                "كل خطوة بتطبع مخرجاتها الحقيقية كاملة عشان نقدر نشخص أي "
                "مشكلة من الطباعة نفسها، مش تخمين. الوقت المتوقع للبناء "
                "لحاله: 15-40 دقيقة تقريبًا، قبل حتى ما توصل لتجربة الحل.",
                "",
                "**لو نجح**: عندنا دليل حقيقي إنه AMG بيحل مشكلة التقارب "
                "بسرعة GPU كاملة. **لو فشل بأي خطوة**: ابعت المخرجات "
                "كاملة، ومنشخصها سوا خطوة بخطوة.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/AmgX_Build_And_Test.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
