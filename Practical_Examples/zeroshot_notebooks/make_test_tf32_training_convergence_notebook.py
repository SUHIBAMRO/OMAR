"""Builds the follow-up TF32-training diagnostic notebook: does TF32
actually hurt FINAL accuracy, or did the first test's trajectory-matching
rule just reject normal stochastic-training noise? See
cell_test_tf32_training_convergence.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_test_tf32_training_convergence.py"


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
            "colab": {"provenance": [], "name": "Test_TF32_Training_Convergence.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# متابعة: هل TF32 فعلاً بيضر الدقة النهائية، ولا كان معيار الفحص "
                "الأول متشدد أكتر من اللازم؟",
                "",
                "**ليش هاد النوتبوك**: الفحص الأول (`Test_TF32_Training.ipynb`) رفض "
                "TF32 لأنه **مسار** الـloss (خطوة بخطوة) اختلف كثير عن fp32. بس "
                "هاد المعيار متشدد أكتر من اللازم للتدريب العشوائي: حتى **تدريبتين "
                "fp32 بنفس الدقة وseed مختلف** بيمشوا بمسار مختلف (طبيعي لـAdam) "
                "وبردو ممكن يوصلوا لموديل نهائي بنفس الجودة تقريباً. يلي فعلاً "
                "بيهمنا مش تطابق المسار -- هو **الدقة النهائية على بيانات التحقق**.",
                "",
                "**سؤال عمر**: ممكن نصلح/ننقذ الطريقة، حتى لو التسريع بس ساعة على "
                "شغلة طويلة (زي الـ1.12x يلي طلع بالفحص الأول)؟ يستاهل إذا كان حقيقي "
                "وما بضر الدقة، ما يستاهلش إذا بيكلفنا دقة.",
                "",
                "**التجربة الصحيحة -- 3 تدريبات مش 2**:",
                "- **A**: fp32, seed=1234 (نفس المرجع الأول)",
                "- **B**: fp32, seed=5678 (**نفس الدقة، seed مختلف** -- هاي بتقيس "
                "التذبذب الطبيعي بين تدريبتين حتى بدون أي TF32)",
                "- **C**: TF32, seed=1234 (نفس seed متل A -- بتعزل تأثير TF32 لحاله)",
                "",
                "**قاعدة القرار**: منقارن دقة التحقق النهائية. إذا |A-C| قريب من "
                "|A-B| (يعني TF32 ما بزود عن التذبذب الطبيعي)، TF32 **آمن نستخدمه** "
                "والتسريع الحقيقي يستاهل. إذا |A-C| أكبر بكثير من |A-B|، الرفض "
                "الأول بيضل قايم.",
                "",
                "**آمن تماماً**: نفس مصدر البيانات (مهمة خلصت خلاص، read-only)، بس "
                "أطول شوي من الفحص الأول (3000 خطوة بدل 200) عشان رقم الدقة "
                "النهائية يعني إشي -- المفروض يخلص خلال كم دقيقة على A100.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    path = f"{HERE}/Test_TF32_Training_Convergence.ipynb"
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)
