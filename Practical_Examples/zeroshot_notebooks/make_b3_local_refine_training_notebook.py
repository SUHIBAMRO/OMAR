"""Builds ONE notebook: real B3 training run with local DEM-integration
refinement near the groove region (Timon's request, item 1 of the
remaining list: "Local refined DEM training integration"). See
cell_b3_local_refine_training.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_local_refine_training.py"


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
            "colab": {"provenance": [], "name": "B3_Local_Refine_Training.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3: تدريب حقيقي بتكامل DEM أدق محليًا حول الأخدود",
                "",
                "**البند الأول من قائمة عمر المتبقية**: \"Local refined "
                "DEM training integration\". هاد **مش تغيير دقة** -- نفس "
                "شبكة الـoperator بالضبط (6,840 عنصر، 21×20×19)، نفس "
                "الـarchitecture، نفس الـnormalization (منسوخة من "
                "التشغيلة 4 بلا تغيير)، نفس توزيعات المادة/الحمل، نفس "
                "50,000 تكرار. **الفرق الوحيد**: العناصر يلي بتلمس منطقة "
                "الأخدود بتحسب مساهمتها بالطاقة بتكامل أدق (n_sub=10، "
                "566 نقطة -- نفس المستوى يلي درسة التقارب لقيته مستقر) "
                "بدل القاعدة الافتراضية (8 نقاط)، بدل ما نرفع دقة "
                "الشبكة كلها -- بالضبط طلب تيمون: فصل discretization "
                "الـoperator عن شبكة تكامل DEM.",
                "",
                "**ليش ما رح تاخد ~11 ساعة متل تجربة رفع الدقة السابقة**: "
                "عدد العناصر/العقد **ما بيتغيّر أبدًا** هون -- بس عدد "
                "قليل من العناصر القريبة من الأخدود (2 من 2184 بحجم "
                "toy؛ نسبة مشابهة متوقعة بدقة الإنتاج) بتاخد نقاط تكامل "
                "إضافية. معدل الثانية/تكرار المتوقع قريب من تشغيلة 4 "
                "(~0.14 ثانية)، مش 0.784 ثانية يلي احتاجتها رفع الدقة "
                "الكاملة. لهيك هاي التشغيلة بتستخدم **كامل ميزانية "
                "50,000 تكرار** (مش pilot مختصر) -- ما في تعارض "
                "(confound) بين الدقة وعدد التكرارات هالمرة.",
                "",
                "**اتفحص محليًا على CPU بأربع خطوات حقيقية قبل أي تشغيلة "
                "GPU**: (1) فحص identity -- عند n_sub=2 (نفس رتبة "
                "القاعدة الافتراضية)، الطاقة المحسوبة بالتكامل المحلي "
                "طابقت الطاقة القياسية بدقة ~1e-12، **لأي** قناع عناصر "
                "(حتى قناع مبالغ فيه نصف الشبكة)؛ (2) فحص تكرير حقيقي -- "
                "عند n_sub=4/8، مساهمة عناصر المنطقة تغيّرت بمقدار صغير "
                "ومعقول (<0.1% من الطاقة الكلية)، بلا انفجار؛ (3) فحص "
                "التدرّج (gradient) -- التدرّجات طلعت منتهية وغير صفرية "
                "عبر autograd العادي، بلا أي معالجة خاصة؛ (4) تشغيلة "
                "تدريب حقيقية كاملة بحجم toy (بيانات FEM حقيقية، حلقة "
                "تدريب حقيقية، checkpoint حقيقي)، بعدين تقييمها بأداة "
                "sweep التقارب الموجودة أصلاً -- بلا أي خطأ بكل الأنبوب.",
                "",
                "**التقييم بعد التدريب** بيعيد استخدام نفس sweep التقارب "
                "(`evaluate_B3_region_local_refine.py`) بلا تعديل، "
                "فالنتيجة قابلة للمقارنة المباشرة مع خط الأساس المستقر "
                "لتشغيلة 4 (**~62.2-62.3%**).",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Local_Refine_Training.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
