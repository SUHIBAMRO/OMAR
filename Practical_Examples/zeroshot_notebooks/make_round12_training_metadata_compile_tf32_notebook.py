"""Builds ONE notebook: Timon round-12 points 2 (training-cost summary,
existing data only) and 3 (compile+TF32 inference extended to all six
cases). See cell_round12_training_metadata_and_compile_tf32.py for the
full rationale and the explicit note that point 2's own controlled
re-run (matching sample counts between direct-N1401 and multi-res) is
skipped per Timon's own words that it isn't worth the time here.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_round12_training_metadata_and_compile_tf32.py"


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
            "colab": {"provenance": [], "name": "Round12_TrainingMetadata_CompileTF32.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# round-12 نقطة 2 (جدول كلفة التدريب) ونقطة 3 (compile+TF32 لكل الحالات)",
                "",
                "**نقطة 2 (الجزء السهل بس)**: جدول يجمع من ملفات `metrics_history.json` "
                "الموجودة أصلاً عالـDrive لكل تجربة تدريب: الدقات، عدد العينات، "
                "الـepochs/steps، الوقت الكلي، والتكلفة لكل عينة/step. **ما في تدريب "
                "جديد هون** -- تيمون نفسه طلب صراحة ما نضيع وقت بتجربة \"مضبوطة\" "
                "(نفس عدد العينات) للدقة العالية هلق (\"we should not waste time on "
                "the fine resolution training here\"), فهاي بس تنظيم الأرقام "
                "الموجودة. **الذاكرة القصوى أثناء التدريب مش مسجلة بأي تجربة** -- "
                "الجدول بيقول هيك صراحة، ما بنخمن رقم.",
                "",
                "**نقطة 3**: استخدام رقم compile+TF32 المُحسّن بدل eager. كان موجود "
                "بس لـB1×Neo-Hookean (الحالة الرئيسية) -- الدالة نفسها "
                "(`profile_with_torch_compile`) كانت مبنية لـB1 بس. **كود جديد**: "
                "`profile_with_torch_compile_b2` (نفس المنطق تماماً، بس بتستخدم "
                "دالة التنبؤ الخاصة بـB2). هاي النوتة بتقيس compile+TF32 لكل الست "
                "حالات، بالشيكبوينت النهائي لكل وحدة.",
                "",
                "**الوقت المتوقع**: نقطة 2 فورية (قراءة ملفات بس). نقطة 3 حوالي "
                "15-20 دقيقة تقريباً/ست حالات (كل حالة عندها compile warmup + "
                "200 تكرار).",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round12_TrainingMetadata_CompileTF32.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
