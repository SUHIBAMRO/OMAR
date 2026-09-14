"""Builds ONE small diagnostic notebook: tests whether B1 x Neo-Hookean
at N=1401 still converges with FEWER incremental load steps (5 or 3)
than the current default (10) -- see cell_test_fewer_load_steps_n1401.py
for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_test_fewer_load_steps_n1401.py"


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
            "colab": {"provenance": [], "name": "Test_FewerLoadSteps_N1401.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# فحص سريع: هل نقدر نقلل عدد خطوات التحميل عند N=1401؟",
                "",
                "**ليش هاد النوتبوك**: مهمة #24 (تدريب مباشر عند N=1401) عم تولّد "
                "بيانات بمعدل 587 ثانية/عينة -- 10 أضعاف التقدير الأصلي، لأنها "
                "محتاجة 10 خطوات تحميل تدريجية (`nsteps=10`) عشان نيوتن يتقارب "
                "(خطوة وحدة `nsteps=1` ما بتتقارب أصلاً عند هاد الحجم، اكتشاف سابق "
                "2026-09-12). **بس كل خطوة من العشرة بتتقارب لدقة أعلى بكثير من "
                "المطلوب فعلاً** (مثلاً 1.4e-09 مقابل التسامح المطلوب 1e-7) -- يعني "
                "محتمل (مش مؤكد) إنه خطوات أقل وأكبر تكفي.",
                "",
                "**هاد النوتبوك آمن تماماً**: بيجرب 3 عينات بس عند nsteps=5 و3 "
                "(بالمقارنة مع nsteps=10 كمرجع)، **بدون ما يكتب أي شي عالمجلد "
                "الحقيقي** -- تشخيص بحت، ما بأثر على النوتبوك التاني الشغال "
                "(`B1_NeoHookean_Direct_N1401_Ablation.ipynb`) نهائياً.",
                "",
                "**إذا طلعت نتيجة \"SAFE to use\" لقيمة أقل من 10**: نقدر نكمّل "
                "توليد باقي العينات (يلي لسا ما اتولدت بالنوتبوك التاني) بـ"
                "`--nsteps <القيمة الأقل>` -- بدون ما نخسر أي شي من العينات "
                "المولّدة أصلاً بـnsteps=10 (تبقى صحيحة، الفرق بس بطريقة الحل "
                "مش بالنتيجة النهائية).",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    path = f"{HERE}/Test_FewerLoadSteps_N1401.ipynb"
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)
