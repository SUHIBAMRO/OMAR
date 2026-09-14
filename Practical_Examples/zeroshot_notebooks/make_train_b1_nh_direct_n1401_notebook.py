"""Builds ONE notebook: trains a NEW B1 x Neo-Hookean checkpoint DIRECTLY
at N=1401 (single resolution), as an ablation against the multi-res
(N=21,33,101,201 -> zero-shot N=1401) checkpoint. Timon round-11 point 4.
See cell_train_b1_nh_direct_n1401.py for the full rationale and the
reduced sample count (100+20, not 400+100) Omar chose given the real
per-sample N=1401 ground-truth cost.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_train_b1_nh_direct_n1401.py"


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
            "colab": {"provenance": [], "name": "B1_NeoHookean_Direct_N1401_Ablation.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# تدريب B1 × Neo-Hookean مباشرة على N=1401 (ablation)",
                "",
                "**طلب تيمون (round-11, نقطة 4)**: درّب موديل جديد مباشرة على N=1401 "
                "(مش zero-shot) كمقارنة (ablation) ضد موديل الـmulti-resolution "
                "(N=21,33,101,201 → zero-shot على N=1401). نتيجة الـzero-shot **بتضل "
                "منفصلة تماماً** -- هاي مش بديل إلها، هاي تجربة مقارنة إضافية.",
                "",
                "**عدد العينات المخفّض (100 تدريب + 20 تحقق، مش 400+100 متل الـmulti-res)** "
                "-- قرار Omar الصريح، بهدف تقليل التكلفة لدراسة مقارنة (مش النتيجة الرئيسية).",
                "",
                "**تصحيح حقيقي، 2026-09-14 (لقيناه بعد تشغيل حقيقي على GPU)**: التكلفة "
                "المقدّرة أصلاً (58.54 ثانية/عينة، من "
                "`assembled_direct_convergence_production_N401_1401.json`) كانت لظرف "
                "مختلف -- حل دفعة-وحدة (`nsteps=1`). التوليد الحقيقي هون بيستخدم 10 خطوات "
                "تحميل تدريجية (`nsteps=10`)، لأنه الحل دفعة-وحدة **ما بيتقارب** عند "
                "N=1401 مع حقول مواد عشوائية (اكتشاف سابق، 2026-09-12). الرقم الحقيقي "
                "المقاس: **587 ثانية/عينة** (587.2/58.54 ≈ 10.03 -- تقريباً بالضبط 10x، "
                "يطابق nsteps=10). يعني 120 عينة بتاخد **~19.6 ساعة GPU** لتوليد البيانات "
                "بس (مش ~1.9 ساعة متل ما كان مقدّر)، قبل ما يبلش التدريب نفسه أصلاً. "
                "الحل السريع (assembled_direct) شغال صح -- كان خطأ بالتقدير نفسه، مش "
                "بالكود.",
                "",
                "**الخطوات**: (1) توليد بيانات FEM حقيقية عند N=1401 (الحل السريع)، "
                "(2) التدريب (نفس البروتوكول: 2000 epoch كحد أقصى، early stopping صبر 8، "
                "batch_size=8، lr=2e-3)، (3) فحص الدقة الحقيقية عند N=1401 مقابل FEM حقيقي، "
                "(4) قياس سرعة الاستدلال، (5) مقارنة مباشرة مع أرقام الـmulti-res الموثقة "
                "أصلاً.",
                "",
                "**كل شي محفوظ على Drive تحت `zeroshot_B1_neo_hookean_direct_n1401/`** -- "
                "منفصل تماماً عن checkpoint الـmulti-res والـzero-shot الأصلي، ما بيلمسهم.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B1_NeoHookean_Direct_N1401_Ablation.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
