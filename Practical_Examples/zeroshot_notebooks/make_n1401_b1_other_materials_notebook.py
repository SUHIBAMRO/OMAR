"""Builds ONE notebook: extends the N=1401 accuracy/QoI analysis (Timon
round-11 point 2) to ALL 5 remaining cases -- B1 x Mooney-Rivlin,
B1 x Arruda-Boyce, and all three of B2 (Neo-Hookean, Mooney-Rivlin,
Arruda-Boyce) -- using their EXISTING zero-shot checkpoints, no
retraining. See cell_n1401_b1_other_materials.py for the full rationale,
the prerequisite fixes (material-parameter arity generalization, the new
B2 fast ground-truth solver and accuracy/peak-stress pipeline), and what
is deliberately NOT included (resolution-matched break-even for these 5
cases -- flagged as a separate follow-up needing torchfem_comparison.py
generalized past B1xNeo-Hookean).
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_n1401_b1_other_materials.py"


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
            "colab": {"provenance": [], "name": "Round6_N1401_AllRemainingCases.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# N=1401 accuracy/QoI لكل الحالات الخمس الباقية (B1×2 و B2×3)",
                "",
                "**طلب تيمون (round-11, نقطة 2)**: وسّع تحليل N=1401 لكل الحالات الست، "
                "ووضّح أي كمية (L2/H1/الطاقة/الإجهاد) بالضبط بتحدد \"أرخص FEM بنفس الدقة\".",
                "",
                "**هاي الخلية بتغطي الحالات الخمس الباقية كلهم** (B1×Mooney-Rivlin، "
                "B1×Arruda-Boyce، وB2 بمواتها التلاتة) -- الـcheckpoints الموجودة أصلاً "
                "(N=21,33)، بدون إعادة تدريب. النطاق الكامل، مش جزء منه فقط -- بطلب Omar "
                "الصريح.",
                "",
                "**شغل تحضيري حقيقي انعمل قبل هاي الخلية (كله موثق ومتحقق منه، مش مجرد "
                "افتراض)**:",
                "1. تصليح خطأ حقيقي: الحل السريع كان مبرمج بس لـNeo-Hookean (كان بينهار "
                "مع Mooney-Rivlin وArruda-Boyce) -- تصلح وتحقق منه لكل المواد الثلاث.",
                "2. بناء مسار GPU سريع جديد بالكامل لـB2 (`solve_b2_fast_gpu`) -- ما كان "
                "موجود أبداً قبل هلق. تحقق منه ضد الحل المرجعي البطيء لكل المواد الثلاث "
                "(دقة ~1e-11، نفس مستوى الدقة الموجود أصلاً لـB1).",
                "3. بناء كامل خط تقييم الدقة والإجهاد لـB2 (`evaluate_no_accuracy_at_n1401_b2`، "
                "`run_no_peak_stress_fixed_location_b2`) -- فحص شامل محلي (CPU، موديل غير "
                "مدرب) قبل ما ينستخدم هون.",
                "",
                "**ملاحظة مهمة عن B2**: كمية \"القوى التفاعلية\" (reaction forces) مش "
                "محسوبة لـB2 -- هاي أصلاً كانت \"B1 بس\" بكل المشروع من قبل (حدود التماثل "
                "بـB2 كل وحدة فيها مركبة وحدة بس مثبتة، ومافي طريقة معتمدة أصلاً بالمشروع "
                "لدمجهم بمقارنة وحدة) -- فجوة صادقة موثقة، مش تقصير.",
                "",
                "**مسارات الـcheckpoints** (تأكدت منهم من النوتبوكات الحقيقية الي بنتهم):",
                "- B1: `zeroshot_B1_{material}/model_best.pt`",
                "- B2: `zeroshot_B2_{material}_fixedsel/model_best.pt` (لاحظ لاحقة "
                "`_fixedsel` -- هاي النسخة المصححة من Round 6، مش النسخة الأصلية).",
                "",
                "**الوقت المتوقع**: 5 حالات × ~16 دقة لكل وحدة -- توليد مرجعي سريع "
                "لـB1 وB2 الاثنين (ثواني لكل عينة حتى N=1401) -- الكل المتوقع أقل من "
                "ساعتين على GPU حقيقي.",
                "",
                "**مش مشمول هون (شغل منفصل تاني، لسا ما بلش)**: مقارنة break-even "
                "المطابقة-بالدقة (NO وFEM الاثنين على N=1401) لهاي الحالات الخمس -- "
                "torch-fem (المكتبة الخارجية المستخدمة لقياس السرعة) مبرمجة حالياً بس "
                "لـNeo-Hookean ولهندسة B1 (فرضية \"كل حدود التثبيت عندها مركبتين مثبتتين\" "
                "غلط لـB2). تعميمها شغلة منفصلة بحجم مشابه لللي عملناه هلق، ولسا ما بلشت "
                "فيها -- حتى ما نخاطر بأرقام round-9 الموثقة أصلاً بدون تحقق كافي.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/Round6_N1401_AllRemainingCases.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
