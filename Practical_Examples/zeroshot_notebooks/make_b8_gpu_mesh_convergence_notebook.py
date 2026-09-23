"""Builds ONE notebook: B8-FINAL's (laminated annular seismic bearing,
Option B, real published-source geometry/materials) mesh-convergence
study extended to GPU resolutions into and above the 10^5-10^6-element
range the advisor asked about.

This replaces the earlier B8-prototype notebook (frozen, kept as-is at
B8_Prototype_GPU_MeshConvergence.ipynb -- its real result, 7.53% region-
Cauchy field error at 791,864 elements, stays valid and is not
discarded). B8-final differs in three real ways, per a 2026-09-23
technical review: real geometry/materials from a published source
(Kalantari & Rofooei 2010) replacing arbitrary numbers; a rubber-only
stress-QoI region (no more interpolation across the rubber/steel
discontinuity); and real linear-elastic steel (E=200 GPa) via a St.
Venant-Kirchhoff psi branch, replacing the prototype's
SHIM_STIFFNESS_RATIO=100 Neo-Hookean proxy. See
cell_b8_gpu_mesh_convergence.py for the full rationale and the real CPU
evidence gathered before this GPU cell was built.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b8_gpu_mesh_convergence.py"


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
            "colab": {"provenance": [], "name": "B8_GPU_MeshConvergence.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B8-FINAL (الوسادة الزلزالية الحقيقية): تقارب الشبكة على "
                "GPU -- داخل نطاق 10^5-10^6 عنصر",
                "",
                "هاي النسخة **النهائية** من B8، مبنية على مصدر منشور حقيقي "
                "(Kalantari & Rofooei, 2010) -- مو أرقام تقريبية متل النسخة "
                "الأولى (B8-prototype، محفوظة لحالها وما انحذفت، ونتيجتها "
                "الحقيقية 7.53% عند 791,864 عنصر لسا صحيحة وموثقة).",
                "",
                "**شو تغيّر عن النسخة الأولى**:",
                "1. **الهندسة والمواد حقيقية 100%**: القطر الداخلي/الخارجي "
                "30mm/152mm، 20 طبقة مطاط × 3mm، 19 شيم فولاذ × 3mm. "
                "المطاط: G=0.68 MPa (تصحيح لرقم قديم غير موثق كان 0.86)، "
                "K=2000 MPa. الفولاذ: E=200 GPa حقيقي (بدل أي تقريب).",
                "2. **الفولاذ فولاذ حقيقي فعليًا**: تم إلغاء "
                "\"SHIM_STIFFNESS_RATIO=100\" (كان يمثل الفولاذ كمطاط أقوى "
                "100 مرة بس) -- الفولاذ هلأ عنده معادلة مادة خاصة فيه "
                "(St. Venant-Kirchhoff) موازية للمطاط (Neo-Hookean) بنفس "
                "الموديل، محسوبة ومتحقق منها رياضيًا قبل الاستخدام.",
                "3. **منطقة قياس الإجهاد صارت على المطاط بس**: النسخة "
                "الأولى كانت تقيس عند منتصف الشيم (تخلط مطاط وفولاذ). "
                "هلأ المنطقة محصورة داخل طبقة المطاط الأولى فقط، جنب "
                "السطح البيني مع الشيم -- بضمانة برمجية (assertion) إنه "
                "ولا عنصر فولاذ يدخل الحساب أبدًا.",
                "",
                "**فحص حقيقي انعمل قبل هاي الخلية**: نسبة صلابة الفولاذ "
                "للمطاط الحقيقية (~294,000:1) أعلى بكتير من نسبة الـ100:1 "
                "المستخدمة سابقًا -- فحصنا هل هاد بيسبب مشكلة تقارب "
                "(conditioning) بالحل الرياضي متل يلي صار بنموذج تاني "
                "بنفس اليوم. النتيجة: لأ -- حل حقيقي على الكمبيوتر بـ15,600 "
                "عنصر بالمواد الحقيقية تقارب بنفس النمط الصحي المعتاد "
                "(5 تكرارات بالخطوة الأولى، بعدين 2 بس لكل خطوة، 23 "
                "بالمجموع). المشكلة يلي صارت بالنموذج التاني كانت مرتبطة "
                "بميزة هندسية حادة (أخدود)، مش موجودة بهندسة B8 الملساء.",
                "",
                "**تعليمات دائمة انلتزم فيها**: منطقة القياس ثابتة بالمكان "
                "الفيزيائي (ما بتتغير مع الدقة)؛ region-Cauchy field error "
                "هو الـQoI الأساسي؛ true_max تشخيصي بس؛ لازم فحص مرجع-مقابل"
                "-مرجع قبل اعتماد أي مرجع كمتقارب.",
                "",
                "**درس مستفاد من نفس اليوم (نوتبوك B3/الأخدود)**: هاي "
                "الخلية بتحل السلّم كامل (بما فيه المرجع الرئيسي) **قبل** "
                "ما تحاول مرجع الفحص الأدق -- هيك إذا صار أي تعليق أو بطء "
                "بأكبر حجم، نتائج السلّم الحقيقية محفوظة عالـDrive أصلًا "
                "وما بتنضاع.",
                "",
                "**السلّم**: (9,5) لـ(21,11) على الـCPU-scale، بعدين "
                "(53,27)/(77,39)/(101,51)/(129,65)/(153,77) على الـGPU "
                "توصل لنطاق 10^5-10^6 (105k-901k عنصر)، مقارنة بمرجع "
                "رئيسي ~1.05 مليون عنصر (165,83) وفحص مرجع أدق ~1.21 "
                "مليون عنصر (177,89).",
                "",
                "بالنسبة لقاعدة عمر الدائمة (2026-09-21): هاي الخلية بتولّد "
                "صورتين (فحص المرجعين، وملخص التقارب الكامل)، بتحفظهم على "
                "Drive **وكمان** بتعرضهم مباشرة بآخر الخلية.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B8_GPU_MeshConvergence.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
