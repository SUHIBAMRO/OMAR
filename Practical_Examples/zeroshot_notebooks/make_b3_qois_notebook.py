"""Builds ONE notebook: real physical QoIs (energy, reaction, region
Cauchy stress) for the best B3 checkpoint (run 4, normalized inputs).
See cell_b3_qois.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_qois.py"


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
            "colab": {"provenance": [], "name": "B3_QoIs.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3: الـQoIs الفيزيائية الحقيقية (طاقة، رد فعل، إجهاد Cauchy)",
                "",
                "**السياق**: تيمون حدد سابقًا إنه معيار النجاح الحقيقي "
                "لـB3 هو **إجهاد Cauchy بمنطقة محدّدة، رد الفعل "
                "(قوة/عزم)، والطاقة** -- مش بس خطأ الإزاحة الخام. "
                "checkpoint_50000.pt (تشغيلة 4، مع input normalization) "
                "وصل لخطأ إزاحة 1.88% مجتمع -- هاد الدفتر بيفحص هل هاد "
                "بيترجم لدقة مماثلة بالكميات الفيزيائية الحقيقية.",
                "",
                "**ما في حل FEM جديد**: كل QoI محسوب مباشرة من حقل إزاحة "
                "معروف (الحقيقي أو تنبؤ الشبكة) باستخدام نفس دالة الطاقة "
                "(`total_potential_energy_B3`) يلي التدريب نفسه بيعتمد "
                "عليها -- رد الفعل = مشتقة الطاقة بالنسبة للإزاحة عند "
                "عقدة مقيّدة (نفس المبدأ التبايني يلي أي حلّال FEM "
                "بيستخدمه داخليًا)، وإجهاد Cauchy من P=مشتقة كثافة "
                "الطاقة بالنسبة لـF (autograd، مش صيغة مكتوبة يدويًا).",
                "",
                "**اتفحص محليًا على CPU قبل هيك بخطوتين**: (1) الفحص "
                "يلي عملناه هو مجموع ∂U/∂u على كل عقدة لحقل FEM حقيقي، "
                "وطلع بدقة الآلة (~1e-15). **تصحيح**: هاي نتيجة متوقعة "
                "رياضيًا من translation invariance لطاقة الـhyperelastic "
                "(أي حقل، متوازن أو لأ، المفروض يعطي نفس الصفر التقريبي "
                "-- الطاقة ما بتحس بإزاحة صلبة موحّدة). يعني هاد "
                "**sanity check ممتاز ضد أخطاء indexing/broadcasting** "
                "بكود رد الفعل، **مش إثبات equilibrium ولا إثبات كامل "
                "إن حساب رد الفعل صحيح فيزيائيًا**. (2) شغّلت المسار "
                "الحقيقي لحساب إجهاد المنطقة (مش بس حالة الصفر "
                "الاحتياطية) على مقياس بحجم كافي، بلا أخطاء.",
                "",
                "**تحذير مهم اكتشفته أثناء الفحص**: عند دقة B3 "
                "الإنتاجية (21×20×19 = 6,840 عنصر)، بس **6 نقاط Gauss** "
                "بتقع بمنطقة الأخدود المحدّدة -- أقل بكتير من الحد "
                "الأدنى الموثوق (20) يلي حدده مشروعنا نفسه سابقًا "
                "(`mesh_convergence_B3.py`). يعني `region_p99_sigma_xx` "
                "رح يطلع \"غير موثوق\" (NaN) بشكل صحيح، و"
                "`region_avg_sigma_xx` نفسه عينة إحصائية رقيقة (6 نقاط "
                "بس). **هاد بيثبت مباشرة إن تقييم إجهاد المنطقة عند "
                "هاي الدقة غير كافٍ للحكم على هدف 5-10%** -- بس **لا "
                "يثبت لوحده** إن سبب أي خطأ بالشبكة هو إن التدريب نفسه "
                "لازم يصير على شبكة أنعم (~480 ألف عنصر، يلي درسة "
                "تقارب الشبكة لـB3 بيّنتها). ممكن يكون الحل بس تقييم "
                "الإجهاد على دقة أعلى (بدون إعادة تدريب)، وممكن يكون "
                "فعلاً لازم تدريب أدق/multi-resolution. الاثنين لسا "
                "لازم ينفصلوا تجريبيًا قبل أي استنتاج نهائي.",
                "",
                "**الخلاصة الصادقة حاليًا**: الموديل (checkpoint_50000، "
                "تشغيلة 4) شغال ممتاز بـ3 من 4 معايير النجاح يلي حددها "
                "تيمون -- إزاحة (1.88%)، طاقة (0.30%)، رد فعل (2.67%). "
                "المعيار الرابع (إجهاد المنطقة) **ما عنا حكم صحيح عليه "
                "لسا**، لأن أداة القياس نفسها (6 نقاط Gauss بس) خشنة "
                "جدًا عند هاي الدقة -- هاي مش مشكلة جديدة بالموديل، هاي "
                "مشكلة resolution باختبار الإجهاد لازم تنحل قبل أي حكم.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_QoIs.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
