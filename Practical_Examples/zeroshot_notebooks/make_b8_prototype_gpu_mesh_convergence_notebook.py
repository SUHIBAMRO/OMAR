"""Builds ONE notebook: B8's (laminated annular seismic bearing, Option B)
mesh-convergence study extended to GPU resolutions into and above the
10^5-10^6-element range the advisor asked about, per explicit
instructions given after the CPU-only study: do not call the CPU
5,184-element mesh a converged reference, do not use rising true peak
stress as evidence of difficulty, keep the region-Cauchy field error as
the primary local QoI with true_max diagnostic-only, and include a
reference-to-reference comparison. See cell_b8_gpu_mesh_convergence.py
for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b8_prototype_gpu_mesh_convergence.py"


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
            "colab": {"provenance": [], "name": "B8_Prototype_GPU_MeshConvergence.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B8 (الـlaminated seismic bearing): تقارب الشبكة على GPU "
                "-- داخل نطاق 10^5-10^6 عنصر",
                "",
                "دراسة CPU السابقة أعطت تقارب حقيقي ونظيف (29.2%->21.2%->"
                "14.1%->7.0% لخطأ حقل Cauchy المحلي عند 576-3600 عنصر)، بس "
                "المرجع (5,184 عنصر) **مش مؤكد إنه متقارب** -- هاي الخلية "
                "بتكمل صح: بتحل مرجعين حقيقيين (القديم ~1.05 مليون عنصر، "
                "الجديد الأدق ~2.9 مليون عنصر) وبتقارنهم مباشرة **قبل** ما "
                "تعتبر أي وحدة منهم مرجع نهائي موثوق -- بالضبط نفس الفحص "
                "يلي انعمل لـB3.",
                "",
                "**تعليمات صريحة انلتزم فيها هون**:",
                "1. ما منسمي أي شبكة \"مرجع متقارب\" لحد ما فحص المرجع-مقابل"
                "-المرجع يثبت هيك فعليًا.",
                "2. ما منستخدم ارتفاع أعلى إجهاد حقيقي (true_max) كدليل "
                "إنو B8 أصعب من B3 -- رقم بيرتفع مع الدقة ممكن يعني \"لسا "
                "الشبكة ناقصة\" بنفس قد ما يعني \"فيه ميزة حادة فعلاً\"، "
                "وما فيه طريقة تفرق بينهم من رقم واحد بس. true_max بيبقى "
                "تشخيصي بس، متل ما هو الحال دائمًا بـB1/B2/B3.",
                "3. منطقة قياس الإجهاد (r=R_out, theta=0, منتصف أول شيم "
                "داخلي، نصف قطر المنطقة=6×سماكة الشيم) **ثابتة من هلق "
                "وطالعة**، ما بتتغير مع الدقة.",
                "4. خطأ حقل Cauchy الإقليمي (region-Cauchy field error) هو "
                "الـQoI المحلي **الأساسي**، مش avg ولا true_max.",
                "",
                "**السلّم**: نفس دقات الـCPU (576 لـ5,184 عنصر) + دقات GPU "
                "جديدة توصل لنطاق 10^5-10^6 (19k, 51k, 136k, 373k, 792k "
                "عنصر)، مقارنة بالمرجع الأدق (~2.9 مليون).",
                "",
                "**الهدف المحدد**: هل خطأ الـQoI الإقليمي بيضل تقريبًا "
                "5-10% داخل نطاق 10^5-10^6 عنصر؟ (متطلب تيمون بالضبط).",
                "",
                "**الكلفة المتوقعة**: 12 حل (10 بالسلّم + مرجعين)، أكبرهم "
                "~2.9 مليون عنصر -- أكبر بكتير من أكبر شبكة حُلّت لـB3 "
                "(424 ألف)، فمتوقع ياخد وقت أطول بكتير (ممكن ساعات على "
                "GPU عادي) -- إذا صار في مشكلة ذاكرة أو وقت، أول شي نجرب "
                "تصغير NEW_FINE_RESOLUTION بالخلية.",
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
    out_path = f"{HERE}/B8_Prototype_GPU_MeshConvergence.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
