"""Builds ONE notebook: a medium-scale test of B8-final's new rigid-shim
model (each internal shim an exact rigid body, motion determined by
equilibrium) before any full 10^5-10^6-element production run. See
cell_rigid_shim_medium_test.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_rigid_shim_medium_test.py"


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
            "colab": {"provenance": [], "name": "B8_RigidShim_MediumTest.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B8-final: اختبار النموذج الجديد (rigid-shim) بحجم متوسط",
                "",
                "**السياق**: نموذج الفولاذ المرن فشل عند 105,456 عنصر مع "
                "أقوى وأضعف حلّال متوفر (Jacobi وAmgX/AMG الحقيقي على "
                "GPU). بنينا نموذج جديد كليًا: كل شيم (19 شيم) بيتمثل "
                "كجسم صلب تمامًا (حركته -- إزاحة ودوران -- مجهولة "
                "وبتتحدد من التوازن، مش مفروضة). تحقق منه عند 15,600 "
                "عنصر ونجح: خطأ الإزاحة 2.09%، خطأ إجهاد Cauchy "
                "(المقياس الأهم) 1.63%، فرق القوة المرتدة 1.36% -- كلها "
                "أقل بكتير من نطاق 5-10% المقبول. **مكسب إضافي**: أسرع "
                "6 مرات من النموذج القديم بنفس الحجم (49 ثانية مقابل "
                "293 ثانية).",
                "",
                "**ليش هاي الخلية، وليش مش رايحين مباشرة للحجم الكبير**: "
                "درسين حقيقيين من نفس اليوم (خلية B3/الأخدود قفزت من "
                "480 ألف مباشرة لـ1.07 مليون عنصر وعلّقت 10 ساعات):",
                "1. النظام المختزل بالنموذج الجديد بيتحل بطريقة \"حل "
                "مباشر\" (direct solve، عبر SciPy) -- دقيق وقوي، بس "
                "معروف عنه إنه ممكن يبطّئ كتير أو ياكل ذاكرة كبيرة كل ما "
                "كبرت المسألة. ما جُرّب أبدًا فوق 15,600 عنصر.",
                "2. تحذير صريح من نفس المراجعة يلي طلبت هاد النموذج: "
                "المطاط قريب من عدم الانضغاطية (near-incompressible)، "
                "وعناصر HEX8 المعتمدة على الإزاحة ممكن يكون فيها مشكلة "
                "\"locking\" خاصة فيها، مستقلة تمامًا عن مشكلة صلابة "
                "الفولاذ يلي حليناها. إذا صارت مشكلة هون، هاد يلي لازم "
                "يتفحص بعده -- مش سبب نرجع نغيّر خواص المواد الحقيقية.",
                "",
                "**السلّم**: (37,19)=50,544 / (45,23)=75,504 / "
                "(53,27)=105,456 عنصر -- آخر وحدة هي **بالضبط** نفس "
                "الحجم يلي فشل فيه النموذج القديم، عشان مقارنة مباشرة "
                "وذات معنى.",
                "",
                "**لو نجح (53,27) بوقت معقول**: عندنا مرشح حقيقي جاهز "
                "للدراسة الإنتاجية الكاملة (10^5-10^6 عنصر). **لو صارت "
                "مشكلة**: نعرف بالضبط وين نحقق (حل مباشر بطيء، أو "
                "locking)، بدل ما نضيع وقت GPU على تخمين.",
                "",
                "**تحديث حي من نفس التشغيلة**: عند (45,23)=75,504 عنصر، "
                "الحل المباشر (direct solve) صار بطيء جدًا وغير "
                "متناسب (تأكدنا هو شغال فعليًا بـ100% CPU، مش عالق) -- "
                "مشكلة معروفة للحلّالات المباشرة بمسائل 3D (fill-in). "
                "أضفنا خيار حلّال تكراري (`linear_solver='cg'`) وتحققنا "
                "منه بنفسنا محليًا: بينتج **نفس النتيجة تمامًا** (فرق "
                "أصغر من 1e-15) عند 2,496 و15,600 عنصر، وأسرع فعليًا "
                "عند 15,600 (44s مقابل 51s). هاي النسخة بتجرب الاثنين "
                "معًا عند (37,19) لتأكيد حي على GPU، وبعدها بتكمل بـCG "
                "بس للأحجام الكبيرة.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B8_RigidShim_MediumTest.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
