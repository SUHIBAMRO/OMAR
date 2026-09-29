"""Builds ONE notebook: local integration refinement for B3's region-
stress QoI, per Timon's explicit request (2026-09-28) -- no retraining,
no mesh change, only a finer quadrature within the already-trained
checkpoint's own coarse mesh. See cell_b3_region_local_refine.py for
the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_b3_region_local_refine.py"


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
            "colab": {"provenance": [], "name": "B3_Region_Local_Refine.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B3: تكامل محلي أدق حول الأخدود -- بلا إعادة تدريب",
                "",
                "**طلب تيمون (2026-09-28)**: قبل أي تدريب على شبكة أنعم، "
                "اختبروا تكرير (refine) شبكة **التكامل** محليًا حول "
                "الأخدود بس، **بشكل مستقل عن دقة الـoperator نفسه** -- "
                "بما إنه حالة 6,840 عنصر عندها بس 6 نقاط Gauss بالمنطقة، "
                "يبدأ بفحص تكرير التكامل a priori هناك.",
                "",
                "**كيف اشتغل**: `checkpoint_50000.pt` **بدون أي تغيير** "
                "-- نفس الشبكة (6,840 عنصر) يلي اتدربت وانقيّمت عليها "
                "دايمًا. torch-fem نفسه بيسمح تستعلم دوال الشكل "
                "(shape functions) عند أي نقاط محلية اخترتها، مش بس نقاط "
                "التكامل الافتراضية (8 نقاط/عنصر) -- فبنستخدم شبكة "
                "Gauss-Legendre أدق بكتير (n_sub³ نقطة/عنصر بدل 8) "
                "لإعادة أخذ عينات من **نفس** حقل الإزاحة الموجود أصلاً "
                "(الحقيقي FEM، وتنبؤ الشبكة على نفس الشبكة الخشنة) -- "
                "**بلا أي حل FEM جديد وبلا أي استعلام جديد للشبكة**. عند "
                "n_sub=10 هاد بيعطي **566 نقطة** بمنطقة الأخدود عند دقة "
                "الإنتاج، بدل 6 بس.",
                "",
                "**التقرير يغطي كل طلبات تيمون بالتفصيل**:",
                "- عدد نقاط المنطقة بعد التكرير (n_region_fine)",
                "- **المكوّنات الستة المستقلة للإجهاد منفصلة** (σxx، σyy، "
                "σzz، σxy، σyz، σxz) -- كل وحدة بمقدارها النموذجي (RMS) "
                "وخطأها النسبي الخاص، عشان أي مكوّن صغير ما يضيع/يضخّم "
                "بمقياس مجمّع",
                "- **خطأ Frobenius-norm واحد موحّد لكل مجموعة البيانات** "
                "(مش لكل عينة لحالها) -- عشان عينة وحدة بقيمة إجهاد صغيرة "
                "ما تقدر تضخّم الرقم",
                "",
                "**اتفحص محليًا على CPU بثلاث خطوات حقيقية قبل أي تشغيلة "
                "GPU**: (1) فحص حجم العنصر -- التكامل الأدق لازم يرجّع "
                "نفس حجم العنصر يلي التكامل الخشن بيرجعه، بدقة ~1e-15؛ "
                "(2) فحص تطابق -- عند n_sub=2 (نفس رتبة التكامل الخشن)، "
                "هاد الكود المستقل الجديد لازم يطابق الكود الموجود "
                "أصلاً المتحقق منه، وطابقه فعلاً بدقة ~1e-13؛ (3) "
                "تشغيلة كاملة بحجم toy بلا أي أخطاء.",
                "",
                "**bug حقيقي اتصلح أثناء الفحص**: أول محاولة استوردت "
                "`GROOVE_DEPTH` من ملف فيه قيمة قديمة خاطئة (0.05 بدل "
                "0.20 الصحيحة) -- اكتُشف مباشرة لما فحص الحجم فشل "
                "(فرق 16-40%)، مش افترضنا الكود صحيح.",
                "",
                "**التكلفة**: هاد **بلا أي تدريب** -- بس إعادة تكامل حقل "
                "موجود أصلاً، فمن المفروض ياخد أقل من دقيقة، مش ساعات.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B3_Region_Local_Refine.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
