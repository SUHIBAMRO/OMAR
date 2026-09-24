"""Builds ONE notebook: the full production GPU mesh-convergence study
for Option B (B8-final's rigid-shim model), finding its own real
region-Cauchy-field-error-vs-resolution crossing into the advisor's
5-10% target band, inside the 10^5-10^6-element range. See
cell_rigid_shim_gpu_mesh_convergence.py for the full rationale.
"""
import json

HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
CELL_FILE = f"{HERE}/cell_rigid_shim_gpu_mesh_convergence.py"


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
            "colab": {"provenance": [], "name": "B8_RigidShim_GPU_MeshConvergence.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(
                "# B8-final (rigid-shim): دراسة التقارب الإنتاجية الكاملة",
                "",
                "**السياق**: النموذج الجديد (rigid-shim -- كل شيم جسم صلب "
                "تمامًا، حركته تتحدد من التوازن) اجتاز اختبار الجدوى "
                "اليوم: نجح بشكل نظيف عند (37,19)=50,544 و(45,23)=75,504 "
                "و**(53,27)=105,456 عنصر -- بالضبط نفس الحجم يلي فشل فيه "
                "النموذج القديم** (الفولاذ المرن)، بفضل إصلاحين حقيقيين: "
                "تصغير خطوة التحميل تلقائيًا عند الفشل (cutback)، وكشف "
                "تباعد نيوتن مبكرًا (يوفر وقت GPU ويمنع فيضان السجل).",
                "",
                "**هاي الخلية مختلفة عن اختبار الجدوى**: هلأ لازم نلاقي "
                "**رقم التقاطع الحقيقي** لهاد النموذج بالذات مع نطاق "
                "5-10% المطلوب من المشرف -- **رقم \"791,864 عنصر\" "
                "القديم يلي كنا حاطينه لـB8 كان محسوب على النموذج القديم "
                "(الفولاذ المرن)**، من قبل ما نكتشف إنه بيفشل بالكبير. "
                "بما إنه التمثيل الفيزيائي اختلف (شيمات صلبة تمامًا "
                "بدل فولاذ مرن)، ما بنفترض نفس رقم التقاطع -- لازم يتأكد "
                "من جديد، رغم إنه النموذجين كانوا متقاربين جدًا (~2%) عند "
                "15,600 عنصر.",
                "",
                "**تصميم السلّم**: (21,11)=15,600 (نفس نقطة التحقق "
                "الأصلية، معاد حلّها هون على نفس المنهجية)، ثم "
                "(37,19)/(45,23)/(53,27) (الثلاثة يلي أثبتوا نجاحهم "
                "اليوم على GPU حقيقي)، وبعدين خطوات نمو معتدلة (1.3-1.7 "
                "ضعف تقريبًا لكل خطوة) لحد ~1,036,000 عنصر -- أكتر جرأة "
                "شوي من سلّم B3 (Option A) قرب الحافة (1.05 ضعف)، بما "
                "إنه آلية الـcutback + كشف التباعد المبكر أثبتت اليوم "
                "إنها بتتعامل صح حتى مع قفزة بقايا (residual) بمقدار 35 "
                "ضعف بخطوة نيوتن وحدة عند 105,456 عنصر.",
                "",
                "**تحذير حقيقي وصريح، مش افتراض صامت**: (53,27)=105,456 "
                "عنصر أخذ 728.24 ثانية على A100 حقيقي اليوم. أعلى صف "
                "بهاد السلّم أكبر بحوالي 10 أضعاف من هيك -- الوقت "
                "لكل صف متوقع يزيد بشكل كبير (عدد دورات CG وتكلفة كل "
                "دورة بيزيدوا الاثنين مع حجم المسألة)، فهاي الخلية ممكن "
                "تاخد كم ساعة إجمالًا بشكل واقعي. كل صف بيطبع نتيجته "
                "الحقيقية **فورًا** لما يخلص (مش بس بالنهاية)، فحتى لو "
                "صار انقطاع بمنتصف التشغيلة، بيضل عنا نتائج حقيقية "
                "ومفيدة بمخرجات الخلية.",
                "",
                "**المقياس الأساسي**: خطأ حقل Cauchy المنطقي (region-"
                "Cauchy field error) -- نفس المنهجية المستخدمة لـOption "
                "A بالضبط، بما فيها التحقق من المرجع مقابل مرجع أدق "
                "(الصفّين الأكبر بالسلّم نفسه، بدون حل منفصل إضافي -- "
                "الدرس الحقيقي من حادثة B3 يلي ضاع فيها ~12.5 ساعة GPU).",
                "",
                "**لما تخلص**: بنعرف بالضبط وين بيدخل rigid-shim ضمن "
                "نطاق 5-10%، وعندها الاثنين (Option A وOption B) بيصيروا "
                "جاهزين للعرض سوا للبروفيسور رابتشوك.",
            ),
            code(*cell_src.splitlines()),
        ],
    }


if __name__ == "__main__":
    nb = build()
    out_path = f"{HERE}/B8_RigidShim_GPU_MeshConvergence.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out_path)
