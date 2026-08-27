"""Generates one self-contained Colab notebook per remaining zero-shot case.

One notebook per case (rather than one notebook looping over cases) so the
five can run in parallel on separate Colab runtimes, and so a crash in one
case costs only that case.

Every notebook is split into cells that mirror the real cost structure:
generation (hours) is its own cell, training (minutes) is its own cell,
evaluation is its own cell. Re-running any cell after a disconnect resumes
from what is already on Drive instead of starting over.
"""
import json
import os

CASES = [
    ("B1", "mooney_rivlin"),
    ("B1", "arruda_boyce"),
    ("B2", "neo_hookean"),
    ("B2", "mooney_rivlin"),
    ("B2", "arruda_boyce"),
]

BRANCH = "claude/claude-code-question-d307wp"
OUT_DIR = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def build(geom, mat):
    case = f"{geom}_{mat}"
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": f"zeroshot_{case}.ipynb"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(f"# Zero-shot resolution invariance — **{geom} × {mat}**",
               "",
               "نوتبوك مستقل لحالة وحدة بس. اشتغل عادي بالتوازي مع باقي الحالات،",
               "كل وحدة على Colab runtime لحالها.",
               "",
               "**أهم إشي: كل إشي بينحفظ على Drive أول بأول.**",
               "إذا فصل الاتصال أو وقف النوتبوك — رجّع شغّل نفس الخلية،",
               "بتكمّل من وين وقفت، ما بترجع من الصفر.",
               "",
               "| خلية | شو بتعمل | الوقت المتوقع |",
               "|---|---|---|",
               "| 1 | تجهيز (Drive + clone) | دقيقة |",
               "| 2 | توليد بيانات FEM — **هاي الغالية** | ساعات (7+ لكل دقة) |",
               "| 3 | التدريب على البيانات المولّدة | دقايق |",
               "| 4 | التقييم zero-shot على دقات ما تدرّب عليها | ساعة تقريبًا |",
               "| 5 | عرض النتائج النهائية | ثواني |",
               "",
               "كل خلية بتكتب سجل JSON فيه التاريخ والوقت والأوامر والنتائج في",
               "`run_manifest.json` جوّا مجلد الحالة على Drive — عشان لما نيجي نكتب",
               "النتائج ما نقعد ندوّر من وين إجى كل رقم."),

            md("## خلية 1 — التجهيز"),
            code(
                "from google.colab import drive",
                "drive.mount('/content/drive')",
                "",
                "import os, shutil, sys",
                "os.chdir('/content')",
                "if os.path.exists('/content/OMAR'):",
                "    shutil.rmtree('/content/OMAR')",
                f"!git clone -q -b {BRANCH} https://github.com/suhibamro/omar.git /content/OMAR",
                "",
                "WORK = '/content/OMAR/Practical_Examples'",
                "os.chdir(WORK); sys.path.insert(0, WORK)",
                "assert os.path.isdir(os.path.join(WORK, 'omar_pfem')), 'clone فشل'",
                "!{sys.executable} -m pip install -q einops timm h5py jax tqdm",
                "",
                f"GEOM, MAT = '{geom}', '{mat}'",
                f"CASE = '{case}'",
                "# مجلد الحالة على Drive — كل المخرجات والكاش بتروح هون، مش على",
                "# قرص Colab المؤقت، عشان تضل موجودة بعد ما ينفصل الاتصال.",
                "OUT  = f'/content/drive/MyDrive/pfem_run/zeroshot_{CASE}'",
                "os.makedirs(OUT, exist_ok=True)",
                "",
                "import torch",
                "print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())",
                "if not torch.cuda.is_available():",
                "    print('WARNING: فعّل GPU من Runtime > Change runtime type')",
                "print('OUT =', OUT)",
                "",
                "# شو موجود من قبل (لو هاي مش أول مرة تشغّل)",
                "for f in sorted(os.listdir(OUT)):",
                "    print('  موجود:', f, os.path.getsize(os.path.join(OUT, f)) // 1024, 'KB')",
                "",
                "# النوتبوكات القديمة ممكن تكون حطّت شغلها بمجلد باسم ثاني. لو طلع",
                "# تحت مجلد فيه samples_cache*.pt وإله اسم غير OUT، احكيلي قبل ما",
                "# تكمّل — بنوفّر ساعات توليد بدل ما نعيدها من الصفر.",
                "import glob",
                "others = [d for d in glob.glob('/content/drive/MyDrive/pfem_run/*zeroshot*')",
                "          if os.path.isdir(d) and os.path.abspath(d) != os.path.abspath(OUT)]",
                "if others:",
                "    print('\\nمجلدات zeroshot ثانية على Drive:')",
                "    for d in others:",
                "        cached = glob.glob(f'{d}/samples_cache*.pt')",
                "        print(' ', d, '<-- فيه كاش عينات!' if cached else '')",
            ),

            md("## خلية 2 — توليد بيانات FEM (الخطوة الغالية)",
               "",
               "بتولّد 400 عينة تدريب + 100 تحقّق لكل دقة من الدقتين (21 و33).",
               "هاي حلول FEM حقيقية، مش تدريب — ولهيك هي أطول إشي بالعملية.",
               "",
               "**بتحفظ على Drive كل 25 عينة.** إذا وقف النوتبوك، رجّع شغّل نفس",
               "الخلية: بتطبع `[resume] found X/400 ...` وبتكمّل من هناك.",
               "",
               "الخلية بتوقف بعد التوليد وما بتدرّب — التدريب بالخلية الجاي."),
            code(
                "!python -m omar_pfem.resolution_invariance_zeroshot train \\",
                "    --geometry {GEOM} --material {MAT} \\",
                "    --train_resolutions 21,33 --n_train_per_res 400 --n_val_per_res 100 \\",
                "    --gen_chunk 25 --stop_after_generation \\",
                "    --out_dir \"{OUT}\"",
            ),

            md("## خلية 3 — التدريب",
               "",
               "بيقرا البيانات المولّدة من الخلية السابقة (ما بيعيد توليدها).",
               "نموذج واحد بيتدرّب على الدقتين مع بعض.",
               "",
               "التدريب كمان بيحفظ حالته (النموذج + حالة Adam) كل تحقّق، فلو",
               "انفصل — رجّع شغّل الخلية وبتكمّل من آخر epoch محفوظ."),
            code(
                "!python -m omar_pfem.resolution_invariance_zeroshot train \\",
                "    --geometry {GEOM} --material {MAT} \\",
                "    --train_resolutions 21,33 --n_train_per_res 400 --n_val_per_res 100 \\",
                "    --epochs 2000 --validate_every 25 --batch_size 8 \\",
                "    --out_dir \"{OUT}\"",
            ),

            md("## خلية 4 — التقييم zero-shot",
               "",
               "نفس checkpoint بالظبط، **بدون أي إعادة تدريب**، بينقيّم على دقات",
               "ما شافها بالتدريب — أخشن من 21 وأنعم من 33 (طلب تيمون):",
               "`13, 17, 25, 29, 37, 41, 49`.",
               "",
               "كل الدقات بتنقاس مقابل نفس المرجع الناعم N=101.",
               "المرجع الناعم بينحفظ بالكاش وبينعاد استعماله لكل الدقات، فأول دقة",
               "بس هي الي بتدفع تكلفته."),
            code(
                "!python -m omar_pfem.resolution_invariance_zeroshot eval \\",
                "    --geometry {GEOM} --material {MAT} \\",
                "    --checkpoint \"{OUT}/model_best.pt\" \\",
                "    --test_resolutions 13,17,25,29,37,41,49 \\",
                "    --fine_N 101 --n_eval_samples 20 \\",
                "    --out_json \"{OUT}/zeroshot_eval_report.json\"",
            ),

            md("## خلية 5 — النتائج النهائية",
               "",
               "بتطبع الجدول الجاهز للتقرير + سجل كل التشغيلات بتواريخها.",
               "انسخ المخرجات كلها وابعتها."),
            code(
                "import json, os",
                "",
                "rep = f'{OUT}/zeroshot_eval_report.json'",
                "if os.path.exists(rep):",
                "    r = json.load(open(rep))",
                "    print('=' * 62)",
                "    print(f'ZERO-SHOT — {CASE}')",
                "    print('=' * 62)",
                "    print(f\"{'N':>6}{'mean rel-L2':>16}{'std':>16}\")",
                "    for row in r['rows']:",
                "        print(f\"{row['N']:>6}{row['mean_rel_L2_vs_fine_reference']:>16.4e}\"",
                "              f\"{row['std_rel_L2_vs_fine_reference']:>16.4e}\")",
                "    print('=' * 62)",
                "else:",
                "    print('لسا ما خلص التقييم:', rep)",
                "",
                "man = f'{OUT}/run_manifest.json'",
                "if os.path.exists(man):",
                "    print('\\nسجل التشغيلات (run_manifest.json):')",
                "    for i, rec in enumerate(json.load(open(man)), 1):",
                "        print(f\"  {i}. {rec['kind']:<20} {rec['finished_at_local']}\"",
                "              f\"  ({rec['duration_human']})\")",
                "    print('\\n--- السجل الكامل ---')",
                "    print(json.dumps(json.load(open(man)), indent=1)[:6000])",
            ),
        ],
    }


os.makedirs(OUT_DIR, exist_ok=True)
for geom, mat in CASES:
    path = os.path.join(OUT_DIR, f"zeroshot_{geom}_{mat}.ipynb")
    with open(path, "w") as f:
        json.dump(build(geom, mat), f, indent=1, ensure_ascii=False)
    print("wrote", path)
