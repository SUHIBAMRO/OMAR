"""Builds TWO notebooks: retrain B1 x Mooney-Rivlin and B1 x Arruda-Boyce
each on the SAME wider set of resolutions (21,33,101,201) that fixed
B1 x Neo-Hookean's own N=1401 accuracy (44.65% -> 5.85%) -- see
make_b1_multires_retrain_notebook.py, the original for Neo-Hookean.

WHY THIS EXISTS: a real GPU run (2026-09-14) of the ORIGINAL (N=21,33-only)
Mooney-Rivlin/Arruda-Boyce checkpoints at N=1401 showed the exact same
degradation signature Neo-Hookean had before its own retraining -- best
accuracy near the original training resolutions, degrading sharply toward
N=1401:
  Mooney-Rivlin: best ~6.5% (N=101) -> 39.2% (N=1401)
  Arruda-Boyce:  best ~6.3% (N=41)  -> 45.6% (N=1401)
Omar's own explicit choice, once this pattern was confirmed: extend the
same multi-resolution retraining fix to both materials rather than leave
them at the old, worse numbers.

Identical protocol to the Neo-Hookean notebook in every respect except
the --material flag and output directory -- deliberately not a different
recipe, so the comparison to Neo-Hookean's own result stays apples-to-
apples.
"""
import json
import os

BRANCH = "claude/claude-code-question-d307wp"
HERE = "/home/user/OMAR/Practical_Examples/zeroshot_notebooks"
TRAIN_RESOLUTIONS = "21,33,101,201"

MATERIAL_INFO = {
    "mooney_rivlin": {
        "arabic_name": "Mooney-Rivlin",
        "best_pct": "6.5%", "best_n": "101",
        "worst_pct": "39.2%",
        "notebook_name": "B1_MooneyRivlin_MultiRes_Retrain.ipynb",
    },
    "arruda_boyce": {
        "arabic_name": "Arruda-Boyce",
        "best_pct": "6.3%", "best_n": "41",
        "worst_pct": "45.6%",
        "notebook_name": "B1_ArrudaBoyce_MultiRes_Retrain.ipynb",
    },
}


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def build(material):
    info = MATERIAL_INFO[material]
    out_name = f"zeroshot_B1_{material}_multires"
    old_ckpt_dir = f"zeroshot_B1_{material}"

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": info["notebook_name"]},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            md(f"# إعادة تدريب B1 × {info['arabic_name']} على مدى أحجام أوسع",
               "",
               f"**ليش هاد النوتبوك**: فحص حقيقي على GPU (2026-09-14) للـcheckpoint "
               f"الأصلي (مدرب بس على N=21,33) طلع بنفس مشكلة B1×Neo-Hookean بالضبط: "
               f"أفضل دقة ({info['best_pct']}) عند N={info['best_n']} (قريب من أحجام "
               f"التدريب)، وبتنهار لـ{info['worst_pct']} عند N=1401. نفس الحل المستخدم "
               f"لـNeo-Hookean (توسيع أحجام التدريب لـ{TRAIN_RESOLUTIONS}) -- نفس "
               f"البروتوكول بالضبط، بدون أي فرق، عشان تبقى المقارنة عادلة.",
               "",
               "**الموديل الأصلي ما رح يتلمس** — هاد النوتبوك بيكتب لمجلد",
               f"جديد (`{out_name}`) عالـ Drive، فالـ checkpoint الأصلي يضل",
               "موجود نقارن معه.",
               "",
               "**السرعة**: بيستخدم `--fast_solver 1` (نفس الحل السريع المستخدم "
               "لـNeo-Hookean، تحقق منه لكل المواد الثلاث بما فيهم هاي المادة).",
               "",
               "| خلية | شو بتعمل | الوقت المتوقع |",
               "|---|---|---|",
               "| 1 | تجهيز | دقيقة |",
               f"| 2 | توليد بيانات FEM (4 أحجام: {TRAIN_RESOLUTIONS}) | من نص ساعة",
               "    لساعتين تقريباً (الخلية نفسها بتطبع تقدير حي) |",
               "| 3 | التدريب | ساعة إلى بضع ساعات (نفس Neo-Hookean تقريباً، ~11.6 ساعة) |",
               "| 4 | مقارنة مع الموديل الأصلي على نفس الأحجام | دقايق |",
               "",
               "**كل شي محفوظ على Drive أول بأول** — إذا وقف الاتصال، رجّع",
               "شغّل نفس الخلية وبتكمّل من وين وقفت."),

            md("## خلية 1 — التجهيز"),
            code(
                "from google.colab import drive",
                "drive.mount('/content/drive')",
                "",
                "import os, shutil, sys, subprocess",
                "os.chdir('/content')",
                "if os.path.exists('/content/OMAR'):",
                "    shutil.rmtree('/content/OMAR')",
                f"subprocess.run(['git', 'clone', '-q', '-b', '{BRANCH}',",
                "                'https://github.com/SUHIBAMRO/OMAR.git', '/content/OMAR'], check=True)",
                "",
                "WORK = '/content/OMAR/Practical_Examples'",
                "os.chdir(WORK); sys.path.insert(0, WORK)",
                "assert os.path.isdir(os.path.join(WORK, 'omar_pfem')), 'clone فشل'",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
                "                'einops', 'timm', 'h5py', 'jax', 'tqdm'], check=True)",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torch-sla'], check=True)",
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
                "                'nvmath-python[cu12]==0.9.0'], check=True)",
                "",
                "import torch",
                "print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())",
                "assert torch.cuda.is_available(), 'لازم GPU: Runtime > Change runtime type'",
                "print('GPU:', torch.cuda.get_device_name(0))",
                "",
                "from torch_sla.backends import is_cudss_available",
                "assert is_cudss_available(), 'cuDSS مش متوفر -- الحل السريع محتاجه'",
                "print('cuDSS متوفر -- الحل السريع جاهز')",
                "",
                f"OUT = '/content/drive/MyDrive/pfem_run/{out_name}'",
                "os.makedirs(OUT, exist_ok=True)",
                "print('OUT =', OUT)",
                "for f in sorted(os.listdir(OUT)):",
                "    print('  موجود:', f)",
            ),

            md("## خلية 2 — توليد بيانات FEM (بالحل السريع)",
               "",
               f"بتولّد 400 عينة تدريب + 100 تحقّق لكل حجم من الأربعة "
               f"({TRAIN_RESOLUTIONS}).",
               "",
               "**بتحفظ على Drive كل 25 عينة** — إذا وقف النوتبوك، رجّع شغّل",
               "نفس الخلية وبتكمّل من وين وقفت."),
            code(
                "!python -m omar_pfem.resolution_invariance_zeroshot train \\",
                f"    --geometry B1 --material {material} \\",
                f"    --train_resolutions {TRAIN_RESOLUTIONS} \\",
                "    --n_train_per_res 400 --n_val_per_res 100 \\",
                "    --fast_solver 1 \\",
                "    --gen_chunk 25 --stop_after_generation \\",
                "    --out_dir \"$OUT\"",
            ),

            md("## خلية 3 — التدريب",
               "",
               "نفس بروتوكول Neo-Hookean بالضبط (2000 epoch كحد أقصى، "
               "early stopping بصبر 8، batch_size=8، lr=2e-3) — بس على",
               "أربع أحجام بدل حجمين."),
            code(
                "!python -m omar_pfem.resolution_invariance_zeroshot train \\",
                f"    --geometry B1 --material {material} \\",
                f"    --train_resolutions {TRAIN_RESOLUTIONS} \\",
                "    --n_train_per_res 400 --n_val_per_res 100 \\",
                "    --fast_solver 1 \\",
                "    --epochs 2000 --validate_every 25 --batch_size 8 \\",
                "    --early_stop_patience 8 --lr 2e-3 \\",
                "    --out_dir \"$OUT\"",
            ),

            md("## خلية 4 — مقارنة مباشرة: الموديل الجديد مقابل الأصلي",
               "",
               "بتشغّل نفس فحص الدقة الصارم (نفس المرجع الحقيقي المحلول لكل N،",
               "نفس الـ QoIs) على **الموديلين الاثنين** عند نفس الأحجام",
               "(13 لغاية 1401)، وتطبع جدول جنب بعض."),
            code(
                "import json, os, sys",
                "sys.path.insert(0, '/content/OMAR/Practical_Examples')",
                "sys.path.insert(0, '/content/OMAR/Practical_Examples/report_builders')",
                "for m in list(sys.modules):",
                "    if m == 'omar_pfem' or m.startswith('omar_pfem.'):",
                "        del sys.modules[m]",
                "",
                "import torch, argparse",
                "from omar_pfem.measure_inference_latency import build_model",
                "from omar_pfem.no_accuracy_at_n1401 import run_accuracy_degradation_sweep",
                "from omar_pfem.resolve_b1_checkpoint import sha256_of",
                "",
                "device = torch.device('cuda')",
                "args = argparse.Namespace(",
                "    model='Transolver_Irregular_Mesh', n_hidden=256, n_layers=4, n_heads=8,",
                "    mlp_ratio=2, dropout=0.1, unified_pos=0, ref=16, slice_num=128, fun_dim=4,",
                "    use_soft_dirichlet=1, Lx=1.0, Ly=1.0, R_out=2.0,",
                ")",
                f"MATERIAL = '{material}'",
                "",
                "R = '/content/drive/MyDrive/pfem_run'",
                "NEW_CKPT = os.path.join(OUT, 'model_best.pt')",
                f"OLD_CKPT = f'{{R}}/{old_ckpt_dir}/model_best.pt'",
                "assert os.path.exists(NEW_CKPT), f'new checkpoint not found: {NEW_CKPT}'",
                "assert os.path.exists(OLD_CKPT), f'old checkpoint not found: {OLD_CKPT}'",
                "",
                "RESOLUTIONS = [13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 101, 201, 401, 701, 1001, 1401]",
                "",
                "def run_one(ckpt, out_json):",
                "    model = build_model(args, device).to(torch.float32)",
                "    model.load_state_dict(torch.load(ckpt, map_location=device))",
                "    fp = sha256_of(ckpt)",
                "    return run_accuracy_degradation_sweep(model, args, RESOLUTIONS, out_json, device,",
                "                                           material=MATERIAL, checkpoint_fingerprint=fp)",
                "",
                "print('=== OLD (original 21,33-only) checkpoint ===')",
                f"old_rows = run_one(OLD_CKPT, f'{{R}}/no_accuracy_degradation_sweep_{material}.json')",
                "",
                "print('\\n=== NEW (multi-resolution) checkpoint ===')",
                f"new_rows = run_one(NEW_CKPT, f'{{OUT}}/no_accuracy_degradation_sweep_{material}_multires.json')",
                "",
                "old_by_n = {r['N']: r for r in old_rows}",
                "new_by_n = {r['N']: r for r in new_rows}",
                "print('\\n' + '=' * 70)",
                "print(f\"{'N':<8}{'OLD disp_rel_L2':<18}{'NEW disp_rel_L2':<18}{'better?':<10}\")",
                "for N in RESOLUTIONS:",
                "    o, n = old_by_n.get(N), new_by_n.get(N)",
                "    if o and n:",
                "        oe, ne = o['fp32']['disp_rel_L2'], n['fp32']['disp_rel_L2']",
                "        print(f\"{N:<8}{oe:<18.4e}{ne:<18.4e}{'YES' if ne < oe else 'no':<10}\")",
            ),
        ],
    }


if __name__ == "__main__":
    for material in MATERIAL_INFO:
        nb = build(material)
        path = os.path.join(HERE, MATERIAL_INFO[material]["notebook_name"])
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", path)
