# ============================================================
# خلية الإصلاح — للصقها في نوتبوكات B2 الشغالة (بدون ما توقفها)
# ============================================================
# ضيف خلية جديدة بآخر النوتبوك، الصق هاد فيها، واضغط تشغيل.
# لو التوليد لسا شغال، Colab بيحط الخلية بالطابور وبيشغّلها لحالها
# أول ما يخلص — فما بتخسر ولا ثانية من التوليد.
#
# بتعمل ثلاث إشياء:
#   1. بتجيب الكود المصلَّح للنسخة الموجودة على Colab
#   2. بتصلّح القوة العقدية بالعينات المولّدة (بثواني، بدون إعادة FEM)
#   3. بتمسح أي نموذج اتدرب على القوة الغلط
#
# آمنة لو شغّلتها أكثر من مرة: بتعيد حساب القوة من البذرة كل مرة،
# فالعينات المصلَّحة أصلًا بتضل زي ما هي.
import os
import sys
import glob
import subprocess

WORK = '/content/OMAR/Practical_Examples'
BRANCH = 'claude/claude-code-question-d307wp'

# --- 1. الكود المصلَّح -------------------------------------------------
# النسخة الي على Colab انستنسخت قبل الإصلاح. التوليد الشغال حاليًا حمّل
# الموديولات بالذاكرة من زمان، فتحديث الملفات ما بيأثر عليه.
print('=== 1. تحديث الكود ===')
subprocess.run(['git', '-C', '/content/OMAR', 'fetch', '--quiet', 'origin', BRANCH], check=True)
subprocess.run(['git', '-C', '/content/OMAR', 'reset', '--hard', '--quiet',
                f'origin/{BRANCH}'], check=True)
sha = subprocess.run(['git', '-C', '/content/OMAR', 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
print('الكود صار على:', sha)

os.chdir(WORK)
if WORK not in sys.path:
    sys.path.insert(0, WORK)

# --- 2. إصلاح العينات --------------------------------------------------
print('\n=== 2. إصلاح القوة بالعينات ===')
if 'OUT' not in dir():
    raise SystemExit('متغير OUT مش معرّف — شغّل خلية 1 الأول')

caches = glob.glob(f'{OUT}/samples_cache*.pt')
if not caches:
    print('ما في كاش عينات لحد هلا — الكود الجديد بيولّد صح من نفسه، ما في إشي تصلحه')
else:
    print(f'لقيت {len(caches)} ملف كاش')
    r = subprocess.run([sys.executable, '-m', 'omar_pfem.repair_b2_sample_cache',
                        '--out_dir', OUT], capture_output=True, text=True)
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit('الإصلاح فشل — ابعت الرسالة لكلود قبل ما تكمّل')

# --- 3. مسح النماذج المسمومة ------------------------------------------
# التدريب بيكمّل من train_state_latest.pt، فلو تركناه بيرجع يبني على
# حالة اتدربت على الحِمل الغلط. ملفات التقييم ما بدها مسح — التقييم
# صار يبصم ال-checkpoint وبيرفض صفوف من نموذج تاني لحاله.
print('\n=== 3. مسح النماذج الي اتدربت على الحِمل الغلط ===')
removed = []
for f in ('model_best.pt', 'model_final.pt', 'train_state_latest.pt',
          'metrics_history.json', 'EARLY_STOPPED'):
    path = os.path.join(OUT, f)
    if os.path.exists(path):
        os.remove(path)
        removed.append(f)
print('انمسح:', ', '.join(removed) if removed else 'ما في إشي — النموذج لسا ما اتدرب')

# --- خلصنا ------------------------------------------------------------
has_cache = bool(glob.glob(f'{OUT}/samples_cache*.pt'))
print('\n' + '=' * 58)
print('تم الإصلاح.')
if has_cache:
    print('الخطوة الجاي: شغّل خلية التدريب (خلية 3).')
    print('لو التوليد لسا ما خلص الدقتين، شغّل خلية 2 الأول — بتكمّل')
    print('من وين وقفت وبتولّد الباقي بالقوة الصحيحة.')
else:
    print('الخطوة الجاي: شغّل خلية 2 (التوليد) عادي.')
print('=' * 58)
