# ============================================================
# جمع المدخلات الحقيقية لنقطة 3 (التسريع ونقطة التعادل)
# ============================================================
# شغّله بـRound5 بعد خلية 1.
#
# بيعمل إشيين:
#   1. بيحسب التسريع بأحجام دفعات متطابقة — هاد ما بده زمن تدريب،
#      فبيطلع رقم نهائي دقيق لكل حالة عندها الملفين.
#   2. بيطبع محتوى ملفات train.log عشان نطلع منها زمن التدريب
#      الحقيقي لكل حالة — بدونه ما في نقطة تعادل، وما بدنا نخمّنها.
import os
import glob
import json
import subprocess
import sys

R = '/content/drive/MyDrive/pfem_run'
WORK = '/content/OMAR/Practical_Examples'
os.chdir(WORK)

print('=' * 74)
print('1. ملفات توقيت GPU FEM')
print('=' * 74)
fem_files = sorted(glob.glob(f'{R}/**/*timing*.json', recursive=True))
for f in fem_files:
    try:
        d = json.load(open(f))
        rows = d.get('rows', [])
        tag = f"{d.get('geometry')}_{d.get('material')}"
        per = ', '.join(f"bs{r['batch_size']}={r['per_sample_ms']:.1f}" for r in rows)
        print(f"  {tag:22s} N={d.get('N')}  {per}")
        print(f"  {'':22s} {f}")
    except Exception as e:
        print(f"  [تعذّر قراءته] {f}: {e}")
if not fem_files:
    print('  ما في ولا ملف')

print()
print('=' * 74)
print('2. ملفات زمن استدلال Transolver')
print('=' * 74)
nn_files = sorted(glob.glob(f'{R}/inference_latency_by_batch_*.json'))
for f in nn_files:
    d = json.load(open(f))
    per = ', '.join(f"bs{r['batch_size']}={r['per_sample_ms']:.4f}" for r in d['rows'])
    print(f"  {d['geometry']}_{d['material']:15s} {per}")
if not nn_files:
    print('  ما في ولا ملف — شغّل خلية نقطة 4 الأول')

print()
print('=' * 74)
print('3. التسريع بأحجام دفعات متطابقة (رقم نهائي — ما بده زمن تدريب)')
print('=' * 74)
# Pair the two benchmarks by case. Speed-up needs only these two files, so
# it is exact right now; break-even is deliberately left out until the
# real training cost is known rather than assumed.
by_case_nn = {}
for f in nn_files:
    d = json.load(open(f))
    by_case_nn[f"{d['geometry']}_{d['material']}"] = f
by_case_fem = {}
for f in fem_files:
    try:
        d = json.load(open(f))
        c = f"{d.get('geometry')}_{d.get('material')}"
        if c in by_case_fem:
            print(f"  [انتبه] أكثر من ملف توقيت لـ{c}: {by_case_fem[c]}  و  {f}")
        by_case_fem[c] = f
    except Exception:
        pass

paired = sorted(set(by_case_nn) & set(by_case_fem))
print(f"  حالات عندها الملفين: {paired if paired else 'ولا وحدة'}")
for c in paired:
    print(f"\n----- {c} -----")
    r = subprocess.run([sys.executable, '-m', 'omar_pfem.break_even_analysis',
                        '--fem_json', by_case_fem[c], '--nn_json', by_case_nn[c],
                        '--out_json', f'{R}/break_even_{c}.json'],
                       capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print(r.stderr[-1500:])

print()
print('=' * 74)
print('4. زمن التدريب — من train.log (هاد الي ناقصنا)')
print('=' * 74)
logs = sorted(glob.glob(f'{R}/**/train.log', recursive=True))
print(f'{len(logs)} ملف train.log\n')
for f in logs:
    print(f'----- {f} -----')
    try:
        lines = [l.rstrip() for l in open(f, errors='replace').read().split('\n') if l.strip()]
        # whatever line carries a total/elapsed figure is what we need; print
        # the candidates plus the tail, since the format is not known yet
        hits = [l for l in lines
                if any(k in l.lower() for k in
                       ('total', 'elapsed', 'wall', 'duration', 'finished', 'time'))]
        for l in hits[-6:]:
            print('   [وقت؟]', l[:170])
        print('   [آخر 4 أسطر]')
        for l in lines[-4:]:
            print('     ', l[:170])
    except Exception as e:
        print('   تعذّر:', e)
    print()

print('=' * 74)
print('ابعت كل المخرجات لكلود.')
print('=' * 74)
