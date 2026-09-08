# =====================================================================
#  CELL -- Inventory every image already on Drive under this project's
#  own pfem_run directory (read-only: lists and counts, changes nothing).
#
#  WHY: a manual Drive-search-API inventory (one paginated call at a
#  time) turned out far too slow to cover everything -- one single
#  subfolder alone (B1_neo_hookean_inputnorm) already had 75+ candidate
#  images. Walking the mounted Drive directly with Python's own
#  filesystem calls is orders of magnitude faster and gets everything
#  in one pass, since Drive is mounted as a normal filesystem in Colab.
#
#  WHAT THIS PRINTS:
#    1. One row per folder that contains at least one image: folder
#       path (relative to pfem_run), image count, total size, and the
#       earliest/latest modification time in that folder (a rough signal
#       for "one final figure" vs. "many candidate snapshots" -- a
#       folder with images spread across many distinct timestamps a
#       few seconds apart is very likely a candidate-selection sweep,
#       like the epoch/sample sweeps already found manually).
#    2. A grand total (folder count, image count, total size).
#    3. Everything also written to a JSON file on Drive
#       (pfem_run/image_inventory.json) so it can be reviewed offline
#       without re-running this cell.
# =====================================================================
import os, json, time
from collections import defaultdict

from google.colab import drive
drive.mount('/content/drive')

ROOT = '/content/drive/MyDrive/pfem_run'
OUT_JSON = os.path.join(ROOT, 'image_inventory.json')
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.svg'}

folders = defaultdict(lambda: {'count': 0, 'size': 0, 'mtimes': []})
total_images = 0
total_size = 0

for dirpath, dirnames, filenames in os.walk(ROOT):
    rel = os.path.relpath(dirpath, ROOT)
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        full = os.path.join(dirpath, fn)
        try:
            st = os.stat(full)
        except OSError:
            continue
        folders[rel]['count'] += 1
        folders[rel]['size'] += st.st_size
        folders[rel]['mtimes'].append(st.st_mtime)
        total_images += 1
        total_size += st.st_size

rows = []
for rel, info in sorted(folders.items(), key=lambda kv: -kv[1]['count']):
    mtimes = info['mtimes']
    span_s = (max(mtimes) - min(mtimes)) if len(mtimes) > 1 else 0.0
    rows.append({
        'folder': rel, 'image_count': info['count'],
        'total_size_mb': round(info['size'] / 1e6, 1),
        'earliest': time.strftime('%Y-%m-%d %H:%M', time.localtime(min(mtimes))),
        'latest': time.strftime('%Y-%m-%d %H:%M', time.localtime(max(mtimes))),
        'span_minutes': round(span_s / 60, 1),
        'likely_kind': ('candidate-sweep (many, close together)'
                         if info['count'] >= 10 and span_s < 3600 * 6
                         else ('single/final' if info['count'] <= 3 else 'mixed/unclear')),
    })

print(f"{'Folder':<55}{'#images':>8}{'MB':>8}{'earliest':>18}{'latest':>18}  kind")
print('-' * 130)
for r in rows:
    print(f"{r['folder']:<55}{r['image_count']:>8}{r['total_size_mb']:>8.1f}"
          f"{r['earliest']:>18}{r['latest']:>18}  {r['likely_kind']}")

print('-' * 130)
print(f"TOTAL: {len(rows)} folders, {total_images} images, {total_size/1e6:.1f} MB")

with open(OUT_JSON, 'w') as f:
    json.dump({'root': ROOT, 'total_folders': len(rows), 'total_images': total_images,
                'total_size_mb': round(total_size / 1e6, 1), 'folders': rows}, f, indent=2)
print(f"\nFull inventory written to {OUT_JSON}")
