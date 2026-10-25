"""Dataset deteksi Roboflow (YOLO) -> dataset_raw/<kelas>/*.jpg + metadata.csv
Split 70/15/15 per GAMBAR SUMBER (anti-leakage)."""
import collections, csv, random, re
from pathlib import Path
import yaml
from PIL import Image

SRC, OUT = Path("/content/roboflow_export"), Path("dataset_raw")
MARGIN, MIN_SIDE, SEED, RATIO = 0.15, 24, 42, (0.70, 0.15, 0.15)
USE_LATAR = True     # kelas tambahan "latar" (area tanpa buah)

random.seed(SEED)
names = yaml.safe_load(open(SRC/"data.yaml"))["names"]
names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
key = lambda p: re.split(r"\.rf\.", p.stem)[0]

items = []
for sp in ["train", "valid", "test"]:
    d = SRC/sp/"images"
    if d.exists():
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((p, SRC/sp/"labels"/(p.stem + ".txt")))

groups = sorted({key(p) for p, _ in items}); random.shuffle(groups)
n = len(groups); n_tr, n_va = int(RATIO[0]*n), int(RATIO[1]*n)
split_of = {g: ("train" if i < n_tr else "valid" if i < n_tr + n_va else "test") for i, g in enumerate(groups)}

def overlap(c, b):  # bagian crop c yang tertutup kotak b
    ix = max(0, min(c[2], b[2]) - max(c[0], b[0])); iy = max(0, min(c[3], b[3]) - max(c[1], b[1]))
    return ix*iy / ((c[2]-c[0]) * (c[3]-c[1]))

rows, seen = [], set()
for img_path, lbl in items:
    g = key(img_path)
    if g in seen: continue
    seen.add(g); sp = split_of[g]
    img = Image.open(img_path).convert("RGB"); W, H = img.size
    boxes = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            t = line.split()
            if len(t) < 5: continue
            c = int(t[0]); xc, yc, w, h = (float(v) for v in t[1:5])
            bw, bh = w*W*(1+MARGIN), h*H*(1+MARGIN)
            boxes.append((names[c], (max(0, xc*W-bw/2), max(0, yc*H-bh/2), min(W, xc*W+bw/2), min(H, yc*H+bh/2))))
    for i, (kelas, b) in enumerate(boxes):
        if b[2]-b[0] < MIN_SIDE or b[3]-b[1] < MIN_SIDE: continue
        (OUT/kelas).mkdir(parents=True, exist_ok=True)
        fn = f"{g}_{i}.jpg"
        img.crop(tuple(int(v) for v in b)).save(OUT/kelas/fn, quality=95)
        rows.append([fn, kelas, sp, g, *[int(v) for v in b]])
    if USE_LATAR:
        for _ in range(30):
            cw, ch = random.uniform(.2, .4)*W, random.uniform(.2, .4)*H
            if cw < MIN_SIDE or ch < MIN_SIDE: continue
            x1, y1 = random.uniform(0, W-cw), random.uniform(0, H-ch)
            c = (x1, y1, x1+cw, y1+ch)
            if all(overlap(c, b) < 0.05 for _, b in boxes):
                (OUT/"latar").mkdir(parents=True, exist_ok=True)
                fn = f"{g}_latar.jpg"
                img.crop(tuple(int(v) for v in c)).save(OUT/"latar"/fn, quality=95)
                rows.append([fn, "latar", sp, g, *[int(v) for v in c]]); break

with open(OUT/"metadata.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["nama_file", "kelas", "split", "gambar_sumber", "x1", "y1", "x2", "y2"]); w.writerows(rows)

tab = collections.Counter((r[1], r[2]) for r in rows)
print(f"{'kelas':10s} train valid test total")
for k in sorted({r[1] for r in rows}):
    t = [tab[(k, s)] for s in ("train", "valid", "test")]
    print(f"{k:10s} {t[0]:5d} {t[1]:5d} {t[2]:4d} {sum(t):5d}")
