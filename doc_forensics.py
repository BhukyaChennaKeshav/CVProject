# %% [markdown]
# # Geometry-Aware Document Forensics & Tampering Detection
# **Lightweight, explainable, GPU-free tamper localisation for structured documents (contracts, receipts, invoices).**
#
# Pipeline (every step is visualised below):
#
# | Phase | Step | What it does |
# |---|---|---|
# | **A - Low-level** | A1 | Grayscale + global character-height estimate |
# | | A2 | Illumination normalisation (morphological background division) |
# | | A3 | **Local scale map** - per-region character height -> scale bins |
# | | A4 | **Scale-adaptive Sauvola binarisation** (window ~ local char height) |
# | | A5 | Deskew via projection-profile sharpness search |
# | | A6 | Sobel gradients, auto-Canny edges |
# | | A7 | Forensic maps: noise residual, block noise-level map, ELA |
# | **B - Mid-level** | B1 | Ruled-line / table-grid extraction (adaptive-length morphology + probabilistic Hough) |
# | | B2 | **Adaptive structuring elements** - word-gap ratio learned per scale bin |
# | | B3 | Word segmentation with the spatially-variant kernel (vs fixed kernel) |
# | | B4 | Horizontal/vertical projection profiles -> text-line bands & segments |
# | | B5 | Per-word baseline + angle (robust bottom-profile fitting) |
# | | B6 | Hough transform on baseline points -> parallelism check |
# | **C - Reasoning** | C1 | Connected-component feature table per word |
# | | C2 | Neighbourhood consistency (robust z-scores) |
# | | C3 | Copy-move detection (background-noise correlation of look-alike words) |
# | | C4 | Fusion -> heat-map, flagged regions, explanations |
# | **Eval** | | Synthetic tampering benchmark with pixel-exact masks + Find-it-again (real forged receipts) |

# %% 0. Setup ---------------------------------------------------------------
import sys, os, io, json, math, glob, time, zipfile, subprocess, warnings, urllib.request
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets'], check=False)

import numpy as np
import cv2
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.stats import theilslopes
from skimage.filters import threshold_sauvola, threshold_otsu
from skimage.morphology import skeletonize
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 100, 'axes.titlesize': 10})
WORK = '/content/docforensics' if IN_COLAB else os.path.abspath('./docforensics_work')
os.makedirs(WORK, exist_ok=True)
print('OpenCV', cv2.__version__, '| Colab:', IN_COLAB, '| work dir:', WORK)


def odd(v, lo=1):
    v = max(lo, int(round(v)))
    return v if v % 2 else v + 1


def show(images, titles=None, cols=3, width=5.5, cmaps=None, suptitle=None, colorbar=False):
    """Grid display; 2-D arrays shown in gray (or given cmap), 3-D assumed RGB."""
    n = len(images)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    aspect = images[0].shape[0] / images[0].shape[1]
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, min(width * aspect, 9) * rows + 0.6), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for i, im in enumerate(images):
        ax = axes.flat[i]
        cm = (cmaps[i] if cmaps else None) or ('gray' if im.ndim == 2 else None)
        art = ax.imshow(im, cmap=cm)
        if colorbar and im.ndim == 2 and cm != 'gray':
            fig.colorbar(art, ax=ax, fraction=0.035)
        if titles:
            ax.set_title(titles[i])
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, weight='bold')
    plt.tight_layout()
    plt.show()


def to_rgb(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB) if g.ndim == 2 else g.copy()


def to_gray(img):
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def jpeg(img, q):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

# %% [markdown]
# ## 1. Data
# **Datasets used**
# 1. **Synthetic structured documents (built-in, always available)** - contracts, receipts and invoices rendered with
#    *mixed font families and sizes* (title / body / fine print, tables, signature rules), passed through a scan simulator
#    (skew, blur, paper texture, illumination gradient, sensor noise, JPEG). Tampering is applied *after* the "scan", like a
#    real fraudster, and yields **pixel-exact ground-truth masks** for 4 attack types: splice-insert, copy-move,
#    baseline shift, whiteout deletion.
# 2. **FUNSD** (noisy scanned forms, HuggingFace `nielsr/funsd`) and **CORD-v2** (receipts, `naver-clova-ix/cord-v2`) -
#    real untampered scans used as clean bases for the same tamper generator (optional, needs internet).
# 3. **Find-it-again** (L3i, ICDAR 2023) - 988 real SROIE receipts, 163 *realistically forged by humans* (copy-paste,
#    imitation, deletion, pixel edits). Used for real-world image-level evaluation (optional ~download).
# 4. Your own image via upload.
#
# Larger benchmark worth knowing: **DocTamper** (CVPR 2023, 170k tampered contracts/invoices/receipts, request-based download).

# %% 1a. Synthetic structured-document renderer -------------------------------
FONT_DIR = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
FONT_FILES = {'sans': 'DejaVuSans.ttf', 'sans_b': 'DejaVuSans-Bold.ttf', 'serif': 'DejaVuSerif.ttf',
              'serif_b': 'DejaVuSerif-Bold.ttf', 'mono': 'DejaVuSansMono.ttf', 'mono_b': 'DejaVuSansMono-Bold.ttf'}
_FONT_CACHE = {}


def font(name, size):
    key = (name, int(size))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(os.path.join(FONT_DIR, FONT_FILES[name]), int(size))
    return _FONT_CACHE[key]


CLAUSES = [
    "The Client shall pay the Service Provider the sum of {amt} within {n} days of receipt of a valid invoice.",
    "This Agreement shall commence on {date} and remain in force for a period of {n} months unless terminated earlier.",
    "Either party may terminate this Agreement by giving not less than {n} days written notice to the other party.",
    "The Service Provider shall maintain the confidentiality of all information disclosed by the Client during the term.",
    "Late payments shall attract interest at the rate of {pct} per annum calculated on a daily basis until settled.",
    "A security deposit of {amt} shall be held by the Client and refunded within {n} days after expiry of the term.",
    "Neither party shall be liable for any failure or delay caused by events beyond its reasonable control.",
    "All disputes arising out of this Agreement shall be referred to arbitration seated at {city} under applicable law.",
    "The total contract value shall not exceed {amt} inclusive of all applicable taxes, duties and levies.",
    "Any amendment to this Agreement must be made in writing and signed by authorised representatives of both parties.",
]
TITLES = ['SERVICE AGREEMENT', 'LEASE AGREEMENT', 'SUPPLY CONTRACT', 'CONSULTING AGREEMENT', 'LOAN AGREEMENT']
CITIES = ['Mumbai', 'Hyderabad', 'Chennai', 'Bengaluru', 'Pune', 'Delhi']
ITEMS = ['Espresso', 'Cafe Latte', 'Masala Tea', 'Veg Sandwich', 'Paneer Wrap', 'Croissant', 'Brownie', 'Cold Coffee',
         'Mineral Water', 'Fruit Bowl', 'Club Sandwich', 'Cheese Toast', 'Muffin', 'Green Tea', 'Pasta Alfredo']
SERVICES = ['Cloud hosting (annual)', 'Software licence', 'Consulting hours', 'Data migration', 'Support plan',
            'Security audit', 'Training session', 'Hardware rental', 'API usage', 'Maintenance fee']


def money(rng, lo=50, hi=99999):
    return f"{rng.uniform(lo, hi):,.2f}"


def rand_date(rng):
    return f"{rng.integers(1, 29):02d}/{rng.integers(1, 13):02d}/20{rng.integers(22, 27)}"


class DocCanvas:
    def __init__(self, W, H):
        self.img = Image.new('L', (W, H), 255)
        self.d = ImageDraw.Draw(self.img)
        self.words, self.rules, self.line_id = [], [], 0
        self.W, self.H = W, H

    def line(self, x, base, text, fname, size, align='left', ink=15):
        f = font(fname, size)
        sp = f.getlength(' ')
        toks = text.split(' ')
        total = sum(f.getlength(t) for t in toks) + sp * (len(toks) - 1)
        if align == 'right':
            x = x - total
        elif align == 'center':
            x = x - total / 2
        for t in toks:
            if t:
                self.d.text((x, base), t, font=f, fill=ink, anchor='ls')
                bb = self.d.textbbox((x, base), t, font=f, anchor='ls')
                self.words.append(dict(text=t, box=list(bb), size=size, font=fname, line=self.line_id))
            x += f.getlength(t) + sp
        self.line_id += 1

    def paragraph(self, x, y, width, text, fname, size, leading=1.4):
        f = font(fname, size)
        cur = ''
        for w in text.split():
            trial = (cur + ' ' + w).strip()
            if f.getlength(trial) > width and cur:
                self.line(x, y + size, cur, fname, size)
                y += size * leading
                cur = w
            else:
                cur = trial
        if cur:
            self.line(x, y + size, cur, fname, size)
            y += size * leading
        return y

    def hline(self, x0, x1, y, w=2):
        self.d.line([(x0, y), (x1, y)], fill=20, width=w)
        self.rules.append((x0, y, x1, y))

    def vline(self, x, y0, y1, w=2):
        self.d.line([(x, y0), (x, y1)], fill=20, width=w)
        self.rules.append((x, y0, x, y1))

    def table(self, x, y, col_w, rows, fname, size, row_h=None):
        row_h = row_h or int(size * 2.0)
        X = [x] + list(x + np.cumsum(col_w))
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                fn = fname + '_b' if r == 0 and not fname.endswith('_b') else fname
                if c == 0:
                    self.line(X[c] + 8, y + r * row_h + int(row_h * 0.68), cell, fn, size)
                else:
                    self.line(X[c + 1] - 8, y + r * row_h + int(row_h * 0.68), cell, fn, size, align='right')
        for r in range(len(rows) + 1):
            self.hline(X[0], X[-1], y + r * row_h)
        for xx in X:
            self.vline(xx, y, y + len(rows) * row_h)
        return y + len(rows) * row_h

    def finish(self):
        g = np.array(self.img)
        for w in self.words:  # tighten boxes to actual ink
            x0, y0, x1, y1 = [int(round(v)) for v in w['box']]
            x0, y0 = max(0, x0 - 1), max(0, y0 - 1)
            crop = g[y0:y1 + 2, x0:x1 + 2] < 128
            if crop.any():
                ys, xs = np.where(crop)
                w['box'] = [x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1]
        return g, self.words, self.rules


def render_contract(rng):
    W, H, m = 1240, 1754, 90
    c = DocCanvas(W, H)
    body = int(rng.integers(19, 27))
    y = m
    c.line(W // 2, y + 40, str(rng.choice(TITLES)), 'serif_b', int(rng.integers(36, 46)), align='center')
    y += 80
    c.line(W // 2, y + 20, f"Agreement No. CN-{rng.integers(10000, 99999)} dated {rand_date(rng)}", 'sans',
           int(rng.integers(16, 21)), align='center')
    y += 70
    for k in range(int(rng.integers(4, 6))):
        c.line(m, y + body + 2, f"{k + 1}. Clause", 'serif_b', body + 2)
        y += (body + 2) * 1.6
        txt = ' '.join(str(rng.choice(CLAUSES)) for _ in range(2)).format(
            amt='INR ' + money(rng, 1000, 900000), n=rng.integers(7, 90), date=rand_date(rng),
            pct=f"{rng.uniform(6, 18):.1f}%", city=rng.choice(CITIES))
        y = c.paragraph(m, y, W - 2 * m, txt, 'serif', body) + body * 0.6
        if y > H * 0.55:
            break
    tsize = max(15, body - 3)
    rows = [['Instalment', 'Due Date', 'Amount (INR)']] + \
           [[f"Milestone {i + 1}", rand_date(rng), money(rng, 5000, 250000)] for i in range(int(rng.integers(3, 5)))]
    y = c.table(m, int(y) + 10, [420, 260, 380], rows, 'sans', tsize) + 40
    y = c.paragraph(m, y, W - 2 * m, ' '.join(str(rng.choice(CLAUSES)) for _ in range(2)).format(
        amt='INR ' + money(rng), n=rng.integers(7, 90), date=rand_date(rng), pct='9.5%', city='Mumbai'),
        'sans', int(rng.integers(11, 14)), leading=1.35) + 50
    for i, who in enumerate(['For the Client', 'For the Service Provider']):
        x0 = m + i * 560
        c.hline(x0, x0 + 400, y + 60)
        c.line(x0, y + 90, who, 'sans', 18)
        c.line(x0, y + 118, f"Date: {rand_date(rng)}", 'sans', 16)
    return c.finish()


def render_receipt(rng):
    W, m = 620, 40
    c = DocCanvas(W, 2200)
    sz = int(rng.integers(18, 23))
    y = 60
    c.line(W // 2, y, str(rng.choice(['CAFE AROMA', 'URBAN BITES', 'BLUE CUP CAFE', 'DAILY GRIND'])), 'mono_b',
           int(rng.integers(30, 38)), align='center')
    y += 40
    for t in [f"Shop {rng.integers(1, 99)}, MG Road, {rng.choice(CITIES)}", f"GSTIN 29ABCDE{rng.integers(1000, 9999)}F1Z5",
              f"Date {rand_date(rng)}  Time {rng.integers(8, 23):02d}:{rng.integers(0, 60):02d}",
              f"Bill No: {rng.integers(10000, 99999)}"]:
        c.line(W // 2, y, t, 'mono', int(rng.integers(14, 18)), align='center')
        y += 26
    c.line(W // 2, y + 10, '-' * 38, 'mono', 16, align='center')
    y += 44
    total = 0
    for _ in range(int(rng.integers(6, 13))):
        q, p = int(rng.integers(1, 4)), float(rng.choice([45, 60, 80, 99, 120, 150, 180, 220, 250]))
        total += q * p
        c.line(m, y, f"{rng.choice(ITEMS)} x{q}", 'mono', sz)
        c.line(W - m, y, f"{q * p:.2f}", 'mono', sz, align='right')
        y += int(sz * 1.5)
    c.line(W // 2, y, '-' * 38, 'mono', 16, align='center')
    y += 34
    tax = total * 0.05
    for lab, val, fn, s in [('SUBTOTAL', total, 'mono', sz), ('CGST 2.5%', tax / 2, 'mono', sz - 2),
                            ('SGST 2.5%', tax / 2, 'mono', sz - 2), ('TOTAL', total + tax, 'mono_b', sz + 6)]:
        c.line(m, y, lab, fn, s)
        c.line(W - m, y, f"{val:,.2f}", fn, s, align='right')
        y += int(s * 1.6)
    y += 20
    c.line(W // 2, y, 'THANK YOU! VISIT AGAIN', 'mono', 16, align='center')
    y += 30
    c.line(W // 2, y, f"Paid via UPI Ref {rng.integers(10 ** 9, 10 ** 10)}", 'mono', 13, align='center')
    g, words, rules = c.finish()
    return g[:y + 50], words, rules


def render_invoice(rng):
    W, H, m = 1240, 1754, 80
    c = DocCanvas(W, H)
    c.line(m, 120, str(rng.choice(['NIMBUS TECH PVT LTD', 'ORBIT SOLUTIONS LLP', 'KAVYA SYSTEMS'])), 'sans_b',
           int(rng.integers(30, 38)))
    c.line(W - m, 120, 'TAX INVOICE', 'sans_b', int(rng.integers(34, 44)), align='right')
    small = int(rng.integers(14, 18))
    y = 170
    for t in [f"Invoice No: INV-{rng.integers(1000, 9999)}", f"Invoice Date: {rand_date(rng)}",
              f"Due Date: {rand_date(rng)}"]:
        c.line(W - m, y, t, 'sans', small, align='right')
        y += small * 1.6
    y2 = 170
    for t in ['Bill To:', 'Meridian Retail Ltd', f"{rng.integers(1, 300)} Ring Road, {rng.choice(CITIES)}",
              'GSTIN 27AAACM1234K1Z2']:
        c.line(m, y2, t, 'sans_b' if t == 'Bill To:' else 'sans', small + 2)
        y2 += (small + 2) * 1.6
    y = int(max(y, y2)) + 40
    tsz = int(rng.integers(16, 21))
    rows, sub = [['Description', 'Qty', 'Unit Price', 'Amount']], 0
    for _ in range(int(rng.integers(5, 10))):
        q, p = int(rng.integers(1, 20)), rng.uniform(100, 20000)
        sub += q * p
        rows.append([str(rng.choice(SERVICES)), str(q), f"{p:,.2f}", f"{q * p:,.2f}"])
    y = c.table(m, y, [500, 120, 220, 240], rows, 'sans', tsz, row_h=int(tsz * 2.2)) + 30
    for lab, val, fn, s in [('Subtotal', sub, 'sans', tsz), ('GST 18%', sub * 0.18, 'sans', tsz),
                            ('Total Due', sub * 1.18, 'sans_b', tsz + 6)]:
        c.line(W - m - 260, y + s, lab, fn, s, align='right')
        c.line(W - m, y + s, f"{val:,.2f}", fn, s, align='right')
        y += s * 1.8
    y += 40
    y = c.paragraph(m, y, W - 2 * m, 'Payment terms: ' + str(rng.choice(CLAUSES)).format(
        amt='INR ' + money(rng), n=rng.integers(7, 60), date=rand_date(rng), pct='12%', city='Pune'),
        'serif', int(rng.integers(12, 15))) + 30
    c.line(m, y + 20, f"Bank: HDFC Bank  A/C {rng.integers(10 ** 11, 10 ** 12)}  IFSC HDFC000{rng.integers(1000, 9999)}",
           'mono', small)
    c.hline(W - m - 360, W - m, y + 140)
    c.line(W - m - 180, y + 170, 'Authorised Signatory', 'sans', small, align='center')
    return c.finish()


RENDERERS = {'contract': render_contract, 'receipt': render_receipt, 'invoice': render_invoice}


def simulate_scan(img, words, rng, angle=None, clean=False):
    """Print-and-scan simulation: skew, optics blur, ink/paper levels, illumination, texture, noise, JPEG."""
    H, W = img.shape
    ang = rng.uniform(-1.5, 1.5) if angle is None else angle
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    out = cv2.warpAffine(img.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
    new_words = []
    for w in words:
        x0, y0, x1, y1 = w['box']
        pts = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], np.float32) @ M.T
        nw = dict(w)
        nw['box'] = [int(pts[:, 0].min()), int(pts[:, 1].min()), int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max()))]
        new_words.append(nw)
    if clean:
        return np.clip(out, 0, 255).astype(np.uint8), new_words
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 0.9))
    paper, ink = rng.uniform(215, 238), rng.uniform(20, 60)
    out = ink + out / 255.0 * (paper - ink)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    th = rng.uniform(0, 2 * np.pi)
    grad = (xx / W) * np.cos(th) + (yy / H) * np.sin(th)
    out *= 1 - rng.uniform(0.03, 0.14) * (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    out += cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 3) * rng.uniform(15, 35)
    out += rng.normal(0, rng.uniform(2.5, 5.0), (H, W))
    out = jpeg(np.clip(out, 0, 255).astype(np.uint8), rng.integers(75, 92))
    return out, new_words


def make_document(kind, rng, angle=None, clean=False):
    g, words, rules = RENDERERS[kind](rng)
    scan, words = simulate_scan(g, words, rng, angle=angle, clean=clean)
    return scan, words

# %% 1b. Tampering generator with pixel-exact ground truth --------------------
FORGE_FONTS = ['sans', 'serif', 'mono', 'sans_b']


def _bg_stats(img, box, pad):
    H, W = img.shape
    x0, y0, x1, y1 = box
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    ring = img[Y0:Y1, X0:X1].astype(np.float32)
    v = ring[ring > np.percentile(ring, 45)]
    med = float(np.median(v))
    sig = float(1.4826 * np.median(np.abs(v - med)))
    return med, sig


def _erase(img, box, rng, value, sigma, pad=2):
    H, W = img.shape
    x0, y0, x1, y1 = box
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    img[Y0:Y1, X0:X1] = np.clip(value + rng.normal(0, sigma, (Y1 - Y0, X1 - X0)), 0, 255).astype(np.uint8)
    return (X0, Y0, X1, Y1)


def _mutate(text, rng):
    digits = [i for i, ch in enumerate(text) if ch.isdigit()]
    if digits:
        s = list(text)
        for i in rng.choice(digits, size=min(len(digits), int(rng.integers(1, 3))), replace=False):
            s[i] = str((int(s[i]) + int(rng.integers(1, 9))) % 10)
        if rng.random() < 0.4:
            s.insert(digits[0], str(rng.integers(1, 9)))
        return ''.join(s)
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(rng.choice(list(letters)) for _ in range(max(2, len(text))))


def _render_patch(text, fname, target_h, angle):
    f0 = font(fname, 60)
    bb = f0.getbbox(text, anchor='ls')
    size = max(8, int(60 * target_h / max(1, bb[3] - bb[1])))
    f = font(fname, size)
    bb = f.getbbox(text, anchor='ls')
    w, h = bb[2] - bb[0] + 8, bb[3] - bb[1] + 8
    pim = Image.new('L', (w, h), 0)
    ImageDraw.Draw(pim).text((4 - bb[0], 4 - bb[1]), text, font=f, fill=255, anchor='ls')
    a = np.array(pim).astype(np.float32) / 255
    if abs(angle) > 0.01:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        a = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR)
    ys, xs = np.where(a > 0.05)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _blend(img, alpha, x, y, ink):
    H, W = img.shape
    h, w = alpha.shape
    x, y = int(x), int(y)
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return None
    reg = img[y:y + h, x:x + w].astype(np.float32)
    img[y:y + h, x:x + w] = np.clip(reg * (1 - alpha) + ink * alpha, 0, 255).astype(np.uint8)
    return (x, y, x + w, y + h)


def _paste(img, patch, x, y):
    H, W = img.shape
    h, w = patch.shape
    x, y = int(x), int(y)
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return None
    img[y:y + h, x:x + w] = patch
    return (x, y, x + w, y + h)


def tamper(img, words, rng, ops=None, n_ops=None, final_q=None):
    """Apply 1-2 forgeries on an already-scanned page. Returns forged image, GT mask, list of records."""
    img = img.copy()
    H, W = img.shape
    mask = np.zeros((H, W), np.uint8)
    hs = np.array([w['box'][3] - w['box'][1] for w in words])
    cand = [i for i, w in enumerate(words) if 9 <= hs[i] <= 80 and (w['box'][2] - w['box'][0]) >= 1.2 * hs[i]
            and w['box'][1] > 10 and w['box'][3] < H - 10]
    numeric = [i for i in cand if any(ch.isdigit() for ch in words[i].get('text', ''))]
    n_ops = n_ops or int(rng.integers(1, 3))
    ops = ops or list(rng.choice(['insert', 'copy_move', 'shift', 'delete'], size=n_ops, replace=False))
    used, recs = [], []

    def free(b):
        return all(b[2] + 10 < u[0] or u[2] + 10 < b[0] or b[3] + 10 < u[1] or u[3] + 10 < b[1] for u in used)

    for op in ops:
        pool = [i for i in (numeric if (numeric and rng.random() < 0.75) else cand) if free(words[i]['box'])]
        if not pool:
            continue
        t = int(rng.choice(pool))
        box = [int(v) for v in words[t]['box']]
        x0, y0, x1, y1 = box
        h = y1 - y0
        bgv, sig = _bg_stats(img, box, max(4, h // 2))
        regions = []
        if op == 'insert':
            crop = img[y0:y1, x0:x1]
            ink = float(np.percentile(crop, 8)) + rng.uniform(-10, 10)
            regions.append(_erase(img, box, rng, bgv, sig * rng.uniform(0.0, 0.4)))
            alpha = _render_patch(_mutate(words[t].get('text', 'x') or 'x', rng), str(rng.choice(FORGE_FONTS)),
                                  h * rng.uniform(0.9, 1.15), rng.choice([-1, 1]) * rng.uniform(0.8, 2.5))
            dy = rng.choice([-1, 1]) * rng.uniform(0.1, 0.3) * h
            r = _blend(img, alpha, x0 + rng.integers(-2, 3), y1 - alpha.shape[0] + dy, ink)
            if r: regions.append(r)
        elif op == 'copy_move':
            src = [i for i in cand if i != t and abs(hs[i] - h) <= 0.25 * h and free(words[i]['box'])
                   and not (words[i]['box'][0] == x0 and words[i]['box'][1] == y0)]
            if not src:
                continue
            s = int(rng.choice(src))
            sx0, sy0, sx1, sy1 = [int(v) for v in words[s]['box']]
            patch = img[max(0, sy0 - 2):sy1 + 2, max(0, sx0 - 2):sx1 + 2].copy()
            regions.append(_erase(img, box, rng, bgv, sig))
            r = _paste(img, patch, x0 - 2, y1 + 2 - patch.shape[0] + rng.uniform(-0.12, 0.12) * h)
            if r: regions.append(r)
        elif op == 'shift':
            patch = img[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2].copy()
            regions.append(_erase(img, box, rng, bgv, sig))
            dy = rng.choice([-1, 1]) * rng.uniform(0.2, 0.4) * h
            r = _paste(img, patch, x0 - 2 + rng.uniform(-0.1, 0.1) * h, y0 - 2 + dy)
            if r: regions.append(r)
        elif op == 'delete':
            regions.append(_erase(img, box, rng, bgv, sig * rng.uniform(0.0, 0.3), pad=3))
        rx0, ry0 = min(r[0] for r in regions), min(r[1] for r in regions)
        rx1, ry1 = max(r[2] for r in regions), max(r[3] for r in regions)
        for r in regions:
            mask[r[1]:r[3], r[0]:r[2]] = 1
        used.append((rx0, ry0, rx1, ry1))
        recs.append(dict(type=str(op), box=(rx0, ry0, rx1, ry1), text=words[t].get('text', '')))
    img = jpeg(img, final_q or rng.integers(80, 96))  # re-save => double compression
    return img, mask, recs

# %% 1c. Demo: one document of each type + its forgery --------------------------
rng = np.random.default_rng(7)
demo = {}
for kind in ['contract', 'receipt', 'invoice']:
    scan, words = make_document(kind, rng)
    forged, gt, recs = tamper(scan, words, rng, n_ops=2)
    demo[kind] = dict(scan=scan, words=words, forged=forged, gt=gt, recs=recs)
    print(kind, '->', [(r['type'], r['text']) for r in recs])

ims, tts = [], []
for k, d in demo.items():
    ov = to_rgb(d['forged'])
    cnts, _ = cv2.findContours(d['gt'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnts, -1, (255, 0, 0), 3)
    ims += [d['scan'], ov]
    tts += [f'{k}: genuine scan', f'{k}: forged (GT in red)']
show(ims, tts, cols=2, width=7, suptitle='Synthetic structured documents & ground-truth forgeries')

# %% 1d. Optional real data sources -----------------------------------------
def load_hf_images(name, split, n, streaming=True):
    """Clean real scans from HuggingFace (FUNSD forms / CORD receipts). Returns list of gray uint8 images."""
    try:
        from datasets import load_dataset
        ds = load_dataset(name, split=split, streaming=streaming)
        out = []
        for ex in ds:
            im = ex.get('image')
            if im is None:
                continue
            out.append(to_gray(np.array(im.convert('RGB'))))
            if len(out) >= n:
                break
        print(f'{name}: loaded {len(out)} images')
        return out
    except Exception as e:
        print(f'[skip] {name}: {e}')
        return []


USE_HF = IN_COLAB  # set True locally if you have internet + `datasets`
hf_bases = []
if USE_HF:
    hf_bases += [('funsd', g) for g in load_hf_images('nielsr/funsd', 'test', 6)]
    hf_bases += [('cord', g) for g in load_hf_images('naver-clova-ix/cord-v2', 'test', 6)]
if hf_bases:
    show([g for _, g in hf_bases[:6]], [n for n, _ in hf_bases[:6]], cols=3, width=4.5, suptitle='Real clean scans (FUNSD / CORD)')

# %% [markdown]
# ## 2. Phase A - Low-level vision
# The key design decision: **every kernel / window size is derived from the local character height**, so the same code works
# on 10-px fine print, 20-px body text and 45-px titles on the *same page*.

# %% A1-A4: scale estimation, illumination normalisation, local scale map, adaptive binarisation
def cc_stats(bw):
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw.astype(np.uint8), 8)
    return n, lab, st, cen


def char_like(st, H, W):
    w, h, a = st[:, 2], st[:, 3], st[:, 4]
    return (h >= 5) & (h < 0.12 * H) & (w < 0.25 * W) & (a >= 10) & (w / np.maximum(h, 1) < 6) & (h / np.maximum(w, 1) < 12)


def estimate_char_height(gray):
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, cen = cc_stats(bw)
    st = st[1:]
    keep = char_like(st, *gray.shape)
    hs = st[keep, 3]
    return float(np.median(hs)) if len(hs) > 10 else 20.0, hs


def normalize_illumination(gray, char_h):
    k = odd(3 * char_h, 15)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.medianBlur(bg, odd(k, 3) if k < 255 else 255)
    norm = cv2.divide(gray, bg, scale=255)
    return norm, bg


def build_scale_map(norm, char_h):
    """Local character-height field: CC heights -> coarse grid median -> NN fill -> smooth -> quantise to 1/3-octave bins."""
    H, W = norm.shape
    _, bw = cv2.threshold(norm, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, cen = cc_stats(bw)
    st, cen = st[1:], cen[1:]
    keep = char_like(st, H, W)
    cell = odd(max(24, 2.5 * char_h))
    gh, gw = math.ceil(H / cell), math.ceil(W / cell)
    gy = np.clip((cen[keep, 1] // cell).astype(int), 0, gh - 1)
    gx = np.clip((cen[keep, 0] // cell).astype(int), 0, gw - 1)
    df = pd.DataFrame({'c': gy * gw + gx, 'h': st[keep, 3]})
    grid = np.full(gh * gw, np.nan)
    if len(df):
        med = df.groupby('c')['h'].median()
        grid[med.index.values] = med.values
    grid = grid.reshape(gh, gw)
    if np.isnan(grid).all():
        grid[:] = char_h
    idx = ndimage.distance_transform_edt(np.isnan(grid), return_distances=False, return_indices=True)
    grid = grid[tuple(idx)]
    grid = ndimage.median_filter(grid, size=3)
    grid = ndimage.gaussian_filter(grid, 0.7)
    smap = cv2.resize(grid.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    q = np.round(np.log2(np.maximum(smap, 4)) * 3) / 3
    levels = np.unique(q)
    bin_map = np.searchsorted(levels, q).astype(np.int32)
    return smap, 2 ** levels, bin_map, (st[keep], cen[keep])


def adaptive_binarize(norm, levels, bin_map, k=0.25):
    """Sauvola threshold whose window is matched to the local character height (~2.2 x h)."""
    bw = np.zeros(norm.shape, np.uint8)
    windows = {}
    for i, lv in enumerate(levels):
        m = bin_map == i
        if not m.any():
            continue
        win = odd(max(15, 2.2 * lv))
        windows[f'{lv:.0f}px'] = win
        T = threshold_sauvola(norm, window_size=win, k=k)
        bw[m] = (norm < T)[m]
    min_area = 3
    n, lab, st, _ = cc_stats(bw)
    small = np.where(st[:, 4] < min_area)[0]
    bw[np.isin(lab, small[small > 0])] = 0
    return bw, windows

# %% A5-A7: deskew, gradients, forensic noise maps
def estimate_skew(bw, rng_deg=5.0):
    H, W = bw.shape
    s = 800 / max(H, W)
    small = cv2.resize(bw * 255, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    c = (small.shape[1] / 2, small.shape[0] / 2)

    def score(a):
        r = cv2.warpAffine(small, cv2.getRotationMatrix2D(c, a, 1), small.shape[::-1], flags=cv2.INTER_NEAREST)
        p = r.sum(1).astype(np.float64)
        return np.sum(np.diff(p) ** 2)
    coarse = np.arange(-rng_deg, rng_deg + 1e-6, 0.2)
    sc = np.array([score(a) for a in coarse])
    b = coarse[sc.argmax()]
    fine = np.arange(b - 0.2, b + 0.2001, 0.02)
    sf = np.array([score(a) for a in fine])
    return float(fine[sf.argmax()]), (coarse, sc)


def rotate(img, M, interp=cv2.INTER_LINEAR, border=0):
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def gradients(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 0.5 * otsu * 0.5, otsu * 0.5)  # auto thresholds from Otsu
    return gx, gy, mag, edges


def noise_maps(gray, bw, char_h):
    """Noise residual (image - median3); SLIDING-window noise level on background pixels only (window ~ 1 char height,
    box-filtered so it aligns with word-sized edits); robust z of log-sigma; ELA; gradient magnitude on the raw grid."""
    g = gray.astype(np.float32)
    resid = g - cv2.medianBlur(gray, 3).astype(np.float32)
    d = odd(0.5 * char_h, 5)
    ink_zone = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))) > 0  # exclude blurred glyph halos
    bgm = ((~ink_zone) & (gray < 253)).astype(np.float32)  # clipped white paper carries no noise -> not evidence
    win = odd(1.2 * char_h, 9)
    num = cv2.boxFilter(np.minimum(np.abs(resid), 30) * bgm, -1, (win, win), normalize=False)
    den = cv2.boxFilter(bgm, -1, (win, win), normalize=False)
    sig = np.where(den > 0.35 * win * win, 1.2533 * num / np.maximum(den, 1), np.nan).astype(np.float32)
    ls = np.log(np.maximum(sig, 0.3))
    sub = ls[::4, ::4]
    med = np.nanmedian(sub)
    mad = max(0.1, 1.4826 * np.nanmedian(np.abs(sub - med)))
    z = (ls - med) / mad
    ela = np.abs(g - jpeg(gray, 90).astype(np.float32))
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return dict(resid=resid, sigma_map=sig, z_map=z, win=win, ela=ela, mag=mag, page_sigma=float(np.exp(med)))


def clean_patch_map(z_map, thr, win):
    """Pixels whose background is abnormally CLEAN (one-sided), kept only as blobs of >= half a window."""
    m = (np.nan_to_num(-z_map) > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.where(st[:, 4] >= 0.5 * win * win)[0]
    keep = keep[keep > 0]
    return np.isin(lab, keep).astype(np.uint8)


def max_clean_z(z_map, win):
    """Largest one-sided z that survives a win/2 erosion (robust page-level 'too clean' evidence)."""
    zl = np.nan_to_num(-z_map, nan=-10).astype(np.float32)
    return float(cv2.erode(zl, np.ones((max(3, win // 2),) * 2, np.uint8)).max())


# %% [markdown]
# ## 3. Phase B - Mid-level vision (structure & geometry)
# ### Adaptive structuring elements ("adaptive morphology matrix")
# For each scale bin `l` (local char height) we **learn** the word-gap ratio from the data: horizontal gaps between
# consecutive glyphs on the same line form a bimodal distribution (intra-word vs inter-word). Otsu on `log(gap/l)` gives
# the split `r_l`, and the structuring element for that region becomes `rect(ceil(r_l*l), ~0.1*l)`. Table rules are removed
# first with *length-adaptive* opening kernels. The result is a spatially-variant closing: fine print gets a 3-px kernel,
# titles a 15-px kernel, on the same page.

# %% B1-B3: rule lines, adaptive kernels, words
def adaptive_morph(bw, bin_map, levels, op, kfn):
    out = np.zeros_like(bw)
    for i, lv in enumerate(levels):
        m = bin_map == i
        if not m.any():
            continue
        kw, kh = kfn(lv)
        res = cv2.morphologyEx(bw, op, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
        out[m] = res[m]
    return out


def extract_rules(bw, char_h, levels):
    H, W = bw.shape
    Lh = int(max(W * 0.04, 5 * char_h))
    Lv = int(max(H * 0.02, 2.5 * max(levels)))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (Lh, 1)))
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, Lv)))
    rules = cv2.bitwise_or(hor, ver)
    segs = cv2.HoughLinesP(rules * 255, 1, np.pi / 720, threshold=int(Lh * 0.8), minLineLength=Lv, maxLineGap=int(char_h))
    segs = np.zeros((0, 4), int) if segs is None else segs.reshape(-1, 4)
    text = bw.copy()
    text[cv2.dilate(rules, np.ones((3, 3), np.uint8)) > 0] = 0
    return text, rules, hor, ver, segs, (Lh, Lv)


def learn_gap_ratios(bw_text, bin_map, levels):
    n, clab, cst, ccen = cc_stats(bw_text)
    if n < 5:
        return {lv: 0.45 for lv in levels}, np.array([]), 0.45
    lines = adaptive_morph(bw_text, bin_map, levels, cv2.MORPH_CLOSE, lambda l: (odd(1.5 * l, 3), 1))
    _, llab, _, _ = cc_stats(lines)
    line_of = np.asarray(ndimage.maximum(llab, labels=clab, index=np.arange(1, n)))
    st, cen = cst[1:], ccen[1:]
    lb = bin_map[np.clip(cen[:, 1].astype(int), 0, bw_text.shape[0] - 1), np.clip(cen[:, 0].astype(int), 0, bw_text.shape[1] - 1)]
    lvl = levels[lb]
    df = pd.DataFrame({'line': line_of, 'x0': st[:, 0], 'x1': st[:, 0] + st[:, 2], 'h': st[:, 3], 'bin': lb, 'lvl': lvl})
    df = df[df.h >= 0.3 * df.lvl].sort_values(['line', 'x0'])
    df['x1max'] = df.groupby('line')['x1'].cummax()
    df['gap'] = df.groupby('line')['x0'].shift(-1) - df['x1max']
    g = df.dropna(subset=['gap'])
    g = g[(g.gap > 0) & (g.gap < 2.5 * g.lvl)]
    ratios = (g.gap / g.lvl).values

    def otsu_ratio(r):
        # Otsu separates the modes coarsely; the split is then moved to the density VALLEY between the
        # intra-word peak and the inter-word peak (Otsu alone tends to cut into the tail of the big intra mode).
        if len(r) < 25:
            return None
        lr = np.log(r)
        t = threshold_otsu(lr)
        hist, edges = np.histogram(lr, bins=40)
        hist = gaussian_filter1d(hist.astype(float), 1.0)
        c = (edges[:-1] + edges[1:]) / 2
        lo, hi = np.where(c < t)[0], np.where(c >= t)[0]
        if len(lo) and len(hi):
            p1, p2 = lo[np.argmax(hist[lo])], hi[np.argmax(hist[hi])]
            if p2 > p1 + 1:
                seg = hist[p1:p2 + 1]
                mins = np.where(seg <= seg.min() + 1e-9)[0]
                t = c[p1 + int(np.round(mins.mean()))]
        return float(np.clip(np.exp(t), 0.25, 1.0))
    pooled = otsu_ratio(ratios) or 0.45
    out = {}
    for i, lv in enumerate(levels):
        r = (g[g.bin == i].gap / g[g.bin == i].lvl).values
        out[lv] = otsu_ratio(r) or pooled
    return out, ratios, pooled


def word_kernel(lv, ratio):
    return (max(3, 2 * int(math.ceil(ratio * lv / 2)) + 1), odd(0.1 * lv))


def detect_words(bw_text, bin_map, levels, ratios, scale_map):
    closed = adaptive_morph(bw_text, bin_map, levels, cv2.MORPH_CLOSE, lambda l: word_kernel(l, ratios[l]))
    n, lab, st, cen = cc_stats(closed)
    words = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        l = float(scale_map[int(cen[i, 1]), int(cen[i, 0])])
        if (h < 0.3 * l and w < 0.3 * l) or a < 6:
            continue
        words.append(dict(id=len(words), label=i, box=(int(x), int(y), int(x + w), int(y + h)), l=l,
                          graphic=bool(h < 0.42 * l or w / max(h, 1) > 30 or (h < 0.6 * l and w < 0.6 * l))))
    return words, closed, lab


def detect_words_fixed(bw_text, kw, kh=3):
    closed = cv2.morphologyEx(bw_text, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)))
    n, lab, st, cen = cc_stats(closed)
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, a in st[1:] if a >= 6], closed

# %% B4-B6: projection profiles, line segments, baselines, Hough parallelism
def projection_lines(bw_text, char_h):
    prof = bw_text.sum(1).astype(np.float64)
    sm = gaussian_filter1d(prof, max(1.0, 0.12 * char_h))
    if sm.max() <= 0:
        return [], prof, sm, []
    pk, _ = find_peaks(sm, distance=max(3, int(0.5 * char_h)), prominence=0.04 * sm.max())
    bands = []
    for j, p in enumerate(pk):
        lo = pk[j - 1] + np.argmin(sm[pk[j - 1]:p]) if j > 0 else 0
        hi = p + np.argmin(sm[p:pk[j + 1]]) if j < len(pk) - 1 else len(sm) - 1
        thr = 0.08 * sm[p]
        ys = np.where(sm[lo:hi + 1] > thr)[0]
        if len(ys):
            bands.append((lo + ys[0], lo + ys[-1], p))
    return bands, prof, sm, pk


def densest(v, tol):
    v = np.sort(v)
    hi = np.searchsorted(v, v + 2 * tol, side='right')
    cnt = hi - np.arange(len(v))
    i = int(np.argmax(cnt))
    return float(np.median(v[i:hi[i]]))


def word_geometry(words, bw_text, lab):
    """Glyph-level geometry: each glyph = connected component inside the word. Baseline = densest cluster of glyph
    bottoms (descenders are a minority), top line likewise, angle = Theil-Sen fit through on-baseline glyph bottoms."""
    for w in words:
        x0, y0, x1, y1 = w['box']
        m = (lab[y0:y1, x0:x1] == w['label']) & (bw_text[y0:y1, x0:x1] > 0)
        l = w['l']
        w.update(baseline=float(y1), top=float(y0), angle=np.nan, mask=m, span=0.0, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2)
        n, _, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        st, cen = st[1:], cen[1:]
        keep = st[:, 3] >= 0.3 * l  # drop dots, commas' tails, specks
        if keep.sum() < 1:
            continue
        st, cen = st[keep], cen[keep]
        bottoms = (st[:, 1] + st[:, 3]).astype(float)
        tops = st[:, 1].astype(float)
        base = densest(bottoms, max(1.0, 0.12 * l))
        top = densest(tops, max(1.0, 0.12 * l))
        sel = np.abs(bottoms - base) <= max(1.0, 0.15 * l)
        span = float(np.ptp(cen[sel, 0])) if sel.sum() > 1 else 0.0
        ang = np.nan
        if sel.sum() >= 4 and span >= 2.5 * l and not w['graphic']:
            ang = float(np.degrees(np.arctan(theilslopes(bottoms[sel], cen[sel, 0])[0])))
        w.update(baseline=y0 + base, top=y0 + top, angle=ang, span=span)
    return words


def group_lines(words, bands):
    """Assign words to projection bands, then split each band into segments at large gaps (columns / table cells)."""
    centers = np.array([b[2] for b in bands]) if bands else np.array([0])
    for w in words:
        yb = w['baseline'] - 0.3 * w['l']
        inside = [k for k, b in enumerate(bands) if b[0] <= yb <= b[1]]
        w['band'] = inside[0] if inside else int(np.argmin(np.abs(centers - yb)))
    segs = []
    for b in sorted(set(w['band'] for w in words)):
        ws = sorted([w for w in words if w['band'] == b and not w['graphic']], key=lambda w: w['box'][0])
        cur = []
        for w in ws:
            if cur and w['box'][0] - cur[-1]['box'][2] > 2.0 * max(w['l'], cur[-1]['l']):
                segs.append(cur)
                cur = []
            cur.append(w)
        if cur:
            segs.append(cur)
    for s_id, s in enumerate(segs):
        for j, w in enumerate(s):
            w['seg'], w['pos'] = s_id, j
    return segs


def hough_baselines(words, shape):
    pts = np.zeros(shape, np.uint8)
    for w in words:
        if w['graphic'] or 'mask' not in w:
            continue
        x0, y0, _, _ = w['box']
        m = w['mask']
        cols = np.where(m.any(0))[0]
        bottoms = m.shape[0] - 1 - np.argmax(m[::-1], 0)[cols]
        ok = np.abs(y0 + bottoms - w['baseline']) <= max(1.5, 0.1 * w['l'])
        pts[y0 + bottoms[ok], x0 + cols[ok]] = 255
    thr = max(25, int(0.04 * shape[1]))
    lines = cv2.HoughLines(pts, 1, np.pi / 1800, thr, min_theta=np.radians(80), max_theta=np.radians(100))
    lines = np.zeros((0, 2)) if lines is None else lines.reshape(-1, lines.shape[-1])[:, :2]  # OpenCV 4/5 shape-agnostic
    angs = np.degrees(lines[:, 1]) - 90
    dom = float(np.median(angs)) if len(angs) else 0.0
    return pts, lines, angs, dom

# %% [markdown]
# ## 4. Phase C - Region clustering, consistency checks, copy-move, fusion

# %% C1-C4
FEATURES = ['baseline', 'angle', 'height', 'gap', 'stroke', 'ink', 'ink_noise', 'noise', 'ela', 'sharp', 'copy']
WEIGHTS = dict(baseline=1.0, angle=0.8, height=0.6, gap=0.5, stroke=0.6, ink=0.6, ink_noise=0.8, noise=1.0, ela=0.5, sharp=0.7, copy=0.7)


def word_features(words, segs, gray, bw_text, mag, nm, dom_angle):
    resid, ela = nm['resid'], nm['ela']
    ink_zone = cv2.dilate(bw_text, np.ones((3, 3), np.uint8)) > 0
    H, W = gray.shape
    rows = []
    for w in words:
        x0, y0, x1, y1 = w['box']
        l = w['l']
        m = w['mask']
        f = dict(id=w['id'], x0=x0, y0=y0, x1=x1, y1=y1, l=l, seg=w.get('seg', -1), band=w.get('band', -1), graphic=w['graphic'])
        crop = gray[y0:y1, x0:x1].astype(np.float32)
        inkpx = crop[m] if m.any() else crop.ravel()
        f['ink_level'] = float(np.percentile(inkpx, 20))  # darkest core: insensitive to stroke-width blur
        core = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        if core.sum() < 15:
            core = m & (crop <= np.percentile(inkpx, 35))
        rc = resid[y0:y1, x0:x1][core]
        # digitally typed/pasted text has flat ink; scanned ink carries sensor noise
        f['ink_noise'] = float(1.2533 * np.mean(np.abs(rc - np.median(rc)))) if len(rc) >= 10 else np.nan
        p = int(0.15 * l) + 1  # tight: a pasted/erased patch rarely extends far beyond the word
        X0, Y0, X1, Y1 = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
        bgm = ~ink_zone[Y0:Y1, X0:X1]
        bgv = gray[Y0:Y1, X0:X1][bgm].astype(np.float32)
        paper = float(np.median(bgv)) if len(bgv) > 10 else 255.0
        r = resid[Y0:Y1, X0:X1][bgm]
        r = r[np.abs(r) < 30]
        f['noise_level'] = float(1.2533 * np.mean(np.abs(r - np.median(r)))) if len(r) > 20 else np.nan
        edge = (cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
        contrast = max(10.0, paper - f['ink_level'])
        g_edge = mag[y0:y1, x0:x1][edge] if edge.any() else np.array([0.0])
        f['sharpness'] = float(np.percentile(g_edge, 90) / contrast)  # edge steepness per unit contrast
        f['ela_level'] = float(np.mean(ela[y0:y1, x0:x1][edge]) / (np.mean(g_edge) + 1)) if edge.any() else np.nan
        if m.sum() > 10:
            dist = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
            sk = skeletonize(m)
            f['stroke_w'] = float(2 * np.mean(dist[sk])) if sk.any() else np.nan
        else:
            f['stroke_w'] = np.nan
        f['baseline_y'], f['top_y'], f['angle'], f['span'] = w['baseline'], w['top'], w['angle'], w.get('span', 0.0)
        rows.append(f)
    df = pd.DataFrame(rows).set_index('id')
    df['off'] = np.nan
    df['asc_excess'] = np.nan
    df['gap_dev'] = np.nan
    wide = {w['id']: (w['box'][2] - w['box'][0]) >= 1.2 * w['l'] for w in words}  # single glyphs: baseline unreliable
    for s in segs:
        ids = [w['id'] for w in s]
        cx = np.array([(w['box'][0] + w['box'][2]) / 2 for w in s])
        base = df.loc[ids, 'baseline_y'].values
        asc = base - df.loc[ids, 'top_y'].values
        gaps = np.array([s[j + 1]['box'][0] - s[j]['box'][2] for j in range(len(s) - 1)], float)
        for j, wid in enumerate(ids):
            l = df.at[wid, 'l']
            if wide[wid]:
                # row peers by PROXIMITY (not band id): same scale, baseline within 0.8 l, nearest 4 along x.
                # Works across table columns / receipt amounts, and a shifted word cannot escape into another band.
                peers = [w for w in words if w['id'] != wid and not w['graphic'] and wide[w['id']]
                         and abs(w['baseline'] - base[j]) < 0.8 * l and abs(np.log(w['l'] / l)) < 0.3
                         and abs(w['cx'] - cx[j]) < 60 * l]
                peers = sorted(peers, key=lambda w: abs(w['cx'] - cx[j]))[:4]
                if peers:
                    df.at[wid, 'off'] = (base[j] - np.median([w['baseline'] for w in peers])) / l
            others = [k for k in range(len(ids)) if k != j]
            if len(others) >= 2 and wide[wid]:
                df.at[wid, 'asc_excess'] = max(0.0, (asc[j] - np.percentile(asc[others], 80)) / l)
            if len(gaps) >= 3:
                mine = [g for g in (j - 1, j) if 0 <= g < len(gaps)]
                rest = np.delete(gaps, mine)
                ref = np.median(rest) + 1
                df.at[wid, 'gap_dev'] = max(abs(np.log((gaps[g] + 1) / ref)) for g in mine)
    df['angle_dev'] = (df['angle'] - dom_angle).abs()
    return df


def robust_z(d, floor):
    d = np.asarray(d, float)
    med = np.nanmedian(d)
    mad = 1.4826 * np.nanmedian(np.abs(d - med))
    return np.nan_to_num((d - med) / max(floor, mad if np.isfinite(mad) else floor), nan=0.0)


def neighbour_residual(df, col, k=8, log=False, match_stroke=False):
    v = df[col].values.astype(float)
    if log:
        v = np.log(np.maximum(v, 1e-3))
    ok = np.isfinite(v) & ~df['graphic'].values
    out = np.full(len(df), np.nan)
    if ok.sum() < 4:
        return out
    pts = np.c_[(df.x0 + df.x1) / 2, (df.y0 + df.y1) / 2]
    ll = np.log(df.l.values)
    tree = cKDTree(pts[ok])
    vv, lo, po = v[ok], ll[ok], pts[ok]
    kk = min(4 * k + 1, ok.sum())
    _, nn = tree.query(pts, k=kk)
    band = df['band'].values
    bo = band[ok]
    sw = np.log(np.maximum(df['stroke_w'].fillna(1).values, 0.5))
    so = sw[ok]
    for i in range(len(df)):
        if not np.isfinite(v[i]):
            continue
        cand = [j for j in np.atleast_1d(nn[i]) if abs(lo[j] - ll[i]) < 0.25 and not np.allclose(po[j], pts[i])
                and (not match_stroke or abs(so[j] - sw[i]) < 0.3)]
        same_row = [j for j in cand if bo[j] == band[i]]  # same text row first: bold rows compare with bold rows
        nb = same_row[:k] if len(same_row) >= 2 else cand[:k]
        if len(nb) >= 2:  # compare only with words of the same font scale
            out[i] = v[i] - np.median(vv[nb])
    return out


def copy_move(df, gray, resid, bw_text, page_sigma):
    """Look-alike words (same size, NCC > 0.9) are aligned; the variance of their pixel DIFFERENCE on flat pixels is
    compared with the page median over all look-alike pairs. Genuine repeats differ by two independent noise
    realisations; a duplicated patch differs only by re-compression error -> ratio << 1. Self-normalising."""
    score = np.zeros(len(df))
    pairs = []
    if page_sigma < 0.8:  # born-digital page: identical glyphs are legitimately identical
        return score, pairs, 'skipped (born-digital / noise-free page)'
    mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    flat = mag < 10 * page_sigma
    ids = list(df.index)
    cand = df[(~df.graphic) & ((df.x1 - df.x0) >= 1.5 * df.l)]
    keys = {}
    for i, r in cand.iterrows():
        keys.setdefault((int((r.x1 - r.x0) // 3), int((r.y1 - r.y0) // 3)), []).append(i)
    stats = []
    for (kw, kh), lst in keys.items():
        pool = []
        for dw in (-1, 0, 1):
            for dh in (-1, 0, 1):
                pool += keys.get((kw + dw, kh + dh), [])
        for a in lst:
            for b in pool:
                if b <= a or len(stats) > 20000:
                    continue
                A, B = df.loc[a], df.loc[b]
                ax0, ay0, ax1, ay1 = int(A.x0), int(A.y0), int(A.x1), int(A.y1)
                bx0, by0, bx1, by1 = int(B.x0) - 3, int(B.y0) - 3, int(B.x1) + 3, int(B.y1) + 3
                if bx0 < 0 or by0 < 0 or bx1 > gray.shape[1] or by1 > gray.shape[0]:
                    continue
                ta, sb = gray[ay0:ay1, ax0:ax1], gray[by0:by1, bx0:bx1]
                if sb.shape[0] < ta.shape[0] or sb.shape[1] < ta.shape[1]:
                    continue
                _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(sb, ta, cv2.TM_CCOEFF_NORMED))
                if mx < 0.9:
                    continue
                ox, oy = bx0 + loc[0], by0 + loc[1]
                tb = gray[oy:oy + ta.shape[0], ox:ox + ta.shape[1]]
                fl = flat[ay0:ay1, ax0:ax1] & flat[oy:oy + ta.shape[0], ox:ox + ta.shape[1]]
                if fl.sum() < 40:
                    continue
                D = ta.astype(np.float32) - tb.astype(np.float32)
                stats.append((a, b, float(mx), float(D[fl].var())))
    if len(stats) < 6:
        return score, pairs, f'{len(stats)} look-alike pairs (too few to normalise)'
    ref = np.median([v for *_, v in stats])
    for a, b, mx, v in stats:
        ratio = v / max(ref, 1e-6)
        if ratio < 0.45:
            c = float(np.clip((0.45 - ratio) / 0.45, 0, 1))
            for k in (a, b):
                score[ids.index(k)] = max(score[ids.index(k)], c)
            pairs.append((a, b, mx, ratio))
    return score, pairs, f'{len(stats)} look-alike pairs, reference diff-variance {ref:.1f}'


def score_words(df, cm_score):
    Z = pd.DataFrame(index=df.index)
    o = df['off'].values
    sig_o = max(0.04, 1.4826 * np.nanmedian(np.abs(o - np.nanmedian(o)))) if np.isfinite(o).any() else 0.04
    Z['baseline'] = np.nan_to_num(np.abs(o) / np.sqrt(sig_o ** 2 + (1.0 / df['l'].values) ** 2))  # 1px quantisation
    sig_w = np.degrees(1.5 / np.maximum(df['span'].values, 1))  # 1.5px endpoint uncertainty over the fitted span
    a = df['angle_dev'].values
    sig_p = max(0.3, 1.4826 * np.nanmedian(np.abs(a - np.nanmedian(a)))) if np.isfinite(a).any() else 0.3
    Z['angle'] = np.nan_to_num(a / np.sqrt(sig_p ** 2 + sig_w ** 2))
    Z['height'] = np.nan_to_num(df['asc_excess'].values / np.sqrt(0.08 ** 2 + (1.0 / df['l'].values) ** 2))
    Z['gap'] = np.abs(robust_z(df['gap_dev'], 0.25)) * np.isfinite(df['gap_dev'])
    Z['stroke'] = np.abs(robust_z(neighbour_residual(df, 'stroke_w', log=True), 0.15))
    Z['ink'] = np.abs(robust_z(neighbour_residual(df, 'ink_level', match_stroke=True), 10.0))
    Z['noise'] = np.abs(robust_z(neighbour_residual(df, 'noise_level', log=True), 0.12))
    Z['ink_noise'] = np.abs(robust_z(neighbour_residual(df, 'ink_noise', log=True, match_stroke=True), 0.12))
    Z['ela'] = np.abs(robust_z(neighbour_residual(df, 'ela_level', log=True), 0.12))
    Z['sharp'] = np.abs(robust_z(neighbour_residual(df, 'sharpness', log=True), 0.08))
    Z['copy'] = np.where(cm_score > 0, 2.5 + 5 * cm_score, 0)
    Z.loc[df.graphic.values, :] = 0
    Z = Z.clip(0, 10)
    # chi-style evidence fusion: under H0 each z is ~|N(0,1)| so sum(w z^2) ~ sum(w); subtracting sqrt(sum w)
    # centres genuine words near 0 while one strong cue OR several moderate cues both raise the score.
    contrib = pd.DataFrame({f: WEIGHTS[f] * Z[f] ** 2 for f in FEATURES})
    S = np.sqrt(contrib.sum(1)) - np.sqrt(sum(WEIGHTS.values()))
    reason = contrib.idxmax(1).where(S > 0, '')
    return Z, contrib, S, reason


PIPE = dict(word_thr=3.0, block_thr=6.0)  # re-calibrated below for a target false-alarm rate


def run_pipeline(img, verbose=True, cfg=PIPE, title=''):
    R = {}
    t0 = time.time()
    gray = to_gray(img)
    H, W = gray.shape
    char_h, hs = estimate_char_height(gray)
    norm, bg = normalize_illumination(gray, char_h)
    smap, levels, bin_map, _ = build_scale_map(norm, char_h)
    bw0, windows = adaptive_binarize(norm, levels, bin_map)
    if verbose:
        print(f'[A1] global char height = {char_h:.1f}px from {len(hs)} glyph components')
        fig, ax = plt.subplots(1, 4, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1, 0.8]})
        for a, im, t in zip(ax[:3], [gray, bg, norm], ['A1 grayscale input', 'A2 estimated background (closing k=3h)', 'A2 illumination-normalised']):
            a.imshow(im, cmap='gray'); a.set_title(t); a.axis('off')
        ax[3].hist(hs, bins=50, color='steelblue'); ax[3].axvline(char_h, color='r'); ax[3].set_title('A1 glyph height histogram (px)')
        plt.tight_layout(); plt.show()
        fig, ax = plt.subplots(1, 3, figsize=(18, 7))
        im = ax[0].imshow(smap, cmap='viridis'); fig.colorbar(im, ax=ax[0], fraction=0.035); ax[0].set_title('A3 local char-height map (px)')
        im = ax[1].imshow(bin_map, cmap='tab10', interpolation='nearest'); ax[1].set_title('A3 scale bins: ' + ', '.join(f'{v:.0f}' for v in levels))
        ax[2].imshow(bw0, cmap='gray_r'); ax[2].set_title('A4 scale-adaptive Sauvola, windows: ' + ', '.join(f'{k}->{v}' for k, v in windows.items()), fontsize=8)
        for a in ax: a.axis('off')
        plt.tight_layout(); plt.show()

    nm0 = noise_maps(gray, bw0, char_h)  # forensic maps on the ORIGINAL pixel grid (rotation would destroy JPEG/noise traces)
    angle, (angs, scs) = estimate_skew(bw0)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    Minv = cv2.invertAffineTransform(M)
    g_r = rotate(gray, M, border=int(np.median(gray)))
    bw = (rotate(bw0 * 255, M, cv2.INTER_NEAREST) > 127).astype(np.uint8)
    smap_r = rotate(smap, M, cv2.INTER_NEAREST, border=float(char_h))
    bin_r = rotate(bin_map.astype(np.float32), M, cv2.INTER_NEAREST, border=float(np.median(bin_map))).astype(np.int32)
    nm = dict(nm0)
    nm['resid'] = rotate(nm0['resid'], M, cv2.INTER_NEAREST)
    nm['ela'] = rotate(nm0['ela'], M, cv2.INTER_NEAREST)
    gx, gy, mag_r, edges = gradients(g_r)
    mag = rotate(nm0['mag'], M, cv2.INTER_NEAREST)  # sharpness measured on the raw grid (rotation would blur all edges)
    if verbose:
        print(f'[A5] estimated skew = {angle:+.2f} deg')
        fig, ax = plt.subplots(1, 2, figsize=(14, 3))
        ax[0].plot(angs, scs); ax[0].axvline(angle, color='r'); ax[0].set_title('A5 profile sharpness vs rotation angle'); ax[0].set_xlabel('deg')
        ax[1].imshow(g_r, cmap='gray', aspect='auto'); ax[1].set_title('A5 deskewed page'); ax[1].axis('off')
        plt.tight_layout(); plt.show()
        show([np.abs(gx), np.abs(gy), mag_r, edges], ['A6 |Sobel x|', 'A6 |Sobel y|', 'A6 gradient magnitude', 'A6 auto-Canny edges'],
             cols=4, width=4.5, cmaps=['magma', 'magma', 'magma', 'gray'])
        r_vis = np.clip(np.abs(nm0['resid']) * 20, 0, 255).astype(np.uint8)
        ela_vis = np.clip(nm0['ela'] * 15, 0, 255).astype(np.uint8)
        show([r_vis, np.nan_to_num(nm0['sigma_map']), np.clip(np.nan_to_num(nm0['z_map']), -8, 8), ela_vis],
             ['A7 noise residual |I - median3(I)|', f"A7 sliding background-noise sigma (win={nm0['win']}px)",
              'A7 noise-level robust z (blue = too clean)', 'A7 Error Level Analysis (q=90)'],
             cols=4, width=4.5, cmaps=['inferno', 'viridis', 'coolwarm', 'inferno'], colorbar=True)

    text, rules, hor, ver, rsegs, (Lh, Lv) = extract_rules(bw, char_h, levels)
    ratios, gap_samples, pooled = learn_gap_ratios(text, bin_r, levels)
    words, closed, lab = detect_words(text, bin_r, levels, ratios, smap_r)
    fixed_boxes, fixed_closed = detect_words_fixed(text, odd(0.45 * char_h, 3))
    if verbose:
        rv = to_rgb(g_r)
        for x0, y0, x1, y1 in rsegs:
            cv2.line(rv, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 3)
        show([hor * 255, ver * 255, rv, text * 255],
             [f'B1 horizontal rules (open {Lh}x1)', f'B1 vertical rules (open 1x{Lv})', f'B1 HoughLinesP: {len(rsegs)} rule segments',
              'B1 text layer (rules removed)'], cols=4, width=4.5)
        fig, ax = plt.subplots(1, 2, figsize=(16, 3.5))
        if len(gap_samples):
            ax[0].hist(np.log(gap_samples), bins=60, color='gray')
            ax[0].axvline(np.log(pooled), color='r', label=f'pooled Otsu split r={pooled:.2f}')
            ax[0].legend(); ax[0].set_title('B2 log(glyph gap / local char height): intra- vs inter-word modes')
        ks = {f'{lv:.0f}px': word_kernel(lv, ratios[lv]) for lv in levels}
        ax[1].axis('off')
        tbl = ax[1].table(cellText=[[k, f'{ratios[lv]:.2f}', f'{v[0]} x {v[1]}'] for (k, v), lv in zip(ks.items(), levels)],
                          colLabels=['scale bin (char h)', 'learned gap ratio', 'structuring element (w x h)'], loc='center')
        tbl.scale(1, 1.6); ax[1].set_title('B2 adaptive structuring-element matrix')
        plt.tight_layout(); plt.show()
        a = to_rgb(g_r); b = to_rgb(g_r)
        for x0, y0, x1, y1 in fixed_boxes:
            cv2.rectangle(a, (x0, y0), (x1, y1), (230, 120, 0), 2)
        for w in words:
            x0, y0, x1, y1 = w['box']
            cv2.rectangle(b, (x0, y0), (x1, y1), (150, 150, 150) if w['graphic'] else (0, 160, 0), 2)
        show([fixed_closed * 255, a, closed * 255, b],
             [f'B3 FIXED kernel closing ({odd(0.45 * char_h, 3)}x3)', f'B3 fixed-kernel words: {len(fixed_boxes)}',
              'B3 ADAPTIVE spatially-variant closing', f'B3 adaptive words: {len(words)}'], cols=4, width=4.5)

    bands, prof, sm, pk = projection_lines(text, char_h)
    words = word_geometry(words, text, lab)
    segs = group_lines(words, bands)
    pts, hlines, hangs, dom = hough_baselines(words, g_r.shape)
    if verbose:
        fig, ax = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 0.35, 1]})
        lv = to_rgb(g_r)
        cmap = plt.get_cmap('tab20')
        for k, (y0, y1, p) in enumerate(bands):
            c = tuple(int(255 * v) for v in cmap(k % 20)[:3])
            cv2.rectangle(lv, (0, int(y0)), (W - 1, int(y1)), c, 2)
        ax[0].imshow(lv); ax[0].set_title(f'B4 {len(bands)} text-line bands from horizontal projection'); ax[0].axis('off')
        ax[1].plot(prof, np.arange(len(prof)), color='lightgray'); ax[1].plot(sm, np.arange(len(sm)), color='k')
        ax[1].plot(sm[pk], pk, 'r.'); ax[1].invert_yaxis(); ax[1].set_title('B4 horizontal profile'); ax[1].set_ylim(H, 0)
        sv = to_rgb(g_r)
        for s_id, s in enumerate(segs):
            c = tuple(int(255 * v) for v in cmap(s_id % 20)[:3])
            xs = [w['box'][0] for w in s] + [w['box'][2] for w in s]
            cv2.line(sv, (min(xs), int(np.median([w['baseline'] for w in s]))), (max(xs), int(np.median([w['baseline'] for w in s]))), c, 2)
            for w in s:
                x0, y0, x1, y1 = w['box']
                cv2.line(sv, (x0, int(w['baseline'])), (x1, int(w['baseline'])), (255, 0, 0), 1)
        ax[2].imshow(sv); ax[2].set_title(f'B4/B5 {len(segs)} line segments + per-word baselines (red)'); ax[2].axis('off')
        plt.tight_layout(); plt.show()
        vprof = text.sum(0)
        hv = to_rgb(cv2.dilate(pts, np.ones((3, 3), np.uint8)))
        for r, t in hlines[:200]:
            a_, b_ = np.cos(t), np.sin(t)
            x0_, y0_ = a_ * r, b_ * r
            cv2.line(hv, (int(x0_ - 3000 * b_), int(y0_ + 3000 * a_)), (int(x0_ + 3000 * b_), int(y0_ - 3000 * a_)), (0, 200, 255), 1)
        fig, ax = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 1]})
        ax[0].plot(vprof, color='k'); ax[0].set_title('B4 vertical projection profile (columns / table cells)')
        ax[1].imshow(hv); ax[1].set_title(f'B6 Hough on baseline points: {len(hlines)} lines'); ax[1].axis('off')
        wa = np.array([w['angle'] for w in words if np.isfinite(w.get('angle', np.nan))])
        ax[2].hist(wa, bins=60, color='steelblue', alpha=0.7, label='per-word baseline angle')
        if len(hangs): ax[2].hist(hangs, bins=30, color='orange', alpha=0.7, label='Hough line angles')
        ax[2].axvline(dom, color='r', label=f'dominant {dom:+.2f} deg'); ax[2].legend(); ax[2].set_title('B6 parallelism check (deg)')
        plt.tight_layout(); plt.show()

    df = word_features(words, segs, g_r, text, mag, nm, dom)
    cm_score, pairs, cm_msg = copy_move(df, g_r, nm['resid'], text, nm['page_sigma'])
    Z, contrib, S, reason = score_words(df, cm_score)
    df['score'], df['reason'] = S.values, reason.values
    df['flag'] = df['score'] > cfg['word_thr']

    # one-sided noise evidence: pasted / white-out patches are abnormally CLEAN (text halos make the upper tail noisy)
    blk_mask = clean_patch_map(nm0['z_map'], cfg['block_thr'], nm0['win'])
    blk = np.where(blk_mask > 0, np.nan_to_num(-nm0['z_map']), 0).astype(np.float32)
    clean_z = max_clean_z(nm0['z_map'], nm0['win'])
    heat_r = np.zeros((H, W), np.float32)
    mask_r = np.zeros((H, W), np.uint8)
    for i, r in df.iterrows():
        x0, y0, x1, y1 = int(r.x0), int(r.y0), int(r.x1), int(r.y1)
        heat_r[y0:y1, x0:x1] = np.maximum(heat_r[y0:y1, x0:x1], r.score)
        if r.flag:
            mask_r[y0:y1, x0:x1] = 1
    heat = np.maximum(rotate(heat_r, Minv), (blk - cfg['block_thr'] + cfg['word_thr']) * (blk > 0))
    heat = cv2.GaussianBlur(heat, (0, 0), max(2, 0.3 * char_h))
    mask = np.maximum(rotate(mask_r, Minv, cv2.INTER_NEAREST), blk_mask)
    img_score = float(max(df.score.max() if len(df) else 0, clean_z - cfg['block_thr'] + cfg['word_thr']))
    R.update(gray=gray, deskewed=g_r, angle=angle, M=M, char_h=char_h, levels=levels, ratios=ratios, words=words, segs=segs,
             df=df, Z=Z, contrib=contrib, heat=heat, mask=mask, img_score=img_score, pairs=pairs, noise=nm0, time=time.time() - t0)

    if verbose:
        print(f'[C3] copy-move: {cm_msg}; {len(pairs)} suspicious duplicate pairs')
        n_z = len(df)
        print(f'[C1] {n_z} word components, {int(df.graphic.sum())} graphic (dashes/rules) excluded from scoring')
        cols = ['x0', 'y0', 'x1', 'y1', 'l', 'off', 'angle_dev', 'asc_excess', 'gap_dev', 'stroke_w', 'ink_level', 'ink_noise', 'noise_level', 'ela_level', 'sharpness', 'score', 'reason']
        print('[C1/C2] top-10 most anomalous words:')
        try:
            display(df.sort_values('score', ascending=False)[cols].head(10).round(3))
        except NameError:
            print(df.sort_values('score', ascending=False)[cols].head(10).round(3).to_string())
        top = df.sort_values('score', ascending=False).head(8).index
        fig, ax = plt.subplots(1, 2, figsize=(18, 4.5))
        contrib.loc[top].plot.barh(stacked=True, ax=ax[0], colormap='tab10')
        ax[0].invert_yaxis(); ax[0].set_title('C2 per-feature evidence  w*z^2  for the top words (word id)')
        ax[0].legend(fontsize=7, ncol=2)
        ax[1].hist(df.score, bins=60, color='gray'); ax[1].axvline(cfg['word_thr'], color='r', ls='--', label='flag threshold')
        ax[1].set_yscale('log'); ax[1].legend(); ax[1].set_title('C2 distribution of word anomaly scores')
        plt.tight_layout(); plt.show()
        zv = to_rgb(g_r)
        for i, r in df.iterrows():
            v = float(np.clip(r.score / (1.5 * cfg['word_thr']), 0, 1))
            col = (int(255 * v), int(200 * (1 - v)), 0)
            cv2.rectangle(zv, (int(r.x0), int(r.y0)), (int(r.x1), int(r.y1)), col, 2)
        cmv = to_rgb(g_r)
        for a, b, mx, c in pairs:
            ra, rb = df.loc[a], df.loc[b]
            for rr in (ra, rb):
                cv2.rectangle(cmv, (int(rr.x0), int(rr.y0)), (int(rr.x1), int(rr.y1)), (255, 0, 255), 3)
            cv2.line(cmv, (int((ra.x0 + ra.x1) / 2), int((ra.y0 + ra.y1) / 2)), (int((rb.x0 + rb.x1) / 2), int((rb.y0 + rb.y1) / 2)), (255, 0, 255), 2)
        show([zv, cmv], ['C1/C2 word consistency score (green=consistent, red=anomalous)', f'C3 copy-move pairs: {len(pairs)}'], cols=2, width=8)
        final = to_rgb(gray)
        hm = cv2.applyColorMap(np.clip(heat / (2 * cfg['word_thr']) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        over = cv2.addWeighted(final, 0.6, hm, 0.4, 0)
        out = to_rgb(gray)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 0, 0), 3)
        for i, r in df[df.flag].iterrows():
            p = np.array([[r.x0, r.y0]], np.float32) @ Minv[:, :2].T + Minv[:, 2]
            cv2.putText(out, r.reason, (int(p[0, 0]), max(12, int(p[0, 1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1)
        show([over, out], ['C4 fused tampering heat-map', f'C4 VERDICT: score={img_score:.2f} -> {"TAMPERED" if img_score > cfg["word_thr"] else "no evidence"}'],
             cols=2, width=8, suptitle=title or None)
        print(f'pipeline time: {R["time"]:.1f}s')
    return R

# %% [markdown]
# ### Constant-false-alarm-rate (CFAR) calibration
# Thresholds are not hand-tuned: we run the pipeline on *genuine* scans and set the word threshold at the 99th percentile
# of genuine word scores (~1 false flag per page) and the block threshold just above the cleanest genuine block.

# %%
def calibrate(n=6, seed=99, q=0.99):
    rng_c = np.random.default_rng(seed)
    ws, bs = [], []
    for i in range(n):
        scan, _ = make_document(['contract', 'receipt', 'invoice'][i % 3], rng_c)
        r = run_pipeline(jpeg(scan, rng_c.integers(80, 96)), verbose=False)
        ws += list(r['df'].score.values)
        bs.append(max_clean_z(r['noise']['z_map'], r['noise']['win']))
    print('cleanest genuine background patch per page (z):', np.round(bs, 1))
    PIPE['word_thr'] = float(max(1.5, np.quantile(ws, q)))
    PIPE['block_thr'] = float(max(5.0, np.max(bs) + 0.25))
    plt.figure(figsize=(8, 3)); plt.hist(ws, bins=80, color='gray'); plt.yscale('log')
    plt.axvline(PIPE['word_thr'], color='r', ls='--', label=f"word_thr={PIPE['word_thr']:.2f}"); plt.legend()
    plt.title(f'Genuine word scores over {n} clean pages'); plt.show()
    print('calibrated:', {k: round(v, 2) for k, v in PIPE.items()})


calibrate(6 if not IN_COLAB else 9)

# %% [markdown]
# ## 5. Run the full pipeline step-by-step on a forged contract

# %%
R = run_pipeline(demo['contract']['forged'], verbose=True, title='Forged contract')
print('ground truth:', [(r['type'], r['box']) for r in demo['contract']['recs']])

# %% [markdown]
# ### Same pipeline on a forged receipt (mono font, narrow, dashed separators) and an invoice (dense table grid)

# %%
R_rec = run_pipeline(demo['receipt']['forged'], verbose=True, title='Forged receipt')
print('ground truth:', [(r['type'], r['box']) for r in demo['receipt']['recs']])

# %%
R_inv = run_pipeline(demo['invoice']['forged'], verbose=True, title='Forged invoice')
print('ground truth:', [(r['type'], r['box']) for r in demo['invoice']['recs']])

# %% [markdown]
# ## 6. Ablation - why adaptive morphology? (word segmentation on mixed font sizes)
# Word detection F1 (IoU >= 0.5 against renderer ground truth) for (a) a fixed tutorial kernel 15x3, (b) a kernel scaled to
# the *global* char height, (c) our *locally adaptive, data-learned* kernel.

# %%
def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_f1(pred, gt, thr=0.5):
    used, tp = set(), 0
    for g in gt:
        best, bj = 0, -1
        for j, p in enumerate(pred):
            if j in used or p[2] < g[0] or p[0] > g[2] or p[3] < g[1] or p[1] > g[3]:
                continue
            v = box_iou(g, p)
            if v > best:
                best, bj = v, j
        if best >= thr:
            used.add(bj); tp += 1
    pr, rc = tp / max(1, len(pred)), tp / max(1, len(gt))
    return 2 * pr * rc / max(1e-9, pr + rc)


def segment_variants(gray):
    char_h, _ = estimate_char_height(gray)
    norm, _ = normalize_illumination(gray, char_h)
    smap, levels, bin_map, _ = build_scale_map(norm, char_h)
    bw, _ = adaptive_binarize(norm, levels, bin_map)
    text, *_ = extract_rules(bw, char_h, levels)
    ratios, _, _ = learn_gap_ratios(text, bin_map, levels)
    ad, _, _ = detect_words(text, bin_map, levels, ratios, smap)
    return {'fixed 15x3': detect_words_fixed(text, 15)[0],
            'global-scale': detect_words_fixed(text, odd(0.45 * char_h, 3), odd(0.1 * char_h))[0],
            'local-adaptive (ours)': [w['box'] for w in ad if not w['graphic']]}


abl = []
rng_a = np.random.default_rng(11)
for kind in ['contract', 'receipt', 'invoice']:
    for _ in range(3):
        scan, gw = make_document(kind, rng_a, angle=0.0)
        gt_boxes = [w['box'] for w in gw]
        for name, boxes in segment_variants(scan).items():
            abl.append(dict(doc=kind, method=name, f1=match_f1(boxes, gt_boxes)))
abl = pd.DataFrame(abl)
piv = abl.pivot_table(index='doc', columns='method', values='f1', aggfunc='mean')
print(piv.round(3))
piv.plot.bar(figsize=(9, 3.5), rot=0, title='Word segmentation F1 by kernel strategy'); plt.ylim(0, 1); plt.show()

# %% [markdown]
# ## 7. Quantitative evaluation on the synthetic benchmark
# * **Image level**: ROC-AUC of the page score (tampered vs genuine scans).
# * **Region level**: a forgery counts as detected if flagged pixels cover >= 30% of its GT region; reported per attack type.
# * **Pixel level**: F1 / IoU of the flagged mask, pixel-AUC of the heat-map.

# %%
def region_hits(mask_pred, recs, gt):
    out = []
    for r in recs:
        x0, y0, x1, y1 = r['box']
        g = gt[y0:y1, x0:x1] > 0
        cov = (mask_pred[y0:y1, x0:x1][g] > 0).mean() if g.any() else 0
        out.append(dict(type=r['type'], hit=cov >= 0.3, coverage=cov))
    return out


def evaluate_set(n_per_kind=6, seed=123, bases=()):
    rng_e = np.random.default_rng(seed)
    img_rows, reg_rows, pix = [], [], []
    jobs = [(k, None) for k in ['contract', 'receipt', 'invoice'] for _ in range(n_per_kind)] + [(n, g) for n, g in bases]
    for kind, base in jobs:
        if base is None:
            scan, words = make_document(kind, rng_e)
        else:
            scan = base
            r0 = run_pipeline(scan, verbose=False)
            Minv = cv2.invertAffineTransform(r0['M'])
            words = []
            for w in r0['words']:
                if w['graphic']:
                    continue
                p = np.array([[w['box'][0], w['box'][1]], [w['box'][2], w['box'][3]]], np.float32) @ Minv[:, :2].T + Minv[:, 2]
                words.append(dict(text='0' if rng_e.random() < 0.3 else 'x', box=[int(p[0, 0]), int(p[0, 1]), int(p[1, 0]), int(p[1, 1])]))
        clean = jpeg(scan, rng_e.integers(80, 96))  # same double-compression as forgeries -> fair negative
        forged, gt, recs = tamper(scan, words, rng_e)
        if not recs:
            continue
        rc = run_pipeline(clean, verbose=False)
        rf = run_pipeline(forged, verbose=False)
        img_rows += [dict(doc=kind, label=0, score=rc['img_score'], false_flags=int(rc['df'].flag.sum())),
                     dict(doc=kind, label=1, score=rf['img_score'], false_flags=np.nan)]
        for h in region_hits(rf['mask'], recs, gt):
            reg_rows.append(dict(doc=kind, **h))
        p, g = rf['mask'].ravel() > 0, gt.ravel() > 0
        tp = (p & g).sum()
        hv = rf['heat'][::4, ::4].ravel(); gg = gt[::4, ::4].ravel()
        pix.append(dict(doc=kind, f1=2 * tp / max(1, p.sum() + g.sum()), iou=tp / max(1, (p | g).sum()),
                        auc=roc_auc_score(gg, hv) if 0 < gg.sum() < len(gg) else np.nan))
    return pd.DataFrame(img_rows), pd.DataFrame(reg_rows), pd.DataFrame(pix)


N_PER_KIND = 6 if IN_COLAB else 4
t0 = time.time()
img_df, reg_df, pix_df = evaluate_set(N_PER_KIND, bases=hf_bases)
print(f'evaluated {len(img_df) // 2} forged/genuine pairs in {time.time() - t0:.0f}s')
auc = roc_auc_score(img_df.label, img_df.score)
fpr, tpr, _ = roc_curve(img_df.label, img_df.score)
print(f'Image-level ROC-AUC = {auc:.3f}')
print('Region detection rate by attack type:\n', reg_df.groupby('type').hit.mean().round(3))
print('Pixel-level (forged pages):\n', pix_df.groupby('doc')[['f1', 'iou', 'auc']].mean().round(3))
print('Mean false-flagged words per GENUINE page:', round(img_df[img_df.label == 0].false_flags.mean(), 2))
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ax[0].plot(fpr, tpr, lw=2); ax[0].plot([0, 1], [0, 1], 'k--'); ax[0].set_title(f'Image-level ROC (AUC={auc:.3f})')
ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR')
reg_df.groupby('type').hit.mean().plot.bar(ax=ax[1], rot=0, color='teal', title='Detection rate per attack'); ax[1].set_ylim(0, 1)
for lab, c in [(0, 'green'), (1, 'red')]:
    ax[2].hist(img_df[img_df.label == lab].score, bins=20, alpha=0.6, color=c, label=['genuine', 'forged'][lab])
ax[2].axvline(PIPE['word_thr'], color='k', ls='--'); ax[2].legend(); ax[2].set_title('Page score distribution')
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Real forged receipts - Find-it-again (L3i, ICDAR 2023)
# 988 SROIE receipts, 163 forged by people with real editing tools. Set `USE_FINDIT=True` to download
# (`http://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip`). The loader inspects the archive and auto-detects the image /
# label / region columns; it prints what it found so you can adapt column names if the release format differs.

# %%
USE_FINDIT = False
FINDIT_URL = 'http://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip'


def parse_regions(val):
    """Accepts VIA-style JSON ({'regions':[{'shape_attributes':{x,y,width,height}}]}) or a list of dicts; returns boxes."""
    boxes = []
    if not isinstance(val, str) or '{' not in val:
        return boxes
    try:
        obj = json.loads(val.replace("'", '"'))
    except Exception:
        return boxes
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            sa = o.get('shape_attributes', o)
            if all(k in sa for k in ('x', 'y', 'width', 'height')):
                boxes.append((int(sa['x']), int(sa['y']), int(sa['x'] + sa['width']), int(sa['y'] + sa['height'])))
            elif 'all_points_x' in sa:
                xs, ys = sa['all_points_x'], sa['all_points_y']
                boxes.append((min(xs), min(ys), max(xs), max(ys)))
            else:
                stack += list(o.values())
        elif isinstance(o, list):
            stack += o
    return boxes


def load_findit(root, split='test', limit=60):
    files = glob.glob(os.path.join(root, '**', '*'), recursive=True)
    tables = [f for f in files if f.lower().endswith(('.txt', '.csv')) and split in os.path.basename(f).lower()]
    print('tables found:', tables[:5])
    imgs = {os.path.basename(f): f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    for t in tables:
        try:
            df = pd.read_csv(t)
        except Exception as e:
            print('cannot read', t, e); continue
        print('columns:', list(df.columns))
        img_col = next((c for c in df.columns if df[c].astype(str).str.contains(r'\.(png|jpg|jpeg)$', case=False).mean() > 0.5), None)
        lab_col = next((c for c in df.columns if 'forg' in c.lower() and set(pd.unique(df[c].dropna())) <= {0, 1, True, False}), None)
        reg_col = next((c for c in df.columns if df[c].astype(str).str.contains('shape_attributes|width', regex=True).mean() > 0.05), None)
        print('-> image:', img_col, '| label:', lab_col, '| regions:', reg_col)
        if img_col is None or lab_col is None:
            continue
        pos, neg = df[df[lab_col] == 1], df[df[lab_col] == 0]
        sel = pd.concat([pos.head(limit // 2), neg.head(limit // 2)])
        out = []
        for _, r in sel.iterrows():
            p = imgs.get(os.path.basename(str(r[img_col])))
            if p:
                out.append(dict(path=p, label=int(r[lab_col]), boxes=parse_regions(r[reg_col]) if reg_col else []))
        return out
    return []


if USE_FINDIT:
    zp = os.path.join(WORK, 'findit2.zip')
    if not os.path.exists(zp):
        print('downloading Find-it-again ...'); urllib.request.urlretrieve(FINDIT_URL, zp)
    root = os.path.join(WORK, 'findit2')
    if not os.path.exists(root):
        zipfile.ZipFile(zp).extractall(root)
    items = load_findit(root, 'test', limit=60)
    print(len(items), 'receipts selected')
    if items:
        it = next(i for i in items if i['label'] == 1)
        rimg = cv2.cvtColor(cv2.imread(it['path']), cv2.COLOR_BGR2RGB)
        Rr = run_pipeline(rimg, verbose=True, title='Find-it-again forged receipt')
        gtv = to_rgb(to_gray(rimg))
        for b in it['boxes']:
            cv2.rectangle(gtv, b[:2], b[2:], (255, 0, 0), 3)
        show([gtv, Rr['heat']], ['official forgery annotation', 'our heat-map'], cols=2, width=7, cmaps=[None, 'jet'])
        rows = []
        for it in items:
            rr = run_pipeline(cv2.cvtColor(cv2.imread(it['path']), cv2.COLOR_BGR2RGB), verbose=False)
            hit = np.nan
            if it['boxes']:
                hit = any(rr['mask'][b[1]:b[3], b[0]:b[2]].mean() > 0.1 for b in it['boxes'])
            rows.append(dict(label=it['label'], score=rr['img_score'], region_hit=hit))
        fd = pd.DataFrame(rows)
        print(f"Find-it-again image-level AUC = {roc_auc_score(fd.label, fd.score):.3f}  "
              f"(n={len(fd)}), region hit-rate on forged = {fd[fd.label == 1].region_hit.mean():.3f}")

# %% [markdown]
# ## 9. Try your own document
# Upload a scan / photo of a contract, receipt or invoice. Every intermediate step is plotted.

# %%
if IN_COLAB:
    from google.colab import files
    up = files.upload()
    for name, data in up.items():
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        run_pipeline(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), verbose=True, title=name)
