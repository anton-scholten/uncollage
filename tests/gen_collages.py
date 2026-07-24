#!/usr/bin/env python3
"""
Deterministic synthetic collages for the uncollage.py regression suite.

Each builder returns one BGR image (a "scanned page") that exercises a distinct
scenario uncollage.py is meant to handle. The tiles are textured (gradient +
shapes + grain) so they clear the background/texture thresholds and the
line-splitter's uniformity checks -- flat colour blocks would be wrongly treated
as background or dropped as "too uniform", which is not representative of real
photos.

Run this file directly to dump the PNGs into tests/collages/ for eyeballing:

    python tests/gen_collages.py

test_uncollage.py imports SCENARIOS and builds the images in memory.
"""
import os

import cv2
import numpy as np

PAPER = np.array([245, 244, 238], np.float32)   # near-white scanner paper


def _paper_bg(rng, h, w):
    bg = np.empty((h, w, 3), np.float32)
    bg[:] = PAPER
    bg += rng.normal(0, 3, (h, w, 3))            # faint scanner noise
    return np.clip(bg, 0, 255).astype(np.uint8)


def _photo(rng, h, w):
    """A textured 'photograph': gradient base + random shapes + film grain."""
    base = rng.integers(40, 210, 3).astype(np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = base + np.linspace(-40, 40, w)[None, :, None] \
                  + np.linspace(-30, 30, h)[:, None, None]
    tmp = img.copy()
    for _ in range(int(rng.integers(6, 12))):
        col = rng.integers(0, 255, 3).tolist()
        x1, x2 = sorted(rng.integers(0, w, 2))
        y1, y2 = sorted(rng.integers(0, h, 2))
        if rng.random() < 0.5:
            cv2.rectangle(tmp, (int(x1), int(y1)), (int(x2), int(y2)), col, -1)
        else:
            cv2.circle(tmp, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                       int(rng.integers(10, max(11, min(h, w) // 3))), col, -1)
    img = 0.55 * img + 0.45 * tmp
    img += rng.normal(0, 8, (h, w, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _faint_scene(rng, h, w):
    """Two barely-there bands ~ paper tone: a borderless faded photo."""
    img = np.empty((h, w, 3), np.float32)
    img[:h // 2] = PAPER - 6
    img[h // 2:] = PAPER - 10
    img += rng.normal(0, 7, (h, w, 3))
    for _ in range(8):
        col = rng.integers(0, 255, 3).tolist()
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))),
                   int(rng.integers(6, 22)), col, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


# --- scenario builders (each seeds its own RNG => order-independent) ----------
def separated():
    """4 photos with clear paper gaps -> plain mode finds 4."""
    rng = np.random.default_rng(1)
    H, W = 900, 1200
    page = _paper_bg(rng, H, W)
    ph, pw, gap, m = 360, 520, 60, 40
    for ry, rx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        y, x = m + ry * (ph + gap), m + rx * (pw + gap)
        page[y:y + ph, x:x + pw] = _photo(rng, ph, pw)
    return page


def rotated():
    """3 tilted photos -> plain mode finds 3 (each cropped as upright bbox)."""
    rng = np.random.default_rng(2)
    H, W = 950, 1250
    page = _paper_bg(rng, H, W)
    specs = [(300, 400, 18, 330, 230), (300, 360, -15, 900, 260),
             (300, 430, 8, 560, 660)]
    for ph, pw, ang, cx, cy in specs:
        tile = _photo(rng, ph, pw)
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), ang, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(ph * sin + pw * cos), int(ph * cos + pw * sin)
        M[0, 2] += nw / 2 - pw / 2
        M[1, 2] += nh / 2 - ph / 2
        rot = cv2.warpAffine(tile, M, (nw, nh))
        # composite only the photo pixels; the rotated corners keep the paper,
        # so no black rectangle is pasted onto the background.
        mask = cv2.warpAffine(np.full((ph, pw), 255, np.uint8), M, (nw, nh)) > 127
        y0, x0 = cy - nh // 2, cx - nw // 2
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(H, y0 + nh), min(W, x0 + nw)
        sub = rot[ys - y0:ye - y0, xs - x0:xe - x0]
        msub = mask[ys - y0:ye - y0, xs - x0:xe - x0]
        page[ys:ye, xs:xe][msub] = sub[msub]
    return page


def touching():
    """2 photos butted with a dark shadow seam -> needs -r to split into 2."""
    rng = np.random.default_rng(3)
    H, W = 800, 1200
    page = _paper_bg(rng, H, W)
    ph, pw, m = 560, 540, 40
    page[m:m + ph, m:m + pw] = _photo(rng, ph, pw)
    x1 = m + pw
    page[m:m + ph, x1:x1 + 6] = (40, 38, 36)            # shadow ridge
    page[m:m + ph, x1 + 6:x1 + 6 + pw] = _photo(rng, ph, pw)
    return page


def grid():
    """2x3 grid of touching photos, thin dark seams -> needs -r to split into 6."""
    rng = np.random.default_rng(4)
    H, W = 900, 1300
    page = _paper_bg(rng, H, W)
    rows, cols, m, seam = 2, 3, 30, 6
    ph = (H - 2 * m - (rows - 1) * seam) // rows
    pw = (W - 2 * m - (cols - 1) * seam) // cols
    for r in range(rows):
        for c in range(cols):
            y, x = m + r * (ph + seam), m + c * (pw + seam)
            page[y:y + ph, x:x + pw] = _photo(rng, ph, pw)
    for r in range(1, rows):
        y = m + r * ph + (r - 1) * seam
        page[y:y + seam, m:W - m] = (38, 36, 34)
    for c in range(1, cols):
        x = m + c * pw + (c - 1) * seam
        page[m:H - m, x:x + seam] = (38, 36, 34)
    return page


def composite():
    """2 distinct photos butted with NO gap/seam -> needs -l to split into 2."""
    rng = np.random.default_rng(5)
    H, W = 900, 700
    page = _paper_bg(rng, H, W)
    ph, pw, m = 820, 620, 40
    tile = np.vstack([_photo(rng, ph // 2, pw), _photo(rng, ph - ph // 2, pw)])
    page[m:m + ph, m:m + pw] = tile
    return page


def hard():
    """A faded borderless photo whose tone ~ paper -> queued for the manual fallback."""
    rng = np.random.default_rng(6)
    H, W = 700, 700
    page = _paper_bg(rng, H, W)
    ph, pw, m = 600, 600, 50
    page[m:m + ph, m:m + pw] = _faint_scene(rng, ph, pw)
    return page


# name, builder, uncollage flags, expected sub-image count, needs-manual?
SCENARIOS = [
    ("separated", separated, {}, 4, False),
    ("rotated", rotated, {}, 3, False),
    ("touching", touching, {"rectangular": True}, 2, False),
    ("grid", grid, {"rectangular": True}, 6, False),
    ("composite", composite, {"split_lines": True}, 2, False),
    ("hard", hard, {}, 0, True),
]


def generate(out_dir):
    """Write every scenario's PNG into out_dir (created if needed)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, build, _flags, _exp, _man in SCENARIOS:
        p = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(p, build())
        paths.append(p)
        print("wrote", p)
    return paths


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    generate(os.path.join(here, "collages"))
