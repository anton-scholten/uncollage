#!/usr/bin/env python3
"""
Real-photograph version of the uncollage.py regression collages.

The six base scenario LAYOUTS of gen_collages.py, but every tile is a real
photograph (landscape / nature / city / people -- a typical collage mix) instead
of a synthetic pattern. The photos live in tests/photos/ and are fetched by
fetch_photos.py; this module center-crops them to each tile's aspect ratio.
It also renders them as old prints (greyscale/sepia, faded, deckled borders,
askew, shadowed) on scan pages for the old_* scenarios.

Run directly to dump the input PNGs into tests/collages_real/:

    python tests/gen_collages_real.py

test_uncollage_real.py imports SCENARIOS_REAL and builds them in memory.
"""
import glob
import os

import cv2
import numpy as np

import gen_collages as g
from gen_collages import PAPER

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(HERE, "photos")


def _pool():
    paths = sorted(glob.glob(os.path.join(PHOTO_DIR, "*.jpg")))
    imgs = [cv2.imread(p) for p in paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        raise FileNotFoundError(
            f"no photos in {PHOTO_DIR} -- run: python tests/fetch_photos.py")
    return imgs


def _crop_resize(img, h, w):
    """Center-crop img to the target aspect (w/h), then resize to (w, h)."""
    ih, iw = img.shape[:2]
    if iw / ih > w / h:                       # source too wide -> crop width
        nw = int(round(ih * w / h))
        x0 = (iw - nw) // 2
        img = img[:, x0:x0 + nw]
    else:                                     # too tall -> crop height
        nh = int(round(iw * h / w))
        y0 = (ih - nh) // 2
        img = img[y0:y0 + nh, :]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


class RealTiles:
    """A `tile(rng, h, w)` provider drawing distinct real photos in a fixed,
    reproducible order (its own seed -- independent of the layout's rng)."""

    def __init__(self, pool, seed):
        self.pool = pool
        self.order = list(np.random.default_rng(seed).permutation(len(pool)))
        self.i = 0

    def __call__(self, rng, h, w):
        idx = self.order[self.i % len(self.order)]
        self.i += 1
        return _crop_resize(self.pool[idx], h, w)


def _faded(pool, seed):
    """`scene(rng, h, w)` -> a real photo blended most of the way to paper tone,
    an old borderless faded print whose edges are ~invisible on the page."""
    order = list(np.random.default_rng(seed).permutation(len(pool)))

    def scene(rng, h, w):
        img = _crop_resize(pool[order[0]], h, w).astype(np.float32)
        a = 0.90                              # 90% paper -> barely visible
        return np.clip(a * PAPER + (1 - a) * img, 0, 255).astype(np.uint8)

    return scene


def _tiles(seed):
    """A fresh RealTiles bound to _pool() -- deterministic per seed."""
    return RealTiles(_pool(), seed)


# --- the real-photo scenarios -------------------------------------------------
# Same layouts as the synthetic suite, but each uses the mode that actually works
# on real photographs. Two code paths are content-dependent and unreliable on
# real photos -- the -r shadow-ridge guillotine on a full grid, and -l line
# splitting on a real composite -- so those scenarios use the robust path (paper
# gaps / the manual fallback) and the weak paths are pinned as limitation
# controls in test_uncollage_real.py instead.
def r_separated():
    return g.separated(tile=_tiles(101))


def r_rotated():
    return g.rotated(tile=_tiles(102))


def r_touching():
    # butted real photos need a near-black seam (their own content is as dark as
    # a faint shadow); with it, -r reliably splits the two.
    return g.touching(tile=_tiles(103), seam_color=(12, 12, 12), seam_w=8)


def r_grid():
    # thin PAPER gaps -> plain mode separates all six (the common real collage);
    # a dark-seam butted grid does NOT fully split with -r on real photos.
    return g.grid(tile=_tiles(104), seam_color=tuple(int(v) for v in PAPER), seam=16)


def r_composite():
    # two real photos butted with no gap: default mode can't split -> queued for
    # manual (the recommended path; -l over-splits real photos, see the control).
    return g.composite(tile=_tiles(105))


def r_hard():
    return g.hard(scene=_faded(_pool(), 106))


# --- old prints on a scanner bed -----------------------------------------------
# Stock photos turned into old prints -- greyscale or sepia, faded, with a pale
# border whose edge may be deckled (the zig-zag cut of mid-century prints) --
# laid slightly askew, casting a soft shadow, on a scan page with a lighting
# gradient and the dark line a scanner lid leaves along one edge.
def _old_print(photo, rng, sepia=False, deckle=True, border=0.07):
    """(print, mask): an old-print rendering of a photo and its shape mask."""
    h, w = photo.shape[:2]
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = 0.72 * g + 38                                       # faded: lifted blacks
    img = np.dstack([g, g, g])
    if sepia:
        img *= np.array([0.80, 0.93, 1.05], np.float32)     # BGR brown tint
    b = int(border * min(h, w))
    H, W = h + 2 * b, w + 2 * b
    card = np.empty((H, W, 3), np.float32)
    card[:] = (205, 222, 232) if sepia else (222, 231, 235)  # cream / yellowed white
    card[b:b + h, b:b + w] = img
    card += rng.normal(0, 4, card.shape)                     # paper grain
    mask = np.zeros((H, W), np.uint8)
    if deckle:                                              # zig-zag edge
        a, t = max(2, b // 4), 9
        teeth = lambda n: (a * np.abs(((np.arange(n) % t) / t) * 2 - 1)).astype(int)
        top, side = teeth(W), teeth(H)
        pts = ([(x, top[x]) for x in range(W)] + [(W - 1 - side[y], y) for y in range(H)]
               + [(x, H - 1 - top[x]) for x in range(W - 1, -1, -1)]
               + [(side[y], y) for y in range(H - 1, -1, -1)])
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    else:
        mask[:] = 255
    return np.clip(card, 0, 255).astype(np.uint8), mask


def _scan_page(rng, H, W, paper=(248, 249, 249), falloff=0.10):
    """A scan page: paper with a left-to-right lighting falloff and a dark
    scanner-lid line along the top edge."""
    page = np.empty((H, W, 3), np.float32)
    page[:] = paper
    page *= np.linspace(1.0, 1.0 - falloff, W)[None, :, None]
    page += rng.normal(0, 2.5, page.shape)
    page[:5] = 20
    return page


def _lay(page, card, mask, cx, cy, angle, shadow=(6, 8)):
    """Place a print centred at (cx, cy), rotated by angle, with a soft shadow.
    Returns the print's true bounding box (x, y, w, h) on the page."""
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rc = cv2.warpAffine(card, M, (nw, nh))
    rm = cv2.warpAffine(mask, M, (nw, nh)).astype(np.float32) / 255
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    sh = cv2.GaussianBlur(np.roll(rm, shadow, (0, 1)), (0, 0), 4)
    region = page[y0:y0 + nh, x0:x0 + nw]
    region *= (1 - 0.35 * sh)[..., None]
    region[:] = region * (1 - rm[..., None]) + rc * rm[..., None]
    ys, xs = np.nonzero(rm > 0.5)
    return (x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)


def _old_layout(seed, H, W, specs, paper=(248, 249, 249), falloff=0.10, **kw):
    """Lay out old prints per specs [(h, w, cx, cy, angle), ...]; returns the
    page and the prints' true boxes."""
    rng = np.random.default_rng(seed)
    pool, page = _pool(), _scan_page(rng, H, W, paper, falloff)
    order = np.random.default_rng(seed + 1).permutation(len(pool))
    truth = []
    for k, (h, w, cx, cy, ang) in enumerate(specs):
        card, mask = _old_print(_crop_resize(pool[order[k % len(pool)]], h, w), rng, **kw)
        truth.append(_lay(page, card, mask, cx, cy, ang))
    return np.clip(page, 0, 255).astype(np.uint8), truth


def old_scan_truth():
    """5 greyscale deckled prints with yellowed-white borders, slightly askew,
    on white scanner paper with a mild lighting falloff. (Where the falloff
    makes the paper exactly the border's tone the border is invisible and is
    lost -- a limit no background method can beat.)"""
    return _old_layout(201, 1400, 1000, [
        (330, 250, 230, 250, 2.0), (250, 330, 700, 230, -3.0),
        (360, 270, 240, 720, -1.5), (270, 360, 710, 700, 3.5),
        (250, 330, 480, 1170, 1.0)], falloff=0.04)


def old_sepia_truth():
    """4 sepia prints with plain cream borders on dark-grey album card with a
    strong lighting falloff -- background colour is not white."""
    return _old_layout(202, 1000, 1300, [
        (330, 250, 240, 260, -2.0), (300, 420, 820, 250, 1.5),
        (300, 420, 400, 740, 2.5), (330, 250, 1020, 730, -3.0)],
        paper=(95, 98, 100), falloff=0.30, sepia=True, deckle=False)


def old_corner_truth():
    """2 greyscale deckled prints whose corners meet, bridged by shadow."""
    return _old_layout(203, 1000, 1000, [
        (360, 280, 300, 290, 0.0), (360, 280, 600, 690, 0.0)])


def old_scan():
    return old_scan_truth()[0]


def old_sepia():
    return old_sepia_truth()[0]


def old_corner():
    return old_corner_truth()[0]


# name, builder, uncollage flags, expected count, needs-manual?
SCENARIOS_REAL = [
    ("separated", r_separated, {}, 4, False),
    ("rotated", r_rotated, {}, 3, False),
    ("touching", r_touching, {"rectangular": True}, 2, False),
    ("grid", r_grid, {}, 6, False),
    ("composite", r_composite, {}, 0, True),
    ("hard", r_hard, {}, 0, True),
    ("old_scan", old_scan, {}, 5, False),
    ("old_sepia", old_sepia, {}, 4, False),
    ("old_corner", old_corner, {}, 2, False),
]

# Old-print layouts whose true print boxes are known, for the "crop holds the
# whole print" check.
OLD_TRUTH = [("old_scan", old_scan_truth), ("old_sepia", old_sepia_truth),
             ("old_corner", old_corner_truth)]


# --- limitation cases (used as negative controls) -----------------------------
def grid_butted_dark():
    """A grid of butted real photos with a dark shadow seam -- the -r guillotine
    does NOT fully recover all six on real content (caps well below 6)."""
    return g.grid(tile=_tiles(104), seam_color=(0, 0, 0), seam=12)


def generate(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, build, _flags, _exp, _man in SCENARIOS_REAL:
        p = os.path.join(out_dir, f"{name}.png")
        cv2.imwrite(p, build())
        paths.append(p)
        print("wrote", p)
    return paths


if __name__ == "__main__":
    generate(os.path.join(HERE, "collages_real"))
