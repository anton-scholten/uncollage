#!/usr/bin/env python3
"""
uncollage.py -- Separate an image collage / scanned page into individual sub-images.

Usage:
    python uncollage.py [--rectangular] PATH

  * If PATH is an image file, its sub-images are extracted.
  * If PATH is a text file, every line is treated as an image path and all of
    them are processed in batch. Blank lines and lines starting with '#' are
    ignored. Relative paths are resolved against the text file's own folder.

  * --rectangular (-r): assume every sub-image is a rectangle and additionally
    use line detection to split merged blobs (see "Rectangular mode" below).
  * --ai (-a): like --rectangular, but the seams are found by a small offline
    neural net (seam_ai.py, weights in seam_model.npz) instead of only the
    hand-crafted cues. Train it once with `python seam_ai.py train`.
  * --split-lines (-l): also split a blob wherever a full-span STRAIGHT content
    boundary shows it is several butted-together photos with no gap between them
    (e.g. touching postcards). OFF by default and opt-in on purpose: a photo's
    own straight internal line -- a horizon, a building edge -- is
    indistinguishable from a real print seam and gets split too, so this can
    over-split ordinary landscapes. Use it only on scans you know are composites;
    for anything ambiguous prefer the manual-boxes fallback below.

Manual-boxes fallback (automatic):
  Some pages cannot be split automatically (e.g. faded, borderless photos on
  shadowed paper, whose edges are not visible in the scan). Every input image is
  processed first; any page whose automatic split looks merged/failed is queued.
  After the batch, uncollage opens a box editor (draw_boxes.py) for each queued
  page in turn -- drag one rough box per photo (s/Enter to save, q/Esc to skip)
  -- and its sub-images are cropped immediately from the boxes you draw. The
  boxes are cached in a manual-boxes file (default manual_boxes.txt, override
  with --manual FILE) so a later run reuses them without asking again.

  --no-draw (or a headless session with no display) skips the editor and just
  lists the queued pages in the manual-boxes file; draw them later with
  `python draw_boxes.py manual_boxes.txt` and re-run.

Output sub-images are written next to each source image, named
    <stem>_1.<ext>, <stem>_2.<ext>, ...
(numbered top-to-bottom, then left-to-right).

Design notes / guarantees:
  * Sub-images may touch each other and may touch the edge of the original
    image -- detection is based on the uniform scanner/paper background, not on
    an assumed margin.
  * Sub-images may be rotated. Each is cropped as the UPRIGHT bounding box of
    its (possibly tilted) region, so the crop keeps the sub-image in its
    original orientation -- it is NOT de-skewed / un-rotated. A rotated
    sub-image therefore keeps small background corners, as expected.

Method:
  The background is the large, uniform, low-texture region that the scanner/
  paper forms around the photos. A pixel is treated as background when its
  colour is close to the estimated background colour AND its local texture is
  low; the background proper is the part of that connected to the image border
  (flood fill). Everything else is foreground. Each sufficiently large,
  rectangle-like foreground blob becomes one sub-image.

Rectangular mode (--rectangular):
  When two rectangular photos touch, the background gap / light border / dark
  shadow between them is a straight line. This mode detects such a straight
  separator and recursively cuts each foreground blob along it (a "guillotine"
  split), so touching photos that the plain method would merge -- e.g. where a
  border seam is bridged at a point -- are separated. A candidate cut is a full
  span horizontal or vertical line most of whose length runs along the
  background OR along a dark shadow ridge, and which actually severs the blob
  into two large parts. Cuts are axis-aligned, so this best suits photos that
  are upright or only slightly rotated.

Limitation:
  Photos that physically overlap, or butt together with no separator at all
  (no gap, border, or tonal edge) between them, cannot be told apart by any
  background/line based method and may be emitted as a single sub-image. Cuts
  in rectangular mode are horizontal/vertical, so strongly rotated touching
  photos are not split.

Requires: opencv-python (cv2), numpy.
"""

import os
import sys

import cv2
import numpy as np

# Image extensions recognised when deciding "an image" vs "a list of paths".
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --- Tunables -----------------------------------------------------------------
MIN_AREA_FRAC = 0.01     # blob must be >= this fraction of the page to count
COLOR_TOL = 30.0         # max colour distance from background to be "background"
TEXTURE_TOL = 12.0       # max local std-dev to be "background" (smooth)
FILL_RATIO_MIN = 0.55    # blob area / its rotated-bounding-rect area (rejects
                         # thin frames / speckle; a real photo is a full quad)
# --- Rectangular-mode (guillotine split) tunables -----------------------------
SEAM_THRESH = 0.70       # min fraction of a cut line that must look like a seam
SEAM_THRESH_AI = 0.55    # ... when seams come from the learned model (softer maps)
SEAM_MIN_SPAN = 0.85     # a cut must span >= this fraction of the blob's extent
SEAM_RIDGE_MARGIN = 14   # a shadow ridge must be this much darker than its sides
SEAM_MAX_DEPTH = 6       # max recursive cuts per blob
AI_PROB_THRESH = 0.30    # learned seam probability above which a pixel is a seam
# --- Manual-boxes fallback ----------------------------------------------------
MANUAL_FILE_DEFAULT = "manual_boxes.txt"  # where failed pages are listed
MANUAL_SPAN = 0.80          # a box spanning >= this of BOTH page dims => failed
MANUAL_DOMINANT_AREA = 0.55 # a blob >= this fraction of the page ...
MANUAL_DOMINANT_RATIO = 3.0 # ... and >= this x the next largest => it merged some
# --- Straight-line composition split ------------------------------------------
LINE_COV_THRESH = 0.80   # a full-span content boundary crosses >= this of the box
LINE_JUMP = 30.0         # per-column colour distance counted as an edge crossing
LINE_MARGIN = 0.10       # never cut within this fraction of a box edge
LINE_MIN_PIECE = 0.15    # each resulting piece >= this fraction of the box extent
LINE_MIN_FG = 0.50       # both sides of a cut must be at least this much foreground
LINE_STRAIGHT_MAX = 6.0  # max px scatter of the edge along the line (real print
                         # edges are straight; ragged internal content is not)
LINE_MAX_DEPTH = 5       # max recursive line cuts per blob
# ------------------------------------------------------------------------------


def estimate_background(img):
    """Estimate the background colour as the image's dominant (modal) colour.

    The scanner/paper background is the single largest uniform area, so the
    most frequent colour is a robust estimate -- unlike a border average, it is
    not fooled by photos that run off the edge of the page.
    """
    # Coarse 3-D colour histogram; the fullest bin is the background.
    bins = 16
    step = 256 // bins
    q = (img // step).reshape(-1, 3)
    codes = (q[:, 0].astype(np.int32) * bins + q[:, 1]) * bins + q[:, 2]
    peak = np.bincount(codes, minlength=bins ** 3).argmax()
    # Average the actual pixels in that bin for a precise colour.
    sel = codes == peak
    return img.reshape(-1, 3)[sel].astype(np.float32).mean(axis=0)


def foreground_mask(img):
    """Return a binary mask (uint8 0/255) whose blobs are the sub-images."""
    h, w = img.shape[:2]
    bg = estimate_background(img)

    # Colour distance from the background colour.
    diff = img.astype(np.float32) - bg[None, None, :]
    dist = np.sqrt((diff * diff).sum(axis=2))

    # Local texture (rolling standard deviation). Kept small so that thin
    # background gaps between photos are not swallowed by neighbouring detail.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = max(3, int(round(min(h, w) * 0.003)) | 1)
    mean = cv2.boxFilter(gray, -1, (win, win), normalize=True)
    sqmean = cv2.boxFilter(gray * gray, -1, (win, win), normalize=True)
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0))

    # Background = close to the background colour AND smooth.
    bg_like = ((dist < COLOR_TOL) & (std < TEXTURE_TOL)).astype(np.uint8)

    # The true background is the bg-like region connected to the image border;
    # bg-like pockets *enclosed* by a photo (sky, snow, pale walls) are kept as
    # foreground, which is what we want.
    n, labels = cv2.connectedComponents(bg_like, connectivity=8)
    border_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :])) \
        | set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    border_labels.discard(0)
    background = np.isin(labels, list(border_labels))
    fg = (~background).astype(np.uint8) * 255

    # Drop dust/speckle. Kernel stays small to preserve gaps between photos.
    k = max(3, int(round(min(h, w) * 0.004)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    return fg


def _seam_maps(img):
    """Return two boolean maps marking pixels that look like a photo boundary.

    seam_h -- suitable for a HORIZONTAL cut: background, or a dark ridge that is
              darker than the pixels a little above and below it (a shadow line
              between a photo above and a photo below).
    seam_v -- the same for a VERTICAL cut (compares left/right neighbours).
    """
    h, w = img.shape[:2]
    bg = estimate_background(img)
    diff = img.astype(np.float32) - bg[None, None, :]
    bg_like = np.sqrt((diff * diff).sum(axis=2)) < COLOR_TOL

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    off = max(4, int(round(min(h, w) * 0.012)))
    m = SEAM_RIDGE_MARGIN
    ridge_h = (gray < np.roll(gray, -off, 0) - m) & (gray < np.roll(gray, off, 0) - m)
    ridge_v = (gray < np.roll(gray, -off, 1) - m) & (gray < np.roll(gray, off, 1) - m)
    return bg_like | ridge_h, bg_like | ridge_v


def _best_cut(mask, seam_h, seam_v, seam_thresh):
    """Find the best full-span horizontal or vertical seam through a blob.

    Returns (orientation, position, lo, hi) or None. The line must run mostly
    along seam pixels and span most of the blob's width/height.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    span_w, span_h = x1 - x0 + 1, y1 - y0 + 1

    best = None
    best_score = seam_thresh

    # Horizontal candidates: score every row by the seam fraction of its
    # foreground pixels; keep rows that are wide enough and not at the very edge.
    sub = mask[:, x0:x1 + 1] > 0
    fg_per_row = sub.sum(axis=1).astype(np.float32)
    seam_per_row = (sub & seam_h[:, x0:x1 + 1]).sum(axis=1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(fg_per_row > 0, seam_per_row / fg_per_row, 0.0)
    margin = int(0.12 * span_h)
    for r in range(y0 + margin, y1 - margin + 1):
        if fg_per_row[r] >= SEAM_MIN_SPAN * span_w and frac[r] > best_score:
            best_score, best = frac[r], ("h", r, x0, x1)

    # Vertical candidates.
    sub = mask[y0:y1 + 1, :] > 0
    fg_per_col = sub.sum(axis=0).astype(np.float32)
    seam_per_col = (sub & seam_v[y0:y1 + 1, :]).sum(axis=0).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(fg_per_col > 0, seam_per_col / fg_per_col, 0.0)
    margin = int(0.12 * span_w)
    for c in range(x0 + margin, x1 - margin + 1):
        if fg_per_col[c] >= SEAM_MIN_SPAN * span_h and frac[c] > best_score:
            best_score, best = frac[c], ("v", c, y0, y1)
    return best


def _split_rectangular(mask, seam_h, seam_v, min_area, seam_thresh, depth=0):
    """Recursively guillotine-split a blob mask into rectangular pieces."""
    if depth >= SEAM_MAX_DEPTH:
        return [mask]
    cut = _best_cut(mask, seam_h, seam_v, seam_thresh)
    if cut is None:
        return [mask]

    orient, pos, lo, hi = cut
    trial = mask.copy()
    thick = max(3, int(round(min(mask.shape) * 0.006)))
    if orient == "h":
        cv2.line(trial, (lo, pos), (hi, pos), 0, thickness=thick)
    else:
        cv2.line(trial, (pos, lo), (pos, hi), 0, thickness=thick)
    trial = cv2.morphologyEx(trial, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(trial)
    pieces = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if len(pieces) < 2:
        return [mask]  # cut did not actually separate two real photos

    out = []
    for i in pieces:
        out += _split_rectangular((labels == i).astype(np.uint8) * 255,
                                  seam_h, seam_v, min_area, seam_thresh, depth + 1)
    return out


def _blob_to_box(blob):
    """Return the upright crop box for a blob mask, or None if it isn't a photo."""
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    # Reject shapes that do not fill their (rotated) bounding rectangle -- thin
    # frames, speckle rings, stray marks. A photo fills its minAreaRect closely.
    if rw * rh == 0 or cv2.contourArea(c) / (rw * rh) < FILL_RATIO_MIN:
        return None
    # Upright bounding box keeps the sub-image in its original orientation
    # (no de-skewing), as required.
    return cv2.boundingRect(c)


def _ai_seam_maps(img):
    """Seam maps from the offline learned model (seam_ai). OR-ed with the
    hand-crafted cues so the split benefits from both."""
    import seam_ai  # local import: only needed for --ai
    prob_h, prob_v = seam_ai.predict_seam_maps(img)
    base_h, base_v = _seam_maps(img)
    return (prob_h > AI_PROB_THRESH) | base_h, (prob_v > AI_PROB_THRESH) | base_v


def _disc_coverage(region, axis):
    """For every line across `region` perpendicular to `axis`, the fraction of
    the span whose colour changes sharply from just-before to just-after -- i.e.
    how fully a straight content boundary crosses the region at that position.
    A boundary between two butted-together photos runs edge to edge, so it scores
    near 1, while most content inside a single photo does not.
    axis=0 scans rows (candidate horizontal boundary); axis=1 scans columns."""
    reg = region.astype(np.float32)
    if axis == 1:
        reg = np.transpose(reg, (1, 0, 2))
    n, m = reg.shape[0], reg.shape[1]
    k = max(8, int(0.015 * n))
    cs = np.cumsum(np.vstack([np.zeros((1, m, 3), np.float32), reg]), axis=0)
    band = lambda a, b: (cs[b] - cs[a]) / max(1, b - a)
    cov = np.zeros(n, np.float32)
    for r in range(k + 2, n - k - 2):
        d = np.sqrt(((band(r - k, r - 2) - band(r + 2, r + k)) ** 2).sum(1))
        cov[r] = (d > LINE_JUMP).mean()
    return cov


def _edge_straightness(gray, pos, axis, k=14):
    """Scatter (std, in px) of where the strongest edge actually falls along a
    candidate cut line. A real print edge is a straight line -> small scatter;
    ragged internal content (foliage meeting a building) -> large scatter."""
    n = gray.shape[axis]
    a, b = max(0, pos - k), min(n, pos + k)
    if axis == 0:
        band = gray[a:b, :]
        peak = np.abs(cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)).argmax(0)
    else:
        band = gray[:, a:b]
        peak = np.abs(cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)).argmax(1)
    return float(np.std(peak))


def _best_line_cut(region, mask_region, min_area):
    """Best full-span straight boundary in a region, or None. Returns
    (orientation 'h' | 'v', position). A cut qualifies only when it (a) is a
    full-span content discontinuity (LINE_COV_THRESH), (b) is a straight line
    (LINE_STRAIGHT_MAX) -- which tells a real print edge from ragged internal
    content of equal contrast -- and (c) leaves two large, mostly-foreground
    pieces (so it separates two photos, not a photo from paper)."""
    best, best_cov = None, LINE_COV_THRESH
    h, w = region.shape[:2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for axis, orient, extent, other in ((0, "h", h, w), (1, "v", w, h)):
        cov = _disc_coverage(region, axis)
        lo, hi = int(LINE_MARGIN * extent), extent - int(LINE_MARGIN * extent)
        for p in range(lo, hi):
            if cov[p] <= best_cov or min(p, extent - p) < LINE_MIN_PIECE * extent:
                continue
            if min(p, extent - p) * other < min_area:
                continue
            if orient == "h":
                f1, f2 = (mask_region[:p] > 0).mean(), (mask_region[p:] > 0).mean()
            else:
                f1, f2 = (mask_region[:, :p] > 0).mean(), (mask_region[:, p:] > 0).mean()
            if min(f1, f2) < LINE_MIN_FG:
                continue
            if _edge_straightness(gray, p, axis) > LINE_STRAIGHT_MAX:
                continue
            best_cov, best = cov[p], (orient, p)
    return best


def _split_blob_by_lines(img, blob, min_area, depth=0):
    """Recursively split a foreground blob wherever a full-span straight content
    boundary shows it actually holds several butted-together photos (e.g. the
    touching postcards in 69.jpg). The blob mask is cut along the line and
    re-labelled with connected components, so each photo comes out as its own
    tightly-bounded blob (no paper corners, no reliance on a gap/seam)."""
    if depth >= LINE_MAX_DEPTH:
        return [blob]
    ys, xs = np.where(blob > 0)
    if len(xs) == 0:
        return []
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    cut = _best_line_cut(img[y0:y1 + 1, x0:x1 + 1],
                         blob[y0:y1 + 1, x0:x1 + 1], min_area)
    if cut is None:
        return [blob]
    orient, p = cut
    trial = blob.copy()
    thick = max(2, int(round(0.004 * min(x1 - x0 + 1, y1 - y0 + 1))))
    if orient == "h":
        cv2.line(trial, (x0, y0 + p), (x1, y0 + p), 0, thick)
    else:
        cv2.line(trial, (x0 + p, y0), (x0 + p, y1), 0, thick)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(trial)
    pieces = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if len(pieces) < 2:
        return [blob]                       # cut did not actually separate two
    return [out for i in pieces
            for out in _split_blob_by_lines(img, (labels == i).astype(np.uint8) * 255,
                                            min_area, depth + 1)]


def find_subimages(img, rectangular=False, ai=False, split_lines=False):
    """Return a list of (x, y, w, h) upright crop boxes, one per sub-image.

    When rectangular is True, each foreground blob is additionally split along
    detected straight seams (see _split_rectangular). When ai is True, the seam
    maps come from the offline learned model in seam_ai.py (implies rectangular).
    """
    h, w = img.shape[:2]
    mask = foreground_mask(img)
    min_area = MIN_AREA_FRAC * h * w

    # Split the mask into connected foreground blobs.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = [(labels == i).astype(np.uint8) * 255
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    if (rectangular or ai) and blobs:
        seam_h, seam_v = _ai_seam_maps(img) if ai else _seam_maps(img)
        seam_thresh = SEAM_THRESH_AI if ai else SEAM_THRESH
        blobs = [piece
                 for blob in blobs
                 for piece in _split_rectangular(blob, seam_h, seam_v, min_area, seam_thresh)]

    # Opt-in: split any blob that a full-span STRAIGHT boundary shows is really
    # several butted-together photos (e.g. touching postcards with no gap). Off
    # by default because a photo's own straight internal line -- a horizon, a
    # building edge -- is indistinguishable from a real print seam and would be
    # split too (see 69.jpg's dune sea-line). Use only on known composite scans.
    if split_lines:
        blobs = [piece for blob in blobs
                 for piece in _split_blob_by_lines(img, blob, min_area)]

    boxes = [box for box in (_blob_to_box(b) for b in blobs) if box is not None]
    return _order_boxes(boxes, h)


def _order_boxes(boxes, h):
    """Number top-to-bottom, then left-to-right (rows quantised for stability)."""
    row_h = max(1, h // 20)
    return sorted(boxes, key=lambda b: (b[1] // row_h, b[0]))


def _needs_manual(img, boxes):
    """True when the automatic split most likely failed and the page should be
    handed off to manual boxing. Catches the common failure -- touching / faded
    photos that merge into one page-spanning blob -- and the empty result."""
    h, w = img.shape[:2]
    page = h * w
    if not boxes:
        return True
    areas = sorted((bw * bh for _, _, bw, bh in boxes), reverse=True)
    big_w, big_h = max(boxes, key=lambda b: b[2] * b[3])[2:4]
    if big_w >= MANUAL_SPAN * w and big_h >= MANUAL_SPAN * h:
        return True            # one box spans most of the page -> merged/failed
    if len(areas) >= 2 and areas[0] >= MANUAL_DOMINANT_AREA * page \
            and areas[0] >= MANUAL_DOMINANT_RATIO * areas[1]:
        return True            # one blob dwarfs the rest -> it ate its neighbours
    return False


def _write_crops(path, img, boxes):
    """Write each box as <stem>_<n>.<ext>. Returns the number written."""
    stem, ext = os.path.splitext(path)
    if ext.lower() not in IMAGE_EXTS:
        ext = ".png"
    count = 0
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        out = f"{stem}_{i}{ext}"
        if cv2.imwrite(out, img[y:y + bh, x:x + bw]):
            count += 1
            print(f"  wrote {out}  ({bw}x{bh})")
        else:
            print(f"  [skip] could not write: {out}")
    return count


def process_image(path, rectangular=False, ai=False, split_lines=False, manual=None):
    """Extract sub-images from one image file. Returns the number written.

    When `manual` (a context from _load_manual) is given:
      * if the cache already holds hand-drawn boxes for this image, they are used
        directly and the automatic detector is bypassed;
      * otherwise the detector runs, and if its result looks merged/failed the
        image is queued (no crops written yet); main() draws boxes for the whole
        queue after the batch.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [skip] could not read image: {path}")
        return 0

    ap = os.path.abspath(path)
    cached = manual["entries"].get(ap) if manual is not None else None
    if cached:
        boxes = _order_boxes(list(cached), img.shape[0])
        print(f"  using {len(boxes)} manual box(es) from {os.path.basename(manual['path'])}")
        return _write_crops(path, img, boxes)

    boxes = find_subimages(img, rectangular=rectangular, ai=ai, split_lines=split_lines)

    if manual is not None and _needs_manual(img, boxes):
        manual["entries"].setdefault(ap, None)
        if path not in manual["pending"]:
            manual["pending"].append(path)
        print("  [manual] automatic split looks merged/failed -- queued for boxing")
        return 0

    count = _write_crops(path, img, boxes)
    if count == 0:
        print(f"  no sub-images found in {path}")
    return count


def _load_manual(manual_path):
    """Read the manual-boxes cache into a context dict:
       entries -- {abs image path: [(x, y, w, h), ...] or None}, order preserved;
                  a list means "use these boxes", None means "still needs boxes",
       pending -- images queued for drawing this run (as given on the command),
       path    -- the file's path.
    A line is "IMAGE" (needs boxes) or "IMAGE x y w h [x y w h ...]".
    """
    ctx = {"entries": {}, "pending": [], "path": manual_path}
    if manual_path and os.path.exists(manual_path):
        with open(manual_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                ap = os.path.abspath(parts[0])
                nums = parts[1:]
                if len(nums) >= 4 and len(nums) % 4 == 0:
                    ctx["entries"][ap] = [
                        tuple(int(round(float(v))) for v in nums[i:i + 4])
                        for i in range(0, len(nums), 4)]
                else:
                    ctx["entries"].setdefault(ap, None)
    return ctx


def _save_manual(ctx):
    """Rewrite the manual-boxes cache from ctx['entries']. Returns entry count."""
    entries = ctx["entries"]
    if not entries:
        return 0
    with open(ctx["path"], "w", encoding="utf-8") as f:
        f.write("# Manual boxes for pages uncollage.py could not split.\n")
        f.write("#   'PATH'              -> still needs boxes\n")
        f.write("#   'PATH x y w h ...'  -> these boxes are used\n")
        f.write(f"# Draw the pending ones with: python draw_boxes.py "
                f"{os.path.basename(ctx['path'])}\n")
        for ap, boxes in entries.items():
            if boxes:
                flat = " ".join(f"{x} {y} {w} {h}" for x, y, w, h in boxes)
                f.write(f"{ap} {flat}\n")
            else:
                f.write(ap + "\n")
    return len(entries)


def _has_display():
    """True when a GUI window can plausibly be opened (X11 or Wayland present)."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _draw_and_crop(path, manual, draw_mod):
    """Open the box editor for one queued image and crop from what is drawn."""
    print(f"\n  draw boxes for {path}  "
          "(drag one box per photo; s/Enter=save, q/Esc=skip)")
    boxes = draw_mod.draw(path)
    if not boxes:
        print("  [manual] skipped -- still needs boxes")
        return 0
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    boxes = _order_boxes(list(boxes), img.shape[0])
    manual["entries"][os.path.abspath(path)] = boxes
    return _write_crops(path, img, boxes)


def iter_input_paths(path):
    """Yield the image path(s) to process for a given input path."""
    if os.path.splitext(path)[1].lower() in IMAGE_EXTS:
        yield path
        return
    # Treat anything else as a text file listing one image path per line.
    base = os.path.dirname(os.path.abspath(path))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line if os.path.isabs(line) else os.path.join(base, line)


def main(argv):
    args = argv[1:]
    rectangular = ai = False
    for flag in ("--rectangular", "-r"):
        if flag in args:
            rectangular = True
            args = [a for a in args if a != flag]
    for flag in ("--ai", "-a"):
        if flag in args:
            ai = True
            args = [a for a in args if a != flag]
    split_lines = False
    for flag in ("--split-lines", "-l"):
        if flag in args:
            split_lines = True
            args = [a for a in args if a != flag]

    manual_path = MANUAL_FILE_DEFAULT
    if "--manual" in args:
        i = args.index("--manual")
        manual_path = args[i + 1]
        del args[i:i + 2]
    no_draw = "--no-draw" in args
    args = [a for a in args if a != "--no-draw"]

    if len(args) != 1:
        print(__doc__)
        return 2

    in_path = args[0]
    if not os.path.exists(in_path):
        print(f"error: path does not exist: {in_path}")
        return 1

    if ai:
        import seam_ai
        if not seam_ai.SeamMLP.load().ready:
            print("error: --ai needs a trained model. Run:\n"
                  "    python seam_ai.py train")
            return 1

    manual = _load_manual(manual_path)

    # Pass 1: process every image; failures are queued in manual["pending"].
    total = 0
    for img_path in iter_input_paths(in_path):
        print(f"processing {img_path}")
        total += process_image(img_path, rectangular=rectangular, ai=ai,
                               split_lines=split_lines, manual=manual)

    # Pass 2: draw boxes for the queued failures, then crop them.
    queued = list(manual["pending"])
    if queued and not no_draw and _has_display():
        try:
            import draw_boxes
        except Exception as exc:                       # pragma: no cover
            draw_boxes = None
            print(f"  [manual] could not open the box editor ({exc}); "
                  "falling back to the manual file.")
        if draw_boxes is not None:
            print(f"\n{len(queued)} page(s) need manual boxes -- opening the box "
                  "editor for each...")
            for img_path in queued:
                total += _draw_and_crop(img_path, manual, draw_boxes)

    _save_manual(manual)

    # Anything still without boxes (skipped in the editor, --no-draw, or headless).
    pending_left = [ap for ap, b in manual["entries"].items() if not b]
    print(f"\ndone: {total} sub-image(s) written")
    if pending_left and (no_draw or not _has_display()):
        print(f"{len(pending_left)} page(s) still need manual boxes -> {manual_path}\n"
              f"  draw them: python draw_boxes.py {manual_path}\n"
              f"  then re-run this command to crop them.")
    elif pending_left:
        print(f"{len(pending_left)} page(s) were skipped and still need boxes "
              f"(listed in {manual_path}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
