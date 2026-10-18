#!/usr/bin/env python3
"""
uncollage.py -- Separate scanned pages / collages into their individual photos.

Usage:
    python uncollage.py INPUT [OUTPUT] [--overwrite] [--cores N] [--touching] [--debug]
                        [--no-draw] [--manual FILE]

  INPUT   an image, a folder of images (non-recursive), or a text file listing
          one image path per line (blank lines and '#' comments ignored;
          relative paths are relative to the text file).
  OUTPUT  folder for the crops (created if needed). Without it, crops are
          written next to each source image.

  Crops are named <stem>_1, <stem>_2, ... (top-to-bottom, then left-to-right).
  They are pixel-identical to the source: a lossless source keeps its format,
  a lossy one (JPEG, WebP) is saved as PNG, since re-encoding it would alter
  the pixels. A folder re-run skips the <stem>_<n> files it wrote itself.

  --touching (-t)  also cut blobs where photos are butted together with no gap
                   (straight shadow seams and full-span straight print edges).
                   Off by default: on ordinary scans it can cut a photo along
                   its own straight horizon or edge.
  --debug (-d)     also write contact sheets uncollage_debug_01.jpg, ... into an
                   uncollage_debug/ subfolder of OUTPUT (or of the input's
                   folder): every page as a thumbnail with its boxes numbered
                   like the crops -- red cropped, magenta queued, green from
                   the manual file, blue drawn this run -- 12 pages a sheet.
  --cores N (-j N) CPU cores to use: pages are processed in parallel, one per
                   core; 1 = one at a time. Default: all cores (fewer only if
                   free memory is short, ~1.5 GB per 1200 dpi page).
  --overwrite      re-crop pages that already have crops in the output folder
                   and replace them. By default such pages are skipped, so a
                   re-run never overwrites and resumes an interrupted batch.
  --no-draw        do not open the box editor for pages that could not be split
                   (see below); just list them in the manual-boxes file.
  --manual FILE    manual-boxes file (default manual_boxes.txt).

Manual-boxes fallback:
  A page whose automatic split looks failed (one page-sized blob, or a blob
  that dwarfs the rest) is queued rather than cropped. After the batch, a box
  editor (draw_boxes.py) opens for each queued page, starting from the boxes
  detection found: adjust them (s/Enter save, q/Esc skip); every edit is kept
  in <manual file>.draft until saved, so an interrupted session loses nothing. The boxes are cached in the manual-boxes
  file, so a later run reuses them. Without a display (or with --no-draw) the
  pages are only listed there; draw them later with
  `python draw_boxes.py manual_boxes.txt` and re-run.

Method:
  Detection runs on a copy downscaled to at most WORK_MAX px; boxes are mapped
  back to full resolution and padded by BOX_PAD so a photo is never clipped --
  neighbouring boxes may overlap.
  * Background colour: of a few candidates (the page's most frequent colours
    and the most frequent colour of its outer ring), the one that covers most of
    the page edge -- paper surrounds the photos; any colour works.
  * Gradients: a smooth quadratic surface is fitted to the paper found and
    refitted, so uneven scanner lighting or tinted paper is followed.
  * A pixel is background when it is smooth and close to that surface, and
    connected to the page border; enclosed pale areas (sky, snow) stay photo.
  * Each large foreground blob is one photo. A blob that is not
    rectangle-shaped (photos touching at a corner or across a shadow) is
    eroded into one core per photo and every pixel given to its nearest core;
    photos parted by a paper gap too thin for the mask are cut along that
    straight paper-coloured line. Either split is kept only if every piece is
    a large, clearly rectangular blob.
  * With --touching, blobs are further cut along straight shadow seams
    (guillotine) and full-span straight print edges.
  Each photo is cropped as the UPRIGHT bounding box of its region (no
  de-skewing), so a tilted photo keeps small background corners.

Limitation:
  Photos that overlap, or butt together along a whole edge with no gap,
  border or shadow between them, come out as one crop (or, with --touching,
  may be cut); faded borderless photos whose edges are invisible in the scan
  are queued for manual boxes.

Requires: opencv-python (cv2), numpy.
"""

import argparse
import multiprocessing
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

# Image extensions recognised when deciding "an image" vs "a list of paths".
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --- Tunables -----------------------------------------------------------------
MIN_AREA_FRAC = 0.01     # blob must be >= this fraction of the page to count
COLOR_TOL = 30.0         # max colour distance from background to be "background"
TOL_K = 4.0              # ... on a page with paper around the photos: this many
TOL_MIN = 20.0           # x the paper's median deviation, but at least TOL_MIN
TEXTURE_TOL = 12.0       # max local std-dev to be "background" (smooth)
FILL_RATIO_MIN = 0.55    # blob area / its rotated-bounding-rect area (rejects
                         # thin frames / speckle; a real photo is a full quad)
KEEP_AREA_FRAC = 0.05    # ... but a blob this large is kept whatever its shape
SLIVER_RATIO = 1 / 8     # a blob thinner than this x its length is not a photo
MERGE_OVERLAP = 0.30     # boxes overlapping more than this x the smaller merge
SPLIT_FILL = 0.95        # a blob filling its rotated rect less than this may be
SPLIT_PIECE_FILL = 0.85  # ... and is cut only into pieces at least this rectangular
SPLIT_KERNELS = (0.01, 0.02, 0.04, 0.08)  # touching photos: tried opening sizes (x page)
PAPER_SEAM_K = 3.0       # thin paper gap filled in by the mask: pixels within
PAPER_SEAM_MIN = 0.90    # this x tol of the paper over >= this of a line ...
PAPER_SEAM_CONTRAST = 0.40  # ... and this much more paper-like than just beside it
EDGE_LINE = 0.01         # lines thinner than this (x page) along its border are dropped
WORK_MAX = 3000          # detection runs on a copy downscaled to this long side
                         # (high-dpi scans: faster, less grain; boxes scaled back)
BOX_PAD = 0.003          # crop boxes grow by this fraction of the page's long
                         # side, so a photo is never clipped (boxes may overlap)
LOSSY_EXTS = {".jpg", ".jpeg", ".webp"}  # crops of these are saved as PNG
BG_EDGE_MIN = 0.50       # paper must cover this much of the image edge
BG_FIT_ITERS = 3         # outlier-rejecting refits of the background surface
# --- Debug contact sheets (--debug) ---------------------------------------------
DEBUG_DIR = "uncollage_debug"   # sheets go in this subfolder of the output
DEBUG_CELL = 480         # each page thumbnail fits a cell this many px square
DEBUG_COLS = 6           # cells per row ...
DEBUG_PER_SHEET = 12     # ... and pages per sheet
DEBUG_COLOURS = {"auto": (0, 0, 230), "manual": (0, 150, 0),   # BGR
                 "drawn": (230, 120, 0), "queued": (230, 0, 230)}
# --- Rectangular-mode (guillotine split) tunables -----------------------------
SEAM_THRESH = 0.70       # min fraction of a cut line that must look like a seam
SEAM_MIN_SPAN = 0.85     # a cut must span >= this fraction of the blob's extent
SEAM_RIDGE_MARGIN = 14   # a shadow ridge must be this much darker than its sides
SEAM_MAX_DEPTH = 6       # max recursive cuts per blob
# --- Manual-boxes fallback ----------------------------------------------------
MANUAL_FILE_DEFAULT = "manual_boxes.txt"  # where failed pages are listed
MANUAL_SPAN = 0.80          # a box spanning >= this of BOTH page dims is checked
                            # for a straight seam (several butted photos)
MANUAL_DOMINANT_AREA = 0.55 # a box >= this fraction of the page that runs to 3
                            # page edges, or that is ...
MANUAL_SOLIDITY = 0.95     # ... unless that box's blob is this solid (a clean
                            # photo pushed into a corner of the scanner)
MANUAL_DOMINANT_RATIO = 3.0 # ... >= this x the next largest => it merged some
# --- Straight-line composition split ------------------------------------------
LINE_COV_THRESH = 0.80   # a full-span content boundary crosses >= this of the box
LINE_JUMP = 30.0         # per-column colour distance counted as an edge crossing
LINE_MARGIN = 0.10       # never cut within this fraction of a box edge
LINE_MIN_PIECE = 0.15    # each resulting piece >= this fraction of the box extent
LINE_MIN_FG = 0.50       # both sides of a cut must be at least this much foreground
LINE_STRAIGHT_MAX = 6.0  # max px scatter of the edge along the line (real print
                         # edges are straight; ragged internal content is not)
LINE_VAR_MIN = 20.0      # a real photo piece has at least this tonal std ...
LINE_EDGE_MIN = 0.015    # ... and this edge density; below => a uniform region
                         # (sky/sea) so the cut sliced one photo -> reject it
LINE_MAX_DEPTH = 5       # max recursive line cuts per blob
# ------------------------------------------------------------------------------


def _page_mode(img):
    """The page's most frequent colour (fullest bin of a coarse 3-D histogram)."""
    px = img.reshape(-1, 3)
    q = (px // 16).astype(np.int32)
    codes = (q[:, 0] * 16 + q[:, 1]) * 16 + q[:, 2]
    peak = np.bincount(codes, minlength=16 ** 3).argmax()
    # float64: a float32 running sum saturates on a large scan (~10^8 pixels at
    # 1200 dpi) and drifts far too dark.
    return px[codes == peak].mean(axis=0, dtype=np.float64).astype(np.float32)


def _color_dist(img, field):
    """Per-pixel colour distance between img and a background colour/field."""
    diff = img.astype(np.float32) - field
    return np.sqrt((diff * diff).sum(axis=-1))


def _border_connected(bg_like):
    """The part of a bg-like map connected to the image border -- the paper
    proper. Pockets *enclosed* by a photo (sky, snow, pale walls) are dropped."""
    n, labels = cv2.connectedComponents(bg_like.astype(np.uint8), connectivity=8)
    border = np.unique(np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    border = border[border > 0]
    return np.isin(labels, border)


def _surface_terms(x, y, w, h):
    """Design matrix of a quadratic surface in normalised page coordinates."""
    x, y = x / w - 0.5, y / h - 0.5
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)


def _fit_surface(xs, ys, colours, w, h):
    """Least-squares quadratic surface (per channel) through the given pixels;
    returns its coefficients."""
    step = max(1, len(xs) // 50000)          # a subsample is plenty for 6 terms
    t = _surface_terms(xs[::step].astype(np.float64), ys[::step].astype(np.float64), w, h)
    coef, *_ = np.linalg.lstsq(t, colours[::step].astype(np.float64), rcond=None)
    return coef.astype(np.float32)


def _eval_surface(coef, w, h):
    """The fitted surface evaluated at every pixel of a w x h page (by
    broadcasting a row of x against a column of y -- no per-term images)."""
    x = (np.arange(w, dtype=np.float32) / w - 0.5)[None, :, None]
    y = (np.arange(h, dtype=np.float32) / h - 0.5)[:, None, None]
    c = coef[:, None, None, :]
    return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y


def background_field(img, smooth):
    """(field, tol): per-pixel background colour of the page (any colour,
    uneven lighting) and the colour tolerance that still counts as paper.

    Paper surrounds the photos, so most of the image's edge is smooth paper. A
    quadratic surface is fitted to the smooth edge pixels with outlier
    rejection (start: their median colour), which follows a scanner lighting
    gradient or tinted paper and ignores photos running off the edge or a dark
    scanner-lid strip; it is then refitted once to the paper found inside. If
    the edge fit's inliers cover less than BG_EDGE_MIN of the edge -- a
    frameless montage, photos to every edge -- the page's most frequent colour
    (typically its seams) is used as a flat background instead, with the
    default tolerance COLOR_TOL.
    """
    h, w = img.shape[:2]
    n = np.arange
    xs = np.concatenate([n(w), n(w), np.zeros(h, int), np.full(h, w - 1)])
    ys = np.concatenate([np.zeros(w, int), np.full(w, h - 1), n(h), n(h)])
    px, ok = img[ys, xs].astype(np.float32), smooth[ys, xs]
    inl = ok.copy()
    if ok.mean() >= BG_EDGE_MIN:
        est = np.median(px[ok], axis=0)
        tol = 2 * COLOR_TOL                   # loose first pass admits a gradient
        for _ in range(BG_FIT_ITERS):
            inl = ok & (_color_dist(px, est) < tol)
            if inl.sum() < 6:
                break
            coef = _fit_surface(xs[inl], ys[inl], px[inl], w, h)
            est = _surface_terms(xs.astype(np.float32), ys.astype(np.float32), w, h) @ coef
            tol = COLOR_TOL
        inl = ok & (_color_dist(px, est) < COLOR_TOL)
    if inl.mean() < BG_EDGE_MIN:
        return np.broadcast_to(_page_mode(img), img.shape).astype(np.float32), COLOR_TOL

    # One refit to all the paper found inside (smooth, close to the edge fit,
    # connected to the border) pins the surface down where the edge is mostly
    # covered by photos. Only one: repeated refits drift, letting pale skies
    # that leak through the gaps pull the surface along.
    field = _eval_surface(coef, w, h)
    pys, pxs = np.nonzero(_border_connected(smooth & (_color_dist(img, field) < COLOR_TOL)))
    field = _eval_surface(_fit_surface(pxs, pys, img[pys, pxs], w, h), w, h)
    # Tolerance from how far the paper actually strays from the surface: clean
    # scanner paper is far tighter than COLOR_TOL, and a tight tolerance keeps
    # a print's off-white border (a few levels off the paper) with the print.
    resid = _color_dist(img[ys[inl], xs[inl]], field[ys[inl], xs[inl]])
    tol = float(np.clip(TOL_K * np.median(resid), TOL_MIN, COLOR_TOL))
    return field, tol


def _local_std(gray, win):
    """Rolling standard deviation over a win x win window."""
    mean = cv2.boxFilter(gray, -1, (win, win), normalize=True)
    sqmean = cv2.boxFilter(gray * gray, -1, (win, win), normalize=True)
    return np.sqrt(np.maximum(sqmean - mean * mean, 0.0))


def foreground_mask(img):
    """Return (mask, field, tol): a binary mask (uint8 0/255) whose blobs are
    the sub-images, the per-pixel background colour it was measured against,
    and the colour tolerance used."""
    h, w = img.shape[:2]

    # Local texture (rolling standard deviation). Kept small so that thin
    # background gaps between photos are not swallowed by neighbouring detail.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = _local_std(gray, max(3, int(round(min(h, w) * 0.003)) | 1)) < TEXTURE_TOL
    field, tol = background_field(img, smooth)

    # Background = close to the (local) background colour AND smooth, and
    # connected to the image border.
    background = _border_connected(smooth & (_color_dist(img, field) < tol))
    fg = (~background).astype(np.uint8) * 255

    # Drop dust/speckle. Kernel stays small to preserve gaps between photos.
    k = max(3, int(round(min(h, w) * 0.004)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Scanner-edge lines: thin dark lines along the page border would join a
    # photo touching the edge and stretch its box to the page corner. Within a
    # band along the border, drop anything thinner than EDGE_LINE of the page.
    b = max(3, int(EDGE_LINE * min(h, w)))
    # (outside the image counts as background, or a line at the very edge
    # would look twice as thick)
    thick = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((b | 1, b | 1), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
    band = np.zeros_like(fg, bool)
    band[:b], band[-b:], band[:, :b], band[:, -b:] = True, True, True, True
    fg[band] = thick[band]
    return fg, field, tol


def _seam_maps(img, field):
    """Return two boolean maps marking pixels that look like a photo boundary.

    seam_h -- suitable for a HORIZONTAL cut: background, or a dark ridge that is
              darker than the pixels a little above and below it (a shadow line
              between a photo above and a photo below).
    seam_v -- the same for a VERTICAL cut (compares left/right neighbours).
    """
    h, w = img.shape[:2]
    bg_like = _color_dist(img, field) < COLOR_TOL

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


def _rect_fill(blob):
    """(fill, contour): how fully a blob's largest contour fills its rotated
    bounding rectangle (1.0 = a perfect, possibly tilted, rectangle)."""
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None
    c = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    return (cv2.contourArea(c) / (rw * rh) if rw * rh else 0.0), c


def _split_touching(blob, min_area):
    """Separate photos that touch -- corner to corner, along a short stretch of
    edge, or bridged by the shadow a thick print casts. Such a blob is not
    rectangle-shaped. Eroding it with a growing square kernel leaves one core
    per photo once the kernel is wider than the contact or bridge. Every blob
    pixel then goes to its nearest core: the cores are the photos shrunk
    evenly, so the dividing line falls on the real contact (or mid-bridge),
    and each photo keeps its full extent. Pieces are split again if need be;
    the split is kept only if every final piece is a large, clearly
    rectangular blob (SPLIT_PIECE_FILL) that fills its rectangle better than
    the merged blob did -- so a single ragged photo, or a pale one whose sky
    leaked into the paper, stays whole."""
    fill, _ = _rect_fill(blob)
    if fill >= SPLIT_FILL:
        return [blob]
    # Kernel sizes scale with the page; the work runs on the blob's own
    # bounding box only, which is much faster than the page.
    page = min(blob.shape)
    x, y, bw, bh = cv2.boundingRect(blob)
    sub = blob[y:y + bh, x:x + bw]
    # Pieces are cut from the blob minus lines thinner than the smallest kernel
    # (a scanner-lid line, a dust trail) so those do not spoil a piece's shape.
    g = max(3, int(SPLIT_KERNELS[0] * page) | 1)
    body = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((g, g), np.uint8))
    for frac in SPLIT_KERNELS:
        k = max(3, int(frac * page) | 1)
        # square kernel: separable, so fast even when large
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv2.erode(sub, np.ones((k, k), np.uint8)))
        cores = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area / 4]
        if len(cores) < 2:
            continue
        core_px = np.isin(labels, cores)
        # nearest-core label for every pixel (labels number the zero pixels
        # of the input, i.e. the core pixels, in scan order)
        _, near = cv2.distanceTransformWithLabels(
            (~core_px).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        owner = np.zeros(near.max() + 1, np.int32)
        owner[near[core_px]] = labels[core_px]
        nearest = owner[near]
        pieces = []
        for i in cores:
            piece = np.zeros_like(blob)
            piece[y:y + bh, x:x + bw] = ((nearest == i) & (body > 0)).astype(np.uint8) * 255
            pieces.append(piece)
        # a piece may hold several photos itself (a wider bridge) -- recurse,
        # and judge the final pieces
        leaves = [q for p in pieces for q in _split_touching(p, min_area)]
        if all(cv2.countNonZero(q) >= min_area and _rect_fill(q)[0] > max(fill, SPLIT_PIECE_FILL)
               for q in leaves):
            return leaves
    return [blob]


def _split_paper_seam(img, blob, near_paper, min_area, depth=0):
    """Split a blob along a thin strip of paper between two photos -- a gap a
    few pixels wide, greyed by the prints' shadows and edges, that the mask
    filled in. Such a strip is a straight line across (almost) the whole blob
    whose pixels are close to the paper colour (within PAPER_SEAM_K x the
    tolerance) while the pixels a little to either side are not. The cut is
    kept only if it leaves two large, clearly rectangular pieces that each
    hold real photo content -- so a pale photo, paper-coloured over wide
    areas, has no such narrow seam and stays whole."""
    if depth >= SEAM_MAX_DEPTH:
        return [blob]
    # no shortcut for already-rectangular blobs: two aligned prints with a
    # thin gap form a near-perfect rectangle
    fill, _ = _rect_fill(blob)
    x, y, bw, bh = cv2.boundingRect(blob)
    sub = blob[y:y + bh, x:x + bw] > 0
    # a seam pixel is near-paper inside the blob, or already background (the
    # part of the gap the mask did keep)
    gap = near_paper[y:y + bh, x:x + bw] | ~sub
    best = None
    for axis, extent in ((0, bw), (1, bh)):                  # 0: vertical seam
        frac = gap.mean(axis)
        m, d = int(LINE_MARGIN * extent), max(3, int(0.02 * extent))
        for j in range(max(m, d), extent - max(m, d)):
            if frac[j] < PAPER_SEAM_MIN:
                continue
            if frac[j] - max(frac[j - d], frac[j + d]) < PAPER_SEAM_CONTRAST:
                continue
            if best is None or frac[j] > best[0]:
                best = (frac[j], axis, j, d)
    if best is None:
        return [blob]
    _, axis, j, d = best
    trial = blob.copy()
    t = max(1, d // 2)
    if axis == 0:
        trial[y:y + bh, x + j - t:x + j + t + 1] = 0
    else:
        trial[y + j - t:y + j + t + 1, x:x + bw] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(trial)
    pieces = [(labels == i).astype(np.uint8) * 255
              for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if len(pieces) < 2 or not all(_rect_fill(p)[0] >= min(fill, SPLIT_FILL)
                                  and _rect_fill(p)[0] > SPLIT_PIECE_FILL
                                  and not _blob_uniform(img, p) for p in pieces):
        return [blob]
    return [q for p in pieces
            for q in _split_paper_seam(img, p, near_paper, min_area, depth + 1)]


def _blob_to_box(blob, keep_area):
    """Return the upright crop box for a blob mask, or None if it isn't a photo."""
    fill, c = _rect_fill(blob)
    # Reject shapes that do not fill their (rotated) bounding rectangle -- thin
    # frames, speckle rings, stray marks. A photo fills its minAreaRect closely.
    # A blob of at least keep_area pixels is kept whatever its shape: it is
    # photos that could not be separated, and cropping them together beats
    # losing them.
    if c is None or (fill < FILL_RATIO_MIN and cv2.countNonZero(blob) < keep_area):
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    if min(rw, rh) < SLIVER_RATIO * max(rw, rh):
        return None              # a long thin strip (scanner-edge shading), no photo
    # Upright bounding box keeps the sub-image in its original orientation
    # (no de-skewing), as required.
    return cv2.boundingRect(c)


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


def _piece_uniform(gray_slice, edge_slice, mask_slice):
    """True if the foreground of a piece is too plain to be a photo on its own
    -- little tonal spread AND/OR almost no edges (sky, sea, a flat wall). Such a
    piece means the cut sliced through a single photo rather than between two."""
    m = mask_slice
    if m.sum() < 1:
        return True
    return gray_slice[m].std() < LINE_VAR_MIN or edge_slice[m].mean() < LINE_EDGE_MIN


def _best_line_cut(region, mask_region, min_area):
    """Best full-span straight boundary in a region, or None. Returns
    (orientation 'h' | 'v', position). A cut qualifies only when it (a) is a
    full-span content discontinuity (LINE_COV_THRESH), (b) is a straight line
    (LINE_STRAIGHT_MAX) -- which tells a real print edge from ragged internal
    content of equal contrast -- (c) leaves two large, mostly-foreground pieces
    (so it separates two photos, not a photo from paper), and (d) leaves two
    pieces that each look like a real photo, not a uniform sky/sea slice. When
    the top-coverage cut fails (d) the next-best qualifying cut is used, so a
    photo's own straight horizon is skipped in favour of the real seam."""
    best, best_cov = None, LINE_COV_THRESH
    h, w = region.shape[:2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120) > 0
    mb = mask_region > 0
    for axis, orient, extent, other in ((0, "h", h, w), (1, "v", w, h)):
        cov = _disc_coverage(region, axis)
        lo, hi = int(LINE_MARGIN * extent), extent - int(LINE_MARGIN * extent)
        for p in range(lo, hi):
            if cov[p] <= best_cov or min(p, extent - p) < LINE_MIN_PIECE * extent:
                continue
            if min(p, extent - p) * other < min_area:
                continue
            if orient == "h":
                f1, f2 = mb[:p].mean(), mb[p:].mean()
            else:
                f1, f2 = mb[:, :p].mean(), mb[:, p:].mean()
            if min(f1, f2) < LINE_MIN_FG:
                continue
            if _edge_straightness(gray, p, axis) > LINE_STRAIGHT_MAX:
                continue
            if orient == "h":
                u1 = _piece_uniform(gray[:p], edges[:p], mb[:p])
                u2 = _piece_uniform(gray[p:], edges[p:], mb[p:])
            else:
                u1 = _piece_uniform(gray[:, :p], edges[:, :p], mb[:, :p])
                u2 = _piece_uniform(gray[:, p:], edges[:, p:], mb[:, p:])
            if u1 or u2:
                continue
            best_cov, best = cov[p], (orient, p)
    return best


def _blob_uniform(img, blob):
    """True if a blob's foreground is too plain to be a photo -- little tonal
    spread or almost no edges. Used to discard sky/sea slivers that a cut can
    leave as their own connected component (e.g. 69.jpg's dune sea-strip)."""
    ys, xs = np.where(blob > 0)
    if len(xs) == 0:
        return True
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    g = cv2.cvtColor(img[y0:y1 + 1, x0:x1 + 1], cv2.COLOR_BGR2GRAY)
    m = blob[y0:y1 + 1, x0:x1 + 1] > 0
    if g[m].std() < LINE_VAR_MIN:
        return True
    return (cv2.Canny(g, 40, 120) > 0)[m].mean() < LINE_EDGE_MIN


def _split_blob_by_lines(img, blob, min_area, depth=0):
    """Recursively split a foreground blob wherever a full-span straight content
    boundary shows it actually holds several butted-together photos (e.g. the
    touching postcards in 69.jpg). The blob mask is cut along the line and
    re-labelled with connected components, so each photo comes out as its own
    tightly-bounded blob (no paper corners, no reliance on a gap/seam). Pieces
    left too uniform by a cut (a detached sky/sea slice) are discarded rather
    than saved as spurious sub-images."""
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
    pieces = [(labels == i).astype(np.uint8) * 255
              for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    pieces = [pb for pb in pieces if not _blob_uniform(img, pb)]
    if len(pieces) < 2:
        return [blob]                       # cut did not separate two real photos
    return [out for pb in pieces
            for out in _split_blob_by_lines(img, pb, min_area, depth + 1)]


def find_subimages(img, rectangular=False, split_lines=False, info=None):
    """Return a list of (x, y, w, h) upright crop boxes, one per sub-image.

    img may be any image cv2 reads (grey, alpha, 16-bit). Detection runs on an
    8-bit BGR copy whose long side is at most WORK_MAX; the boxes are mapped
    back to full resolution and padded by BOX_PAD (clipped to the page), so a
    photo is never clipped even if neighbouring boxes then overlap.

    rectangular: also guillotine-split blobs along straight background/shadow
    seams (_split_rectangular). split_lines: also split blobs along full-span
    straight print edges (_split_blob_by_lines). The CLI's --touching sets both.
    With an `info` dict, info["solidity"] receives each box's blob solidity
    (area / convex-hull area), in box order -- used by _needs_manual.
    """
    H, W = img.shape[:2]
    img = _to_bgr8(img)
    scale = min(1.0, WORK_MAX / max(H, W))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    mask, field, tol = foreground_mask(img)
    min_area = MIN_AREA_FRAC * h * w

    # Split the mask into connected foreground blobs, separating photos that
    # touch at a corner or across a shadow, then photos separated only by a
    # gap too thin to survive in the mask.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = [piece for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area
             for piece in _split_touching((labels == i).astype(np.uint8) * 255, min_area)]
    near_paper = _color_dist(img, field) < PAPER_SEAM_K * tol
    blobs = [piece for blob in blobs
             for piece in _split_paper_seam(img, blob, near_paper, min_area)]

    if rectangular and blobs:
        seam_h, seam_v = _seam_maps(img, field)
        blobs = [piece
                 for blob in blobs
                 for piece in _split_rectangular(blob, seam_h, seam_v, min_area, SEAM_THRESH)]

    # Opt-in: split any blob that a full-span STRAIGHT boundary shows is really
    # several butted-together photos (e.g. touching postcards with no gap). Off
    # by default because a photo's own straight internal line -- a horizon, a
    # building edge -- is indistinguishable from a real print seam and would be
    # split too (see 69.jpg's dune sea-line). Use only on known composite scans.
    if split_lines:
        blobs = [piece for blob in blobs
                 for piece in _split_blob_by_lines(img, blob, min_area)]

    keep = KEEP_AREA_FRAC * h * w
    found = [(box, b) for box, b in ((_blob_to_box(b, keep), b) for b in blobs)
             if box is not None]
    pad = BOX_PAD * max(H, W)
    full = _merge_overlaps([(_full_res_box(box, scale, pad, W, H), _solidity(b))
                            for box, b in found])
    boxes = _order_boxes([bx for bx, _ in full])
    if info is not None:
        sol_of = dict(full)
        info["solidity"] = [sol_of[bx] for bx in boxes]
    return boxes


def _merge_overlaps(items):
    """Merge boxes that overlap by more than MERGE_OVERLAP of the smaller one
    into their union. Separate photos overlap only by the crop margin or a
    tilted corner; a big overlap means two pieces of one photo (a pale
    portrait that fell apart, a mount border around its picture). items are
    (box, solidity); a merged box keeps the higher solidity."""
    items = list(items)
    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, sa), (b, sb) = items[i], items[j]
                ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if ix > 0 and iy > 0 and \
                        ix * iy > MERGE_OVERLAP * min(a[2] * a[3], b[2] * b[3]):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2])
                    y1 = max(a[1] + a[3], b[1] + b[3])
                    items[i] = ((x0, y0, x1 - x0, y1 - y0), max(sa, sb))
                    del items[j]
                    merged = True
                    break
            if merged:
                break
    return items


def _solidity(blob):
    """Blob area / its convex hull's area: ~1 for a photo (even a torn or
    tilted one), clearly less for several photos merged into one blob."""
    _, c = _rect_fill(blob)
    hull = cv2.contourArea(cv2.convexHull(c)) if c is not None else 0
    return cv2.contourArea(c) / hull if hull else 0.0


def _to_bgr8(img):
    """An 8-bit, 3-channel BGR view of any image cv2 reads, for detection."""
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1, np.iinfo(img.dtype).max)
                                  if img.dtype.kind in "ui" else 255.0)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _full_res_box(box, scale, pad, W, H):
    """Map a box found on the detection copy back to full resolution, grown
    by pad on every side and clipped to the W x H page."""
    x, y, bw, bh = box
    x0 = max(0, int(np.floor(x / scale - pad)))
    y0 = max(0, int(np.floor(y / scale - pad)))
    x1 = min(W, int(np.ceil((x + bw) / scale + pad)))
    y1 = min(H, int(np.ceil((y + bh) / scale + pad)))
    return (x0, y0, x1 - x0, y1 - y0)


def _order_boxes(boxes):
    """Number top-to-bottom, then left-to-right. Rows are formed by vertical
    overlap -- a box joins the current row when its top lies above the middle
    of that row's first box -- so photos side by side keep left-to-right
    order even if their tops differ a little."""
    rows = []
    for b in sorted(boxes, key=lambda b: b[1]):
        if rows and b[1] < rows[-1][0][1] + rows[-1][0][3] / 2:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b[0])]


def split_page(img, rectangular=False, split_lines=False):
    """(boxes, queued): the crop boxes and whether the page should go to the
    manual-boxes fallback instead -- exactly what uncollage.py decides."""
    info = {}
    boxes = find_subimages(img, rectangular=rectangular, split_lines=split_lines, info=info)
    return boxes, _needs_manual(img, boxes, info["solidity"])


def _needs_manual(img, boxes, solidity=None):
    """True when the automatic split most likely failed and the page should be
    handed off to manual boxing. Catches the common failure -- touching / faded
    photos that merge into one blob spanning the page edge to edge -- and the
    empty result. One large photo with paper visible around it is fine --
    unless a full-span straight print edge runs across it (_looks_butted) --
    and so is a clean photo pushed into a corner of the scanner (given
    `solidity`, the boxes' blob solidity from find_subimages)."""
    h, w = img.shape[:2]
    page = h * w
    if not boxes:
        return True
    areas = sorted((bw * bh for _, _, bw, bh in boxes), reverse=True)
    x, y, big_w, big_h = max(boxes, key=lambda b: b[2] * b[3])
    near = 2 * BOX_PAD * max(h, w) + 2         # padding may push a box to the edge
    edges = (x <= near) + (y <= near) + (x + big_w >= w - near) + (y + big_h >= h - near)
    big_sol = solidity[boxes.index((x, y, big_w, big_h))] if solidity else 0.0
    if edges == 4 and big_w >= MANUAL_SPAN * w and big_h >= MANUAL_SPAN * h:
        return True            # the whole page came out as foreground -> failed
    if edges >= 3 and areas[0] >= MANUAL_DOMINANT_AREA * page \
            and big_sol < MANUAL_SOLIDITY:
        return True            # a big irregular box runs to the page edges -> merged
    if big_w >= MANUAL_SPAN * w and big_h >= MANUAL_SPAN * h \
            and _looks_butted(img, (x, y, big_w, big_h)):
        return True            # one photo-sized box, but really several butted
    if len(areas) >= 2 and areas[0] >= MANUAL_DOMINANT_AREA * page \
            and areas[0] >= MANUAL_DOMINANT_RATIO * areas[1]:
        return True            # one blob dwarfs the rest -> it ate its neighbours
    return False


def _looks_butted(img, box):
    """True if a full-span straight print edge crosses the box -- i.e. it is
    probably several photos butted together rather than one. Only used to
    queue a page for manual boxes, so a false alarm merely asks the user."""
    x, y, bw, bh = box
    crop = _to_bgr8(img[y:y + bh, x:x + bw])
    s = min(1.0, WORK_MAX / max(bw, bh))
    if s < 1.0:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    mask = np.full((ch, cw), 255, np.uint8)
    return _best_line_cut(crop, mask, MIN_AREA_FRAC * ch * cw) is not None


def read_image(path):
    """Read an image exactly as stored -- bit depth, alpha and grey kept, EXIF
    orientation NOT applied -- so crops are pixel-identical to the file."""
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def _crop_base(path, out_dir=None):
    """(stem, ext) of a source's crop files: <stem>_<n><ext>. A lossless
    format is kept, a lossy one (LOSSY_EXTS) becomes PNG; with out_dir the
    crops go there (keeping the source's base name), otherwise next to it."""
    stem, ext = os.path.splitext(path)
    if ext.lower() not in IMAGE_EXTS or ext.lower() in LOSSY_EXTS:
        ext = ".png"
    if out_dir:
        stem = os.path.join(out_dir, os.path.basename(stem))
    return stem, ext


def _existing_crops(path, out_dir=None):
    """{n: file} of the crops of `path` already in its output folder."""
    stem, ext = _crop_base(path, out_dir)
    folder, base = os.path.split(stem)
    pat = re.compile(re.escape(base) + r"_(\d+)" + re.escape(ext) + "$")
    try:
        names = os.listdir(folder or ".")
    except OSError:
        return {}
    return {int(m.group(1)): os.path.join(folder, n)
            for n in names for m in [pat.match(n)] if m}


def _write_crops(path, img, boxes, out_dir=None, overwrite=False):
    """Write each box as <stem>_<n><ext> (see _crop_base), pixel-identical to
    the source. A crop of a colour image whose channels are all equal (a grey
    photo, or a grey scan stored as RGB) is written as one grey channel -- the
    same pixel values in a third of the space; each crop is judged on its own.
    Boxes are clipped to the image. An existing file is kept unless
    `overwrite`. A page's crops are written under temporary ".part" names
    and renamed once all are written, so an interrupted or failed write never
    leaves a page looking finished (re-runs skip pages that have crops).
    Returns the number of crops in place (written or kept)."""
    stem, ext = _crop_base(path, out_dir)
    params = [cv2.IMWRITE_PNG_COMPRESSION, 1] if ext.lower() == ".png" else []
    H, W = img.shape[:2]
    count, done, failed = 0, [], False
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + bw), min(H, y + bh)
        out = f"{stem}_{i}{ext}"
        part = f"{stem}_{i}.part{ext}"          # cv2 picks the format by extension
        if not overwrite and os.path.exists(out):
            print(f"  [keep] already exists: {out}")
            count += 1
            continue
        crop = img[y0:y1, x0:x1]
        if crop.ndim == 3 and crop.shape[2] == 3 and \
                (crop[..., 0] == crop[..., 1]).all() and (crop[..., 1] == crop[..., 2]).all():
            crop = crop[..., 0]
        try:
            ok = x1 > x0 and y1 > y0 and cv2.imwrite(part, crop, params)
        except cv2.error:                       # some write failures raise instead
            ok = False
        if ok:
            done.append((part, out))
            print(f"  wrote {out}  ({x1 - x0}x{y1 - y0})")
        else:
            print(f"  [skip] could not write: {out}")
            failed = True
            if os.path.exists(part):
                os.remove(part)
    if failed:            # publish nothing, so a re-run redoes the whole page
        for part, _ in done:
            os.remove(part)
        return count
    for part, out in done:                       # the page is complete: publish
        os.replace(part, out)
        count += 1
    return count


def _process_one(path, rectangular, split_lines, cached, check, out_dir, want_tile,
                 overwrite=False):
    """Do one page's work without touching shared state, so it can run in a
    worker process. Returns {"count", "queued", "boxes", "tile"}: crops
    written (and "failed": crops that could not be written), whether the page
    needs manual boxes, the boxes used or detected,
    and (want_tile) its debug contact-sheet cell."""
    res = {"count": 0, "failed": 0, "queued": False, "boxes": [], "tile": None}
    img = read_image(path)
    if img is None:
        print(f"  [skip] could not read image: {path}")
        return res
    if cached:
        boxes = _order_boxes(list(cached))
        print(f"  using {len(boxes)} manual box(es)")
        status = "manual"
    else:
        boxes, queued = split_page(img, rectangular, split_lines)
        status = "queued" if check and queued else "auto"
    res["boxes"] = boxes
    if want_tile:
        res["tile"] = _debug_tile(path, img, boxes, status)
    if status == "queued":
        print("  [manual] automatic split looks merged/failed -- queued for boxing")
        res["queued"] = True
        return res
    res["count"] = _write_crops(path, img, boxes, out_dir, overwrite)
    res["failed"] = len(boxes) - res["count"]
    _note_leftovers(path, out_dir, len(boxes))
    if res["count"] == 0:
        print(f"  no sub-images found in {path}")
    return res


def process_image(path, rectangular=False, split_lines=False,
                  manual=None, out_dir=None, debug=None, overwrite=False):
    """Extract sub-images from one image file. Returns the number written.

    When `manual` (a context from _load_manual) is given:
      * if the cache already holds hand-drawn boxes for this image, they are used
        directly and the automatic detector is bypassed;
      * otherwise the detector runs, and if its result looks merged/failed the
        image is queued (no crops written yet) with the boxes it did find, as
        the editor's starting point; main() draws boxes for the whole queue
        after the batch.
    Crops are written to out_dir if given, else next to the source. With a
    `debug` dict, a thumbnail of the page and its boxes is added to it for the
    contact sheets (see _debug_tile). Existing crop files are kept unless
    `overwrite`.
    """
    cached = manual["entries"].get(os.path.abspath(path)) if manual is not None else None
    res = _process_one(path, rectangular, split_lines, cached, manual is not None,
                       out_dir, debug is not None, overwrite)
    _record(path, res, manual, debug)
    return res["count"]


def _note_leftovers(path, out_dir, n):
    """After re-cropping a page into n crops, point out older crops of it with
    higher numbers (from a run that found more photos); they are left alone,
    since a file named like a crop could be an original."""
    old = sorted(f for k, f in _existing_crops(path, out_dir).items() if k > n)
    if old:
        print(f"  [note] {len(old)} older crop(s) of this page remain: "
              + ", ".join(os.path.basename(f) for f in old))


def _record(path, res, manual, debug):
    """Merge one page's result into the shared manual / debug state."""
    ap = os.path.abspath(path)
    if res["queued"] and manual is not None:
        manual["entries"].setdefault(ap, None)
        manual.setdefault("detected", {})[ap] = res["boxes"]
        if path not in manual["pending"]:
            manual["pending"].append(path)
    if debug is not None and res["tile"] is not None:
        debug[ap] = res["tile"]


def _worker_init():
    """Each worker process takes one core; OpenCV's own threads would only
    compete with the other workers."""
    cv2.setNumThreads(1)


def _process_job(job):
    path, kw = job
    print(f"processing {path}", flush=True)
    return _process_one(path, **kw)


def _default_jobs():
    """All cores, but no more workers than free memory allows (a 1200 dpi
    page needs about 1.5 GB at peak)."""
    jobs = os.cpu_count() or 1
    try:
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable"))
        jobs = min(jobs, max(1, int(avail / (1.5 * 1024 ** 2))))
    except (OSError, StopIteration, ValueError):
        pass
    return jobs


def _load_manual(manual_path):
    """Read the manual-boxes cache into a context dict:
       entries -- {abs image path: [(x, y, w, h), ...] or None}, order preserved;
                  a list means "use these boxes", None means "still needs boxes",
       pending -- images queued for drawing this run (as given on the command),
       detected -- {abs path: boxes found} for queued images (editor start),
       path    -- the file's path.
    A line is "IMAGE" (needs boxes) or "IMAGE x y w h [x y w h ...]".
    """
    ctx = {"entries": {}, "pending": [], "detected": {}, "path": manual_path}
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


def _draw_and_crop(path, manual, draw_mod, out_dir=None, index=None, total=None,
                   debug=None, overwrite=False):
    """Open the box editor for one queued image and crop from what is drawn.
    Returns (crops written, crops that could not be written)."""
    print(f"\n  draw boxes for {path}  "
          "(adjust the boxes; s/Enter=save, q/Esc=skip)")
    boxes = draw_mod.draw(path, index, total,
                          initial=manual.get("detected", {}).get(os.path.abspath(path)),
                          draft_file=manual["path"] + ".draft")
    if not boxes:
        print("  [manual] skipped -- still needs boxes")
        return 0, 0
    img = read_image(path)
    boxes = _order_boxes(list(boxes))
    manual["entries"][os.path.abspath(path)] = boxes
    if debug is not None:
        debug[os.path.abspath(path)] = _debug_tile(path, img, boxes, "drawn")
    n = _write_crops(path, img, boxes, out_dir, overwrite)
    _note_leftovers(path, out_dir, len(boxes))
    return n, len(boxes) - n


def _debug_tile(path, img, boxes, status):
    """A contact-sheet cell for one page: its thumbnail (fitted into a
    DEBUG_CELL square) with the boxes drawn and numbered like the crop files,
    coloured by status (DEBUG_COLOURS), and a caption. Kept JPEG-encoded, so a
    run over hundreds of pages stays small in memory."""
    c = DEBUG_CELL
    cap = 26                                       # caption strip height
    page = _to_bgr8(img)
    h, w = page.shape[:2]
    s = min(c / w, (c - cap) / h)
    thumb = cv2.resize(page, (max(1, int(w * s)), max(1, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    cell = np.full((c, c, 3), 90, np.uint8)
    ox, oy = (c - thumb.shape[1]) // 2, (c - cap - thumb.shape[0]) // 2
    cell[oy:oy + thumb.shape[0], ox:ox + thumb.shape[1]] = thumb
    col = DEBUG_COLOURS[status]
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        p0 = (ox + int(x * s), oy + int(y * s))
        p1 = (ox + int((x + bw) * s), oy + int((y + bh) * s))
        cv2.rectangle(cell, p0, p1, col, 2)
        cv2.putText(cell, str(i), (p0[0] + 4, p0[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(cell, str(i), (p0[0] + 4, p0[1] + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 2, cv2.LINE_AA)
    if status == "queued":
        cv2.putText(cell, "QUEUED", (10, c - cap - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    col, 3, cv2.LINE_AA)
    label = {"auto": "", "manual": " manual", "drawn": " drawn", "queued": " queued"}[status]
    name = os.path.basename(path)
    if len(name) > 40:
        name = name[:18] + "..." + name[-19:]
    cell[c - cap:] = 40
    cv2.putText(cell, f"{name}  {len(boxes)}{label}", (6, c - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, col if status != "auto" else (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode(".jpg", cell, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]


def write_debug_sheets(tiles, folder):
    """Write the debug cells as contact sheets uncollage_debug_01.jpg, ... --
    DEBUG_PER_SHEET pages each, DEBUG_COLS per row -- into folder. Returns the
    sheet paths."""
    tiles = list(tiles)
    if not tiles:
        return []
    os.makedirs(folder, exist_ok=True)
    c, paths = DEBUG_CELL, []
    for k in range(0, len(tiles), DEBUG_PER_SHEET):
        cells = [cv2.imdecode(t, cv2.IMREAD_COLOR) for t in tiles[k:k + DEBUG_PER_SHEET]]
        cols = min(DEBUG_COLS, len(cells))
        rows = -(-len(cells) // cols)
        sheet = np.full((rows * c + (rows - 1) * 4, cols * c + (cols - 1) * 4, 3), 255, np.uint8)
        for i, cell in enumerate(cells):
            r, q = divmod(i, cols)
            sheet[r * (c + 4):r * (c + 4) + c, q * (c + 4):q * (c + 4) + c] = cell
        out = os.path.join(folder, f"uncollage_debug_{k // DEBUG_PER_SHEET + 1:02d}.jpg")
        cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(out)
    return paths


def _looks_generated(stem, stems):
    """True if `stem` looks like a sub-image this tool wrote -- "<parent>_<n>"
    whose parent image is also present -- so a folder scan skips its own outputs.
    """
    m = re.match(r"^(.*)_\d+$", stem)
    return bool(m) and m.group(1) in stems


def iter_input_paths(path):
    """Yield the image path(s) to process for a given input path.

    An image file yields itself; a folder yields every image in it (sorted,
    non-recursive, minus its own <stem>_<n> outputs); anything else is treated
    as a text file listing one image path per line.
    """
    if os.path.isdir(path):
        names = sorted(os.listdir(path))
        stems = {os.path.splitext(n)[0] for n in names
                 if os.path.splitext(n)[1].lower() in IMAGE_EXTS}
        for name in names:
            stem, ext = os.path.splitext(name)
            if ext.lower() in IMAGE_EXTS and not _looks_generated(stem, stems):
                yield os.path.join(path, name)
        return
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
    ap = argparse.ArgumentParser(prog="uncollage.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 usage="%(prog)s INPUT [OUTPUT] [options]")
    ap.add_argument("input", metavar="INPUT", help="image, folder, or text file of image paths")
    ap.add_argument("output", metavar="OUTPUT", nargs="?",
                    help="folder for the crops (default: next to each source)")
    ap.add_argument("-t", "--touching", action="store_true",
                    help="also cut photos butted together with no gap")
    ap.add_argument("--no-draw", action="store_true",
                    help="do not open the box editor for failed pages")
    ap.add_argument("--manual", default=MANUAL_FILE_DEFAULT, metavar="FILE",
                    help="manual-boxes file (default %(default)s)")
    ap.add_argument("-j", "--cores", dest="jobs", type=int, default=0,
                    metavar="N", help="CPU cores to use, i.e. pages processed in "
                    "parallel; 1 = one at a time (default: all cores)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-crop pages that already have crops in the output "
                         "folder, replacing them (default: skip such pages)")
    ap.add_argument("-d", "--debug", action="store_true",
                    help=f"also write contact sheets of every page and its boxes "
                         f"into {DEBUG_DIR}/ in the output folder")
    opt = ap.parse_args(argv[1:])
    in_path, out_dir, manual_path, no_draw = opt.input, opt.output, opt.manual, opt.no_draw

    if not os.path.exists(in_path):
        print(f"error: path does not exist: {in_path}")
        return 1
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    manual = _load_manual(manual_path)
    debug = {} if opt.debug else None

    # Pass 1: process every image, several at once; failures are queued in
    # manual["pending"] (in input order).
    paths = list(iter_input_paths(in_path))
    if not opt.overwrite:                       # never redo / overwrite finished pages
        done = [p for p in paths if _existing_crops(p, out_dir)]
        if done:
            print(f"{len(done)} page(s) already have crops -- skipped "
                  f"(add --overwrite to re-crop them)")
            paths = [p for p in paths if p not in set(done)]
    kw = dict(rectangular=opt.touching, split_lines=opt.touching, check=True,
              out_dir=out_dir, want_tile=debug is not None, overwrite=opt.overwrite)
    jobs = [(p, dict(kw, cached=manual["entries"].get(os.path.abspath(p)))) for p in paths]
    n_jobs = min(len(jobs), opt.jobs if opt.jobs > 0 else _default_jobs())
    if n_jobs > 1:
        # "spawn": forking a process whose OpenCV thread pool is running can
        # deadlock the children
        with ProcessPoolExecutor(n_jobs, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init) as ex:
            results = list(ex.map(_process_job, jobs))
    else:
        results = [_process_job(job) for job in jobs]
    total = failed = 0
    for p, res in zip(paths, results):
        _record(p, res, manual, debug)
        total += res["count"]
        failed += res["failed"]

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
            nq = len(queued)
            for k, img_path in enumerate(queued, 1):
                n, nf = _draw_and_crop(img_path, manual, draw_boxes, out_dir, k, nq,
                                       debug=debug, overwrite=opt.overwrite)
                total, failed = total + n, failed + nf
                _save_manual(manual)    # persist after each crop, in case of a crash

    _save_manual(manual)

    if debug:
        if out_dir:
            base = out_dir
        elif os.path.isdir(in_path):
            base = in_path
        else:
            base = os.path.dirname(os.path.abspath(in_path))
        sheets = write_debug_sheets(debug.values(), os.path.join(base, DEBUG_DIR))
        print(f"\ndebug: {len(sheets)} contact sheet(s) in {os.path.join(base, DEBUG_DIR)}")

    # Anything still without boxes (skipped in the editor, --no-draw, or headless).
    pending_left = [ap for ap, b in manual["entries"].items() if not b]
    print(f"\ndone: {total} sub-image(s) written")
    if failed:
        print(f"ERROR: {failed} crop(s) could not be written (disk full or no write "
              f"permission?) -- see the [skip] lines above; re-run once fixed.")
    if pending_left and (no_draw or not _has_display()):
        print(f"{len(pending_left)} page(s) still need manual boxes -> {manual_path}\n"
              f"  draw them: python draw_boxes.py {manual_path}\n"
              f"  then re-run this command to crop them.")
    elif pending_left:
        print(f"{len(pending_left)} page(s) were skipped and still need boxes "
              f"(listed in {manual_path}).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
