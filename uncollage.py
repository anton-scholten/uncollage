#!/usr/bin/env python3
"""
uncollage.py -- Separate an image collage / scanned page into individual sub-images.

Usage:
    python uncollage.py PATH

  * If PATH is an image file, its sub-images are extracted.
  * If PATH is a text file, every line is treated as an image path and all of
    them are processed in batch. Blank lines and lines starting with '#' are
    ignored. Relative paths are resolved against the text file's own folder.

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

Limitation:
  Photos that physically overlap or butt together with no strip of background
  (or only a dark shadow) between them cannot be told apart by a background
  based method and may be emitted as a single sub-image.

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


def find_subimages(img):
    """Return a list of (x, y, w, h) upright crop boxes, one per sub-image."""
    h, w = img.shape[:2]
    mask = foreground_mask(img)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_AREA_FRAC * h * w

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        # Reject shapes that do not fill their (rotated) bounding rectangle --
        # thin frames, speckle rings, stray marks. A photo, at any rotation,
        # fills its minAreaRect almost completely.
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if rw * rh == 0 or area / (rw * rh) < FILL_RATIO_MIN:
            continue
        # Upright bounding box keeps the sub-image in its original orientation
        # (no de-skewing), as required.
        boxes.append(cv2.boundingRect(c))

    # Number top-to-bottom, then left-to-right (rows quantised for stability).
    row_h = max(1, h // 20)
    boxes.sort(key=lambda b: (b[1] // row_h, b[0]))
    return boxes


def process_image(path):
    """Extract sub-images from one image file. Returns the number written."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [skip] could not read image: {path}")
        return 0

    boxes = find_subimages(img)
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

    if count == 0:
        print(f"  no sub-images found in {path}")
    return count


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
    if len(argv) != 2:
        print(__doc__)
        return 2

    in_path = argv[1]
    if not os.path.exists(in_path):
        print(f"error: path does not exist: {in_path}")
        return 1

    total = 0
    for img_path in iter_input_paths(in_path):
        print(f"processing {img_path}")
        total += process_image(img_path)
    print(f"done: {total} sub-image(s) written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
