#!/usr/bin/env python3
"""
sam_uncollage.py -- split a scanned collage into its sub-images using Meta's
pretrained Segment Anything Model (SAM).

Background: on scans with uneven illumination, a faded / borderless print on the
shadowed side of the page is locally indistinguishable from the paper -- same
colour, saturation and (noise-swamped) texture -- so no background/colour/edge
threshold, and no small model trained on synthetic data, reliably separates them
(all verified experimentally on 63.jpg). SAM, pretrained on 11M images, segments
by learned object-ness and does far better. It still is not a guaranteed
one-click-free solution: photos that are borderless scenes get over-segmented
into their internal objects, and touching similar prints can merge. Hence two
modes:

  AUTOMATIC (default): SamAutomaticMaskGenerator proposes masks; we keep
    photo-sized, rectangle-like, non-paper, non-page-spanning ones and take the
    best non-overlapping set. Great on bordered/separated photos; may miss or
    split borderless faded ones.

  BOX-PROMPT (--boxes FILE): the reliable mode. FILE has one rough box per photo
    ("x y w h" per line, pixels in the source image -- a loose rectangle drawn
    around each print). SAM refines each box into the exact photo region. Use
    this for pages the automatic mode gets wrong.

CLI:
    python sam_uncollage.py IMG [--checkpoint sam_vit_b.pth] [--debug]
    python sam_uncollage.py IMG --boxes boxes.txt [--debug]

Requires: torch, torchvision, segment-anything, opencv-python (cv2), numpy.
Checkpoint: sam_vit_b_01ec64.pth (ViT-B); default path ./sam_vit_b.pth.
"""

import os
import sys

# Keep SAM within a small/2nd-hand GPU: reduce allocator fragmentation OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "sam_vit_b.pth")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# --- selection tunables (automatic mode) -------------------------------------
MIN_AREA_FRAC = 0.015    # a print is at least this fraction of the page
MAX_AREA_FRAC = 0.60     # ... and at most this
FILL_MIN = 0.55          # mask area / rotated-bounding-rect area (rectangle-ness)
MAX_ASPECT = 4.0         # reject long thin slivers
IOU_MAX = 0.30           # greedy overlap suppression threshold
PAPER_MAX = 0.55         # reject masks more than this fraction paper-like
PAGE_SPAN = 0.92         # reject masks whose bbox spans this much of both dims
# ------------------------------------------------------------------------------


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sam(checkpoint):
    from segment_anything import sam_model_registry
    return sam_model_registry["vit_b"](checkpoint=checkpoint).to(_device())


def _rect_stats(seg):
    cnts, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    if rw * rh == 0:
        return None
    fill = cv2.contourArea(c) / (rw * rh)
    asp = max(rw, rh) / max(1.0, min(rw, rh))
    return fill, asp, cv2.boundingRect(c)


def _paper_frac(img, seg):
    """Fraction of a mask's pixels that look like paper (low-sat AND smooth).

    Lets us drop SAM masks that are really the scanner background, page margins
    or inter-photo gaps rather than a photo.
    """
    h, w = img.shape[:2]
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = max(3, int(round(min(h, w) * 0.003)) | 1)
    m = cv2.boxFilter(gray, -1, (win, win))
    sq = cv2.boxFilter(gray * gray, -1, (win, win))
    std = np.sqrt(np.maximum(sq - m * m, 0.0))
    paper = (sat < 30) & (std < 12)
    s = seg.astype(bool)
    return float(paper[s].mean()) if s.any() else 1.0


def _order(boxes, h):
    row_h = max(1, h // 20)
    boxes.sort(key=lambda b: (b[1] // row_h, b[0]))
    return boxes


def find_subimages_auto(img, checkpoint=DEFAULT_CKPT):
    """Automatic mode: return upright (x, y, w, h) crop boxes."""
    from segment_anything import SamAutomaticMaskGenerator
    h, w = img.shape[:2]
    page = h * w
    gen = SamAutomaticMaskGenerator(
        _sam(checkpoint), points_per_side=32, points_per_batch=16,
        pred_iou_thresh=0.86, stability_score_thresh=0.85, min_mask_region_area=1000)
    masks = gen.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cand = []
    for mobj in masks:
        frac = mobj["area"] / page
        if frac < MIN_AREA_FRAC or frac > MAX_AREA_FRAC:
            continue
        st = _rect_stats(mobj["segmentation"])
        if st is None:
            continue
        fill, asp, box = st
        if fill < FILL_MIN or asp > MAX_ASPECT:
            continue
        if box[2] >= PAGE_SPAN * w and box[3] >= PAGE_SPAN * h:
            continue
        if _paper_frac(img, mobj["segmentation"]) > PAPER_MAX:
            continue
        cand.append((fill * mobj["area"], box))

    cand.sort(key=lambda t: -t[0])
    kept = []
    for _, box in cand:
        x, y, bw, bh = box
        drop = False
        for kx, ky, kbw, kbh in kept:
            ix = max(0, min(x + bw, kx + kbw) - max(x, kx))
            iy = max(0, min(y + bh, ky + kbh) - max(y, ky))
            inter = ix * iy
            union = bw * bh + kbw * kbh - inter
            if (union > 0 and inter / union > IOU_MAX) or inter >= 0.7 * bw * bh:
                drop = True
                break
        if not drop:
            kept.append(box)
    return _order(kept, h)


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def find_subimages_boxes(img, prompt_boxes, checkpoint=DEFAULT_CKPT):
    """Box-prompt mode: refine each rough (x, y, w, h) into the exact photo box.

    SAM tightens a box only when its mask genuinely agrees with the drawn box
    (a photo with a visible edge, e.g. a bordered print). When the print's edge
    is invisible in the scan -- faded photo on same-tone paper -- SAM finds no
    boundary and returns an internal object; we detect that (poor agreement or a
    much smaller mask) and keep the user's drawn box, which is then the ground
    truth. So the user's rough rectangles are never made worse.
    """
    from segment_anything import SamPredictor
    h, w = img.shape[:2]
    pred = SamPredictor(_sam(checkpoint))
    pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out = []
    for prompt in prompt_boxes:
        x, y, bw, bh = prompt
        box = np.array([x, y, x + bw, y + bh], dtype=np.float32)
        masks, scores, _ = pred.predict(box=box, multimask_output=True)
        best = prompt
        best_iou = 0.55                      # require real agreement to override
        parea = bw * bh
        for m in masks:
            st = _rect_stats(m.astype(np.uint8))
            if st is None:
                continue
            rb = st[2]
            if not (0.5 * parea <= rb[2] * rb[3] <= 1.15 * parea):
                continue
            iou = _iou(prompt, rb)
            if iou > best_iou:
                best_iou, best = iou, rb
        out.append(best)
    return _order(out, h)


def _read_boxes(path):
    boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            boxes.append(tuple(int(round(float(v))) for v in parts[:4]))
    return boxes


def process_image(path, checkpoint=DEFAULT_CKPT, boxes_file=None, debug=False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [skip] could not read image: {path}")
        return 0
    if boxes_file:
        boxes = find_subimages_boxes(img, _read_boxes(boxes_file), checkpoint)
    else:
        boxes = find_subimages_auto(img, checkpoint)

    stem, ext = os.path.splitext(path)
    if ext.lower() not in IMAGE_EXTS:
        ext = ".png"
    count = 0
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        out = f"{stem}_{i}{ext}"
        if cv2.imwrite(out, img[y:y + bh, x:x + bw]):
            count += 1
            print(f"  wrote {out}  ({bw}x{bh})")
    if debug:
        vis = img.copy()
        for x, y, bw, bh in boxes:
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 6)
        dbg = f"{stem}.sam_debug.png"
        cv2.imwrite(dbg, vis)
        print(f"  wrote {dbg}")
    if count == 0:
        print(f"  no sub-images found in {path}")
    return count


def main(argv):
    args = argv[1:]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]
    checkpoint = DEFAULT_CKPT
    boxes_file = None
    if "--checkpoint" in args:
        i = args.index("--checkpoint"); checkpoint = args[i + 1]; del args[i:i + 2]
    if "--boxes" in args:
        i = args.index("--boxes"); boxes_file = args[i + 1]; del args[i:i + 2]
    if len(args) != 1:
        print(__doc__)
        return 2
    if not os.path.exists(checkpoint):
        print(f"error: checkpoint not found: {checkpoint}\n"
              "download sam_vit_b_01ec64.pth from Meta's SAM release.")
        return 1
    print(f"processing {args[0]} on {_device()}"
          f"{' (box-prompted)' if boxes_file else ' (automatic)'}")
    process_image(args[0], checkpoint, boxes_file, debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
