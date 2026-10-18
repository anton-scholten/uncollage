#!/usr/bin/env python3
"""
Headless regression suite for uncollage.py.

Builds the synthetic collages from gen_collages.py in memory and asserts that
each scenario yields the expected number of sub-images with its intended flags,
that the flags are actually what drive the split (negative controls), and that
an un-splittable page is queued for the manual fallback instead of mis-cropped.
No display or GUI is needed.

Run directly (prints a PASS/FAIL table, non-zero exit on failure):

    python tests/test_uncollage.py

Or under pytest:

    pytest tests/test_uncollage.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import uncollage
import gen_collages
from gen_collages import SCENARIOS


def _count(img, flags):
    return len(uncollage.find_subimages(img, **flags))


# --- pytest-style tests -------------------------------------------------------
def test_scenarios_match_expected():
    """Each scenario yields its expected count (or queues) with its flags."""
    for name, build, flags, expected, needs_manual in SCENARIOS:
        img = build()
        boxes, queued = uncollage.split_page(img, **flags)
        if needs_manual:
            assert queued, \
                f"{name}: expected the page to be queued for manual boxing"
        else:
            assert len(boxes) == expected, \
                f"{name}: expected {expected} sub-images, got {len(boxes)}"


def test_flags_drive_the_split():
    """Without its flag, a composite/grid/touching page does NOT cleanly split."""
    # touching: plain mode merges the two into one blob
    assert _count(gen_collages.touching(), {}) == 1
    # grid & composite: plain mode leaves a page-spanning merge -> manual, not N
    for build in (gen_collages.grid, gen_collages.composite):
        img = build()
        assert uncollage.split_page(img)[1]
    # line-splitter must NOT over-split cleanly separated photos
    assert _count(gen_collages.separated(), {"split_lines": True}) == 4


def test_process_image_writes_and_queues(tmp_path=None):
    """End-to-end: process_image writes the right file count and queues the hard page."""
    out = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    src = os.path.join(out, "src")
    gen_collages.generate(src)
    for name, _build, flags, expected, needs_manual in SCENARIOS:
        manual = {"entries": {}, "pending": [], "path": os.path.join(out, "m.txt")}
        crops = os.path.join(out, f"crops_{name}")
        os.makedirs(crops, exist_ok=True)          # main() makes out_dir; mirror that
        n = uncollage.process_image(os.path.join(src, f"{name}.png"),
                                    manual=manual, out_dir=crops, **flags)
        if needs_manual:
            assert n == 0 and manual["pending"], f"{name}: should be queued"
        else:
            assert n == expected, f"{name}: wrote {n}, expected {expected}"
            assert len(os.listdir(crops)) == expected


def _crops_pixel_identical():
    """A JPEG source is cropped to PNGs whose pixels equal the decoded source."""
    import cv2
    import numpy as np
    out = tempfile.mkdtemp()
    src = os.path.join(out, "page.jpg")
    cv2.imwrite(src, gen_collages.separated())
    img = uncollage.read_image(src)
    boxes = uncollage.find_subimages(img)
    if uncollage.process_image(src, out_dir=out) != len(boxes):
        return False
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        crop = cv2.imread(os.path.join(out, f"page_{i}.png"), cv2.IMREAD_UNCHANGED)
        if crop is None or not np.array_equal(crop, img[y:y + h, x:x + w]):
            return False
    return True


def _grey_crops_one_channel():
    """On a colour page, crops of grey photos are written as one grey channel
    (same pixel values) and crops of colour photos stay colour."""
    import cv2
    import numpy as np
    page = gen_collages.separated()
    page[:, :600] = cv2.cvtColor(cv2.cvtColor(page[:, :600], cv2.COLOR_BGR2GRAY),
                                 cv2.COLOR_GRAY2BGR)       # left photos -> grey
    out = tempfile.mkdtemp()
    src = os.path.join(out, "mixed.png")
    cv2.imwrite(src, page)
    boxes = uncollage.find_subimages(page)
    uncollage.process_image(src, out_dir=out)
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        raw = cv2.imread(os.path.join(out, f"mixed_{i}.png"), cv2.IMREAD_UNCHANGED)
        grey = x + w <= 600
        if (raw.ndim == 2) != grey:
            return False
        if not np.array_equal(cv2.imread(os.path.join(out, f"mixed_{i}.png")),
                              page[y:y + h, x:x + w]):
            return False
    return True


def _boxes_contain_photos():
    """Every crop box fully contains its photo (padded, never clipped)."""
    ph, pw, gap, m = 360, 520, 60, 40              # separated() layout
    truth = [(m + c * (pw + gap), m + r * (ph + gap), pw, ph)
             for r in (0, 1) for c in (0, 1)]
    boxes = uncollage.find_subimages(gen_collages.separated())
    return len(boxes) == 4 and all(
        bx <= tx and by <= ty and bx + bw >= tx + tw and by + bh >= ty + th
        for (bx, by, bw, bh), (tx, ty, tw, th) in zip(boxes, truth))


def _order_ok():
    """Photos side by side are numbered left-to-right even when their tops
    differ a little; rows run top-to-bottom."""
    boxes = [(600, 1000, 400, 500), (100, 1060, 400, 500),   # row 2, right top higher
             (600, 40, 400, 500), (100, 0, 400, 500)]        # row 1
    return uncollage._order_boxes(boxes) == [boxes[3], boxes[2], boxes[1], boxes[0]]


def _debug_sheets_ok():
    """--debug pages its cells into sheets of DEBUG_PER_SHEET."""
    page = gen_collages.separated()
    cell = uncollage._debug_tile("p.png", page, uncollage.find_subimages(page), "auto")
    out = tempfile.mkdtemp()
    n = uncollage.DEBUG_PER_SHEET + 1
    paths = uncollage.write_debug_sheets([cell] * n, out)
    return len(paths) == 2 and all(os.path.exists(p) for p in paths)


def _corner_box_tight():
    """The scanner-edge line does not stretch the corner photo's box."""
    boxes = uncollage.find_subimages(gen_collages.scanner_corner())
    return len(boxes) == 1 and boxes[0][2] < 900               # photo is 820 wide


def test_corner_box_tight():
    assert _corner_box_tight()


def _no_overwrite_ok():
    """Existing crops are kept by default and replaced with overwrite=True."""
    import cv2
    out = tempfile.mkdtemp()
    src = os.path.join(out, "page.png")
    cv2.imwrite(src, gen_collages.separated())
    uncollage.process_image(src, out_dir=out)
    crop = os.path.join(out, "page_1.png")
    with open(crop, "wb") as f:
        f.write(b"marker")                          # stand-in for a user's file
    uncollage.process_image(src, out_dir=out)
    kept = open(crop, "rb").read() == b"marker"
    uncollage.process_image(src, out_dir=out, overwrite=True)
    return kept and open(crop, "rb").read() != b"marker"


def _failed_page_unpublished():
    """If any crop of a page fails to write, none is published (so a re-run,
    which skips pages that have crops, redoes it) and no temp file remains."""
    import cv2
    out = tempfile.mkdtemp()
    page = gen_collages.separated()
    real, calls = cv2.imwrite, []

    def flaky(path, *a):
        calls.append(path)
        return False if len(calls) == 3 else real(path, *a)
    cv2.imwrite = flaky
    try:
        n = uncollage._write_crops(os.path.join(out, "page.png"), page,
                                   uncollage.find_subimages(page), out)
    finally:
        cv2.imwrite = real
    return n == 0 and os.listdir(out) == []


def test_no_overwrite():
    assert _no_overwrite_ok()


def test_failed_page_unpublished():
    assert _failed_page_unpublished()


def test_order_boxes():
    assert _order_ok()


def test_debug_sheets():
    assert _debug_sheets_ok()


def test_crops_pixel_identical():
    assert _crops_pixel_identical()


def test_grey_crops_one_channel():
    assert _grey_crops_one_channel()


def test_boxes_contain_photos():
    assert _boxes_contain_photos()


# --- standalone runner --------------------------------------------------------
def _main():
    print(f"uncollage regression suite  (cv2 {getattr(__import__('cv2'), '__version__', '?')})\n")
    rows, ok = [], True
    for name, build, flags, expected, needs_manual in SCENARIOS:
        img = build()
        boxes, queued = uncollage.split_page(img, **flags)
        got = "queued" if queued else len(boxes)
        want = "queued" if needs_manual else expected
        passed = (got == want)
        ok &= passed
        flag_s = " ".join(f"--{k.replace('_', '-')}" for k, v in flags.items() if v) or "(plain)"
        rows.append((name, flag_s, want, got, "PASS" if passed else "FAIL"))

    # negative controls
    def check(label, cond):
        nonlocal ok
        ok &= cond
        rows.append((label, "control", "ok", "ok" if cond else "BAD",
                     "PASS" if cond else "FAIL"))

    check("touching w/o -r", _count(gen_collages.touching(), {}) == 1)
    check("separated w/ -l", _count(gen_collages.separated(), {"split_lines": True}) == 4)
    check("pixel-identical", _crops_pixel_identical())
    check("box holds photo", _boxes_contain_photos())
    check("grey crop 1-ch", _grey_crops_one_channel())
    check("crop numbering", _order_ok())
    check("edge line ignored", _corner_box_tight())
    check("keep existing", _no_overwrite_ok())
    check("failed page", _failed_page_unpublished())
    check("debug sheets", _debug_sheets_ok())

    w = max(len(r[0]) for r in rows)
    print(f"{'scenario'.ljust(w)}  {'flags':<14} {'want':<8} {'got':<8} result")
    print("-" * (w + 40))
    for name, flag_s, want, got, res in rows:
        print(f"{name.ljust(w)}  {flag_s:<14} {str(want):<8} {str(got):<8} {res}")
    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
