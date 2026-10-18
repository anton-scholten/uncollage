#!/usr/bin/env python3
"""
Characterization ("golden") suite over 10 real photo collages from Wikimedia
Commons -- genuine city montages (grids of butted rectangular photos), fetched
by fetch_real_collages.py into tests/real_collages/.

Unlike the synthetic / generated-real suites, these have no clean ground truth
we control, and several are genuinely hard (photos butted with no light seam, or
dark night shots whose edges read like the shadow between prints). So this suite
runs each collage through uncollage in its default mode (which, since the
background-surface rework, beats --touching on these: 7/10 exact, the rest
safely queued) and pins the CURRENT outcome -- a crop count, or "queued" when
the page merges and would go to the manual fallback. The point is to flag any
change in behavior; the `TRUTH` column (approximate, counted by eye) shows how
far the current result is from the real number of photos.

Findings at pin time: 7 of 10 montages split to the exact photo count and the
other 3 queue for the manual fallback (tight or dark seams); none lands on a
wrong count -- see the table the runner prints.

    python tests/fetch_real_collages.py     # once, to download the images
    python tests/test_real_collages.py
    pytest tests/test_real_collages.py
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cv2
import uncollage

COLLAGE_DIR = os.path.join(HERE, "real_collages")

# Pinned pipeline outcome in default mode: crop count, or "queued" when the
# page merges into a page-spanning blob and would go to the manual fallback.
GOLDEN = {
    "01_bishkek": 5,
    "02_melbourne": 6,
    "03_nyc": "queued",
    "04_quebec": 7,
    "05_saltlake": 7,
    "06_warsaw": "queued",
    "07_johannesburg": 6,
    "08_doha": "queued",
    "09_fukuoka": 7,
    "10_vyborg": 8,
}

# Number of real photos in each montage (counted by eye) -- shown for context
# only, NOT asserted. The split ones were verified against the images; the ones
# that merge/queue are approximate (exact count doesn't change the outcome).
TRUTH = {
    "01_bishkek": 5, "02_melbourne": 6, "03_nyc": 8, "04_quebec": 7,
    "05_saltlake": 7, "06_warsaw": 6, "07_johannesburg": 6, "08_doha": 7,
    "09_fukuoka": 7, "10_vyborg": 8,
}


def _outcome(img):
    boxes = uncollage.find_subimages(img)
    return "queued" if uncollage._needs_manual(img, boxes) else len(boxes)


def _have_pool():
    return bool(glob.glob(os.path.join(COLLAGE_DIR, "*.jpg")))


# --- pytest-style test --------------------------------------------------------
def test_real_collages_match_golden():
    if not _have_pool():
        import pytest  # type: ignore
        pytest.skip("run tests/fetch_real_collages.py first")
    for slug, expected in GOLDEN.items():
        img = cv2.imread(os.path.join(COLLAGE_DIR, f"{slug}.jpg"))
        assert img is not None, f"missing image: {slug}.jpg"
        assert _outcome(img) == expected, f"{slug}: outcome changed vs golden"


# --- standalone runner --------------------------------------------------------
def _main():
    if not _have_pool():
        print("no images in tests/real_collages/ -- run: "
              "python tests/fetch_real_collages.py")
        return 2
    rows, ok, split_ok = [], True, 0
    for slug, expected in GOLDEN.items():
        img = cv2.imread(os.path.join(COLLAGE_DIR, f"{slug}.jpg"))
        got = _outcome(img)
        passed = (got == expected)
        ok &= passed
        truth = TRUTH[slug]
        hit = (got == truth)
        split_ok += hit
        rows.append((slug, truth, got, expected,
                     "PASS" if passed else "FAIL", "OK" if hit else "-"))

    w = max(len(r[0]) for r in rows)
    print("uncollage REAL-collage characterization suite  (default mode)\n")
    print(f"{'collage'.ljust(w)}  {'~photos':>7} {'got':>7} {'golden':>7}  "
          f"{'regress':>7} {'vs truth':>8}")
    print("-" * (w + 42))
    for slug, truth, got, exp, res, hit in rows:
        print(f"{slug.ljust(w)}  {truth:>7} {str(got):>7} {str(exp):>7}  "
              f"{res:>7} {hit:>8}")
    queued = sum(1 for _, _, got, *_ in rows if got == "queued")
    print(f"\n{split_ok}/10 montages split to the exact photo count; "
          f"{queued}/10 merge and queue for manual (tight or dark seams); "
          f"the rest split to a near-but-wrong count.")
    print("golden regression check: " + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
