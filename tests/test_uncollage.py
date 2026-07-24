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
        boxes = uncollage.find_subimages(img, **flags)
        if needs_manual:
            assert uncollage._needs_manual(img, boxes), \
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
        assert uncollage._needs_manual(img, uncollage.find_subimages(img))
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


# --- standalone runner --------------------------------------------------------
def _main():
    print(f"uncollage regression suite  (cv2 {getattr(__import__('cv2'), '__version__', '?')})\n")
    rows, ok = [], True
    for name, build, flags, expected, needs_manual in SCENARIOS:
        img = build()
        boxes = uncollage.find_subimages(img, **flags)
        got = "queued" if uncollage._needs_manual(img, boxes) else len(boxes)
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

    w = max(len(r[0]) for r in rows)
    print(f"{'scenario'.ljust(w)}  {'flags':<14} {'want':<8} {'got':<8} result")
    print("-" * (w + 40))
    for name, flag_s, want, got, res in rows:
        print(f"{name.ljust(w)}  {flag_s:<14} {str(want):<8} {str(got):<8} {res}")
    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
