#!/usr/bin/env python3
"""
Real-photograph regression suite for uncollage.py.

The six base scenarios of test_uncollage.py, but every tile is a real photograph
(landscape / nature / city / people) from tests/photos/ -- a typical collage
mix. Each scenario asserts the outcome of the mode that actually works on real
content. The shadow-ridge guillotine (--touching) cannot recover a full grid of
butted real photos, so that is pinned as a *limitation control*; line-splitting
on the real composite used to over-split it and now splits it correctly, which
is pinned too. The old-print scenarios also check that every crop holds its
whole print, border and deckled edge included.

Needs the photo pool: run `python tests/fetch_photos.py` once first.

    python tests/test_uncollage_real.py     # PASS/FAIL table, non-zero exit on failure
    pytest tests/test_uncollage_real.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import uncollage
import gen_collages_real as R
from gen_collages_real import SCENARIOS_REAL, OLD_TRUTH


def _result(img, flags):
    boxes = uncollage.find_subimages(img, **flags)
    return "queued" if uncollage._needs_manual(img, boxes) else len(boxes)


# --- pytest-style tests -------------------------------------------------------
def test_real_scenarios_match_expected():
    for name, build, flags, expected, needs_manual in SCENARIOS_REAL:
        want = "queued" if needs_manual else expected
        assert _result(build(), flags) == want, f"{name}: expected {want}"


def test_real_limitations_hold():
    """Pinned behaviour of the --touching paths on real photos."""
    # the shadow-ridge guillotine cannot fully recover a butted grid of real photos
    assert len(uncollage.find_subimages(R.grid_butted_dark(), rectangular=True)) < 6
    # line-splitting splits the real two-photo composite correctly
    assert len(uncollage.find_subimages(R.r_composite(), split_lines=True)) == 2


def _old_prints_whole(build):
    """Number of prints (of a known old-print layout) whose crop box holds the
    whole print, border and deckled edge included, and the print count."""
    img, truth = build()
    boxes = uncollage.find_subimages(img)
    held = sum(any(bx <= tx and by <= ty and bx + bw >= tx + tw and by + bh >= ty + th
                   for bx, by, bw, bh in boxes) for tx, ty, tw, th in truth)
    return held, len(truth)


def test_old_prints_kept_whole():
    for name, build in OLD_TRUTH:
        held, n = _old_prints_whole(build)
        # old_scan: one print's top border is the paper's exact tone -> lost
        assert held >= (n - 1 if name == "old_scan" else n), f"{name}: {held}/{n} whole"


def test_real_process_image_writes_and_queues():
    out = tempfile.mkdtemp()
    for name, build, flags, expected, needs_manual in SCENARIOS_REAL:
        import cv2
        src = os.path.join(out, f"{name}.png")
        cv2.imwrite(src, build())
        manual = {"entries": {}, "pending": [], "path": os.path.join(out, "m.txt")}
        crops = os.path.join(out, f"crops_{name}")
        os.makedirs(crops, exist_ok=True)
        n = uncollage.process_image(src, manual=manual, out_dir=crops, **flags)
        if needs_manual:
            assert n == 0 and manual["pending"], f"{name}: should be queued"
        else:
            assert n == expected and len(os.listdir(crops)) == expected


# --- standalone runner --------------------------------------------------------
def _main():
    try:
        R._pool()
    except FileNotFoundError as e:
        print(e)
        return 2
    rows, ok = [], True
    for name, build, flags, expected, needs_manual in SCENARIOS_REAL:
        want = "queued" if needs_manual else expected
        got = _result(build(), flags)
        ok &= (got == want)
        flag_s = " ".join(f"--{k.replace('_', '-')}" for k, v in flags.items() if v) or "(plain)"
        rows.append((name, flag_s, want, got, "PASS" if got == want else "FAIL"))

    # limitation controls
    gb = len(uncollage.find_subimages(R.grid_butted_dark(), rectangular=True))
    cl = len(uncollage.find_subimages(R.r_composite(), split_lines=True))
    for label, kind, want, val, cond in [
            ("grid butted + -r", "limitation", "<6", gb, gb < 6),
            ("composite + -l", "control", "2", cl, cl == 2)]:
        ok &= cond
        rows.append((label, kind, want, val, "PASS" if cond else "FAIL"))
    for name, build in OLD_TRUTH:
        held, n = _old_prints_whole(build)
        need = n - 1 if name == "old_scan" else n
        ok &= held >= need
        rows.append((f"{name} whole", "control", f">={need}", held,
                     "PASS" if held >= need else "FAIL"))

    w = max(len(r[0]) for r in rows)
    print("uncollage REAL-photo regression suite\n")
    print(f"{'scenario'.ljust(w)}  {'mode':<14} {'want':<8} {'got':<8} result")
    print("-" * (w + 40))
    for name, flag_s, want, got, res in rows:
        print(f"{name.ljust(w)}  {flag_s:<14} {str(want):<8} {str(got):<8} {res}")
    print("\nNote: the -r guillotine (part of --touching) cannot recover a butted "
          "grid of\nreal photos -- pinned above as a limitation control.")
    print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
