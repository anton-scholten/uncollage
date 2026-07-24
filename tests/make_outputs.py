#!/usr/bin/env python3
"""
Render the regression suite's images to disk for inspection.

For every scenario in gen_collages.SCENARIOS this writes, under tests/output/:

    tests/output/<scenario>/<scenario>.png        the input collage
    tests/output/<scenario>/<scenario>_1.png ...   the sub-images uncollage crops
                                                    (with the scenario's flags)

The `hard` scenario produces no crops -- it is queued for the manual fallback --
so its folder holds only the input collage plus a QUEUED.txt marker.

    python tests/make_outputs.py

These files are committed so the expected results are visible without running
anything; regenerate them after any change that should alter the crops.
"""
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import uncollage
from gen_collages import SCENARIOS

OUT = os.path.join(HERE, "output")


def main():
    for name, build, flags, expected, needs_manual in SCENARIOS:
        folder = os.path.join(OUT, name)
        os.makedirs(folder, exist_ok=True)
        img = build()
        cv2.imwrite(os.path.join(folder, f"{name}.png"), img)

        manual = {"entries": {}, "pending": [], "path": os.path.join(folder, "m.txt")}
        n = uncollage.process_image(os.path.join(folder, f"{name}.png"),
                                    manual=manual, out_dir=folder, **flags)
        flag_s = " ".join(f"--{k.replace('_', '-')}" for k, v in flags.items() if v) or "(plain)"
        if needs_manual:
            with open(os.path.join(folder, "QUEUED.txt"), "w") as f:
                f.write("This page cannot be split automatically; uncollage queues "
                        "it for the manual-boxes fallback (draw_boxes.py).\n")
            print(f"{name:<11} {flag_s:<14} -> queued for manual (0 crops)")
        else:
            print(f"{name:<11} {flag_s:<14} -> {n} crop(s)")
    print(f"\nwrote images under {OUT}")


if __name__ == "__main__":
    main()
