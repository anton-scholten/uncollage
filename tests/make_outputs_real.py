#!/usr/bin/env python3
"""
Render the real-photo suite's images to disk under tests/output_real/.

For every scenario: the input collage plus the sub-images uncollage crops from
it (with the scenario's flags). Cases that queue for manual get a QUEUED.txt.

    python tests/fetch_photos.py       # once, to get the photo pool
    python tests/make_outputs_real.py
"""
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import uncollage
from gen_collages_real import SCENARIOS_REAL

OUT = os.path.join(HERE, "output_real")


def main():
    for name, build, flags, expected, needs_manual in SCENARIOS_REAL:
        folder = os.path.join(OUT, name)
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, f"{name}.png"), build())
        manual = {"entries": {}, "pending": [], "path": os.path.join(folder, "m.txt")}
        n = uncollage.process_image(os.path.join(folder, f"{name}.png"),
                                    manual=manual, out_dir=folder, overwrite=True, **flags)
        if needs_manual:
            with open(os.path.join(folder, "QUEUED.txt"), "w") as f:
                f.write("Cannot be split automatically on real photos; uncollage "
                        "queues this for the manual-boxes fallback.\n")
            print(f"{name:<11} -> queued for manual (0 crops)")
        else:
            print(f"{name:<11} -> {n} crop(s)")
    print(f"\nwrote images under {OUT}")


if __name__ == "__main__":
    main()
