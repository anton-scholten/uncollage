#!/usr/bin/env python3
"""
draw_boxes.py -- draw a rough rectangle around each photo in a scanned collage
and save them as a boxes file for `sam_uncollage.py --boxes`.

For pages whose photos are faded / borderless (invisible edges on shadowed
paper), no automatic method can find them -- you indicate each one here with a
quick drag, and SAM tightens the ones it can while keeping your rectangle for
the rest.

Usage:
    python draw_boxes.py IMG [-o boxes.txt] [--run]     # single image -> boxes file
    python draw_boxes.py manual_boxes.txt [--redo]      # fill uncollage.py's list

  In list mode the argument is uncollage.py's manual-boxes file (one image path
  per line). Each image lacking boxes is opened in turn; the coordinates you
  draw are written back onto its line (add --redo to also redraw lines that
  already have boxes). Then re-run your uncollage.py command to use them.

Controls (in the window):
    drag left mouse   draw a box around one photo
    u                 undo the last box
    r                 reset (clear all)
    s / Enter         save boxes and quit
    q / Esc           quit without saving

Boxes are written one "x y w h" per line in SOURCE-image pixels. With --run,
sam_uncollage.py is invoked on the image with the saved boxes afterwards.

Requires: opencv-python (cv2) with GUI support, numpy.
"""

import os
import subprocess
import sys

# This cv2 build's Qt has no Wayland platform plugin; use XWayland (xcb) so the
# window opens on a Wayland desktop. Harmless on X11. Override by exporting
# QT_QPA_PLATFORM yourself before running.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np

MAX_VIEW = 1100          # window is scaled so its long side is at most this
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def draw(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: could not read image: {image_path}")
        return None
    h, w = img.shape[:2]
    scale = min(1.0, MAX_VIEW / max(h, w))
    view0 = cv2.resize(img, (int(round(w * scale)), int(round(h * scale)))) \
        if scale < 1.0 else img.copy()

    boxes = []                      # in VIEW coords: (x, y, w, h)
    state = {"drawing": False, "x0": 0, "y0": 0, "cur": None}

    def render():
        vis = view0.copy()
        for i, (x, y, bw, bh) in enumerate(boxes, 1):
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(vis, str(i), (x + 3, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if state["cur"] is not None:
            x, y, bw, bh = state["cur"]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 200, 0), 2)
        cv2.putText(vis, f"{len(boxes)} boxes  drag=add  u=undo  r=reset  "
                    "s/Enter=save  q/Esc=quit", (8, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 220), 1)
        return vis

    def on_mouse(event, mx, my, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.update(drawing=True, x0=mx, y0=my, cur=(mx, my, 0, 0))
        elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
            x, y = min(state["x0"], mx), min(state["y0"], my)
            state["cur"] = (x, y, abs(mx - state["x0"]), abs(my - state["y0"]))
        elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
            x, y = min(state["x0"], mx), min(state["y0"], my)
            bw, bh = abs(mx - state["x0"]), abs(my - state["y0"])
            state.update(drawing=False, cur=None)
            if bw > 5 and bh > 5:
                boxes.append((x, y, bw, bh))

    win = "draw boxes -- " + os.path.basename(image_path)
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    saved = False
    while True:
        cv2.imshow(win, render())
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("s"), 13):                 # s or Enter
            saved = True; break
        if k in (ord("q"), 27):                 # q or Esc
            break
        if k == ord("u") and boxes:
            boxes.pop()
        if k == ord("r"):
            boxes.clear()
    cv2.destroyAllWindows()
    if not saved:
        return None
    # map view coords back to source pixels
    inv = 1.0 / scale
    return [(int(round(x * inv)), int(round(y * inv)),
             int(round(bw * inv)), int(round(bh * inv))) for (x, y, bw, bh) in boxes]


def run_manual_file(list_path, redo=False):
    """Manual-list mode: `list_path` is uncollage.py's manual-boxes file, one
    image path per line (optionally already followed by boxes). Draw boxes for
    every line that has none (or all, with redo=True) and write the coordinates
    back onto the same lines, preserving comments and order."""
    if not os.path.exists(list_path):
        print(f"error: file not found: {list_path}")
        return 1
    base = os.path.dirname(os.path.abspath(list_path))
    parsed = []                      # ("raw", text) | ("img", name, [num strs])
    for line in open(list_path, encoding="utf-8").read().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            parsed.append(("raw", line))
            continue
        parts = s.split()
        nums = parts[1:]
        has = len(nums) >= 4 and len(nums) % 4 == 0
        parsed.append(("img", parts[0], nums if has else []))

    results = {}
    needed = False
    for item in parsed:
        if item[0] != "img":
            continue
        name, existing = item[1], item[2]
        if existing and not redo:
            continue
        needed = True
        ip = name if (os.path.isabs(name) or os.path.exists(name)) \
            else os.path.join(base, name)
        print(f"drawing boxes for {ip}  (drag one box per photo; "
              "s/Enter=save, q/Esc=skip)")
        boxes = draw(ip)
        if boxes:
            results[name] = boxes
    if not needed:
        print("all listed images already have boxes (use --redo to redraw).")
        return 0

    with open(list_path, "w", encoding="utf-8") as f:
        for item in parsed:
            if item[0] == "raw":
                f.write(item[1] + "\n")
                continue
            name, existing = item[1], item[2]
            boxes = results.get(name)
            if boxes is not None:
                flat = " ".join(f"{x} {y} {w} {h}" for x, y, w, h in boxes)
                f.write(f"{name} {flat}\n")
            elif existing:
                f.write(name + " " + " ".join(existing) + "\n")
            else:
                f.write(name + "\n")          # skipped -> stays pending
    print(f"updated {list_path}: drew boxes for {len(results)} image(s). "
          "Re-run your uncollage.py command to use them.")
    return 0


def main(argv):
    args = argv[1:]
    run = "--run" in args
    args = [a for a in args if a != "--run"]
    redo = "--redo" in args
    args = [a for a in args if a != "--redo"]
    out = "boxes.txt"
    if "-o" in args:
        i = args.index("-o"); out = args[i + 1]; del args[i:i + 2]
    if len(args) != 1:
        print(__doc__)
        return 2
    target = args[0]
    if os.path.splitext(target)[1].lower() not in IMAGE_EXTS:
        # a list file (uncollage.py's manual-boxes file), not a single image
        return run_manual_file(target, redo=redo)

    image_path = target
    boxes = draw(image_path)
    if not boxes:
        print("no boxes saved.")
        return 1
    with open(out, "w") as f:
        f.write("# x y w h  (one rough rectangle per photo, source pixels)\n")
        for x, y, bw, bh in boxes:
            f.write(f"{x} {y} {bw} {bh}\n")
    print(f"wrote {len(boxes)} boxes to {out}")
    if run:
        here = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, os.path.join(here, "sam_uncollage.py"),
               image_path, "--boxes", out]
        print("running:", " ".join(cmd))
        return subprocess.call(cmd)
    print(f"next: python sam_uncollage.py {image_path} --boxes {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
