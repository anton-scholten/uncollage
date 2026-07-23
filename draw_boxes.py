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
    drag inside / empty    draw a new box (may overlap existing boxes)
    drag a corner handle   resize the box under the cursor
    drag a box's edge      move that box
    toolbar buttons        Undo / Clear / Save / Cancel  (hover for a tooltip)
    keys                   u=undo  r=clear  s/Enter=save  q/Esc=cancel

Overlapping boxes are supported: because only a box's edge (or corner) grabs it,
a drag that starts inside a box makes a new box instead of moving the old one,
so overlapping sub-images can each get their own box.

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
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GRAB = 10               # how close (px) the cursor must be to grab a corner
_HALF = 4                # half-size of a drawn corner handle

# The only options the window needs: id, label, and the hover tooltip text.
_BUTTONS = [
    ("undo", "Undo", "Remove the last box"),
    ("clear", "Clear", "Remove all boxes"),
    ("save", "Save", "Save boxes and close  (Enter)"),
    ("cancel", "Cancel", "Discard changes and close  (Esc)"),
]


def _norm(box):
    """Order a [x0, y0, x1, y1] box so x0<=x1 and y0<=y1."""
    x0, y0, x1, y1 = box
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def _corners(box):
    """Corner points in order TL, TR, BR, BL."""
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _hit_corner(boxes, mx, my, r=_GRAB):
    """Topmost (box index, corner id 0=TL/1=TR/2=BR/3=BL) near the cursor, or None."""
    for i in range(len(boxes) - 1, -1, -1):
        for c, (cx, cy) in enumerate(_corners(_norm(boxes[i]))):
            if abs(mx - cx) <= r and abs(my - cy) <= r:
                return i, c
    return None


def _hit_edge(boxes, mx, my, r=_GRAB):
    """Topmost box whose border (edge, not interior) is within r px of the
    cursor, or None. Only the border grabs a box; the interior stays free so a
    drag there starts a NEW box -- letting the user draw overlapping boxes."""
    for i in range(len(boxes) - 1, -1, -1):
        x0, y0, x1, y1 = _norm(boxes[i])
        on_vert = (abs(mx - x0) <= r or abs(mx - x1) <= r) and y0 - r <= my <= y1 + r
        on_horz = (abs(my - y0) <= r or abs(my - y1) <= r) and x0 - r <= mx <= x1 + r
        if on_vert or on_horz:
            return i
    return None


def _layout_buttons():
    buttons = []
    bx, by, bh = 8, 8, 30
    for bid, label, tip in _BUTTONS:
        (tw, _), _ = cv2.getTextSize(label, _FONT, 0.55, 1)
        bw = tw + 22
        buttons.append({"id": bid, "label": label, "tip": tip,
                        "rect": (bx, by, bw, bh)})
        bx += bw + 8
    return buttons


def _hit_button(buttons, mx, my):
    for b in buttons:
        bx, by, bw, bh = b["rect"]
        if bx <= mx <= bx + bw and by <= my <= by + bh:
            return b["id"]
    return None


def _tooltip(vis, text, mx, my):
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.5, 1)
    pad = 6
    x = min(mx + 14, vis.shape[1] - tw - 2 * pad - 2)
    y = min(my + 18, vis.shape[0] - th - 2 * pad - 2)
    cv2.rectangle(vis, (x, y), (x + tw + 2 * pad, y + th + 2 * pad), (30, 30, 30), -1)
    cv2.rectangle(vis, (x, y), (x + tw + 2 * pad, y + th + 2 * pad), (200, 200, 200), 1)
    cv2.putText(vis, text, (x + pad, y + pad + th), _FONT, 0.5, (255, 255, 255), 1)


def draw(image_path):
    """Open a window to draw/edit one box per photo. Returns [(x,y,w,h), ...] in
    source pixels on save, or None if cancelled.

    Drag to add a box; drag a corner handle to resize; drag a box's EDGE to move
    it. Only the edge/corner grabs a box, so a drag starting inside a box makes a
    new (possibly overlapping) box -- letting overlapping sub-images each get one.
    The toolbar buttons (Undo / Clear / Save / Cancel) show a tooltip on hover;
    the same actions are on u / r / Enter / Esc.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: could not read image: {image_path}")
        return None
    h, w = img.shape[:2]
    scale = min(1.0, MAX_VIEW / max(h, w))
    view0 = cv2.resize(img, (int(round(w * scale)), int(round(h * scale)))) \
        if scale < 1.0 else img.copy()
    vh, vw = view0.shape[:2]
    buttons = _layout_buttons()

    boxes = []               # each a [x0, y0, x1, y1] in VIEW coords
    # mode: None | 'new' | 'resize' | 'move' | 'button'
    state = {"mode": None, "idx": -1, "corner": -1, "off": (0, 0),
             "mx": -1, "my": -1, "btn": None, "action": None}

    def render():
        vis = view0.copy()
        active_edit = state["mode"] in ("new", "resize", "move")
        for i, box in enumerate(boxes, 1):
            x0, y0, x1, y1 = _norm(box)
            colour = (0, 200, 0) if (state["mode"] != "button"
                                     and i - 1 == state["idx"]) else (0, 0, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), colour, 2)
            cv2.putText(vis, str(i), (x0 + 4, y0 + 20), _FONT, 0.6, colour, 2)
            for cx, cy in _corners([x0, y0, x1, y1]):
                cv2.rectangle(vis, (cx - _HALF, cy - _HALF),
                              (cx + _HALF, cy + _HALF), (255, 255, 255), -1)
                cv2.rectangle(vis, (cx - _HALF, cy - _HALF),
                              (cx + _HALF, cy + _HALF), (0, 0, 255), 1)
        hovering = _hit_button(buttons, state["mx"], state["my"]) \
            if not active_edit else None
        for b in buttons:
            bx, by, bw, bh = b["rect"]
            bg = (110, 110, 110) if b["id"] == hovering else (70, 70, 70)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), bg, -1)
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (220, 220, 220), 1)
            (tw, th), _ = cv2.getTextSize(b["label"], _FONT, 0.55, 1)
            cv2.putText(vis, b["label"], (bx + (bw - tw) // 2, by + (bh + th) // 2),
                        _FONT, 0.55, (255, 255, 255), 1)
        if hovering:
            tip = next(b["tip"] for b in buttons if b["id"] == hovering)
            _tooltip(vis, tip, state["mx"], state["my"])
        return vis

    def on_mouse(event, mx, my, flags, _):
        mx = max(0, min(mx, vw - 1))
        my = max(0, min(my, vh - 1))
        state["mx"], state["my"] = mx, my
        if event == cv2.EVENT_LBUTTONDOWN:
            b = _hit_button(buttons, mx, my)
            if b is not None:
                state.update(mode="button", btn=b)
                return
            hc = _hit_corner(boxes, mx, my)
            if hc is not None:
                state.update(mode="resize", idx=hc[0], corner=hc[1])
                return
            he = _hit_edge(boxes, mx, my)          # only the border grabs a box
            if he is not None:
                x0, y0, _, _ = _norm(boxes[he])
                state.update(mode="move", idx=he, off=(mx - x0, my - y0))
                return
            boxes.append([mx, my, mx, my])         # interior/empty -> new box
            state.update(mode="new", idx=len(boxes) - 1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if state["mode"] == "new":
                boxes[state["idx"]][2:] = [mx, my]
            elif state["mode"] == "resize":
                box = boxes[state["idx"]]
                box[0 if state["corner"] in (0, 3) else 2] = mx
                box[1 if state["corner"] in (0, 1) else 3] = my
            elif state["mode"] == "move":
                box = boxes[state["idx"]]
                bw, bh = box[2] - box[0], box[3] - box[1]
                ox, oy = state["off"]
                nx = max(0, min(mx - ox, vw - 1 - bw))
                ny = max(0, min(my - oy, vh - 1 - bh))
                boxes[state["idx"]] = [nx, ny, nx + bw, ny + bh]
        elif event == cv2.EVENT_LBUTTONUP:
            if state["mode"] == "button":
                if _hit_button(buttons, mx, my) == state["btn"]:
                    state["action"] = state["btn"]
            elif state["mode"] in ("new", "resize", "move"):
                i = state["idx"]
                boxes[i] = _norm(boxes[i])
                x0, y0, x1, y1 = boxes[i]
                if x1 - x0 <= 5 or y1 - y0 <= 5:      # discard degenerate box
                    boxes.pop(i)
            state.update(mode=None, idx=-1, corner=-1, btn=None)

    win = "draw boxes -- " + os.path.basename(image_path)
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    saved = False
    while True:
        cv2.imshow(win, render())
        action, state["action"] = state["action"], None
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("s"), 13):
            action = "save"
        elif k in (ord("q"), 27):
            action = "cancel"
        elif k == ord("u"):
            action = "undo"
        elif k == ord("r"):
            action = "clear"
        if action == "save":
            saved = True; break
        if action == "cancel":
            break
        if action == "undo" and boxes:
            boxes.pop()
        if action == "clear":
            boxes.clear()
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:   # window closed
            break
    cv2.destroyAllWindows()
    if not saved:
        return None
    inv = 1.0 / scale
    out = []
    for x0, y0, x1, y1 in (_norm(b) for b in boxes):
        out.append((int(round(x0 * inv)), int(round(y0 * inv)),
                    int(round((x1 - x0) * inv)), int(round((y1 - y0) * inv))))
    return out


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
