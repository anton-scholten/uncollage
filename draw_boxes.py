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

The window opens showing the whole page and is resizable. In list mode the
title shows progress as "current/total - name.ext", and the file is rewritten
after every image so a crash never loses drawn boxes.

Controls (in the window):
    drag inside / empty    draw a new box (may overlap existing boxes)
    drag a corner handle   resize the box under the cursor
    drag a box's edge      move that box
    mouse wheel            zoom in/out around the cursor
    right/middle drag      pan
    toolbar buttons        Undo / Clear / Save / Cancel  (hover for a tooltip)
    keys                   u=undo  r=clear  s/Enter=save  q/Esc=cancel
                           f=whole page  w=fit width  +/-=zoom

The boxes automatic detection finds are shown to start with (one box around
the whole image if it finds none), so usually you only adjust them. Every edit
is saved at once to a draft file next to the boxes file (`<file>.draft`), so an
interrupted session loses nothing; the draft is offered again next time.

Overlapping boxes are supported: because only a box's edge (or corner) grabs it,
a drag that starts inside a box makes a new box instead of moving the old one,
so overlapping sub-images can each get their own box.

Boxes are written one "x y w h" per line in SOURCE-image pixels. With --run,
sam_uncollage.py is invoked on the image with the saved boxes afterwards.

The window is plain Tk (Python's standard tkinter), so it uses the OS's fonts
and needs no GUI build of OpenCV.

Requires: opencv-python (cv2), numpy, tkinter. Optional: Pillow (faster redraws).
"""

import os
import subprocess
import sys
import tkinter as tk

import cv2
import numpy as np

try:                     # optional: Pillow hands frames to Tk ~4x faster than PPM
    from PIL import Image, ImageTk
except ImportError:
    ImageTk = None

MAX_VIEW = 3000          # working buffer long side: the detail available when zoomed
INIT_VIEW = 950          # initial window long side (fits a laptop screen)
ZOOM_STEP = 1.2          # zoom factor per wheel notch
ZOOM_MAX = 8.0           # max screen px per buffer px
BUTTON_FONT = ("TkDefaultFont", 12)
TIP_FONT = ("TkDefaultFont", 12)
TIP_DX, TIP_DY = 40, 24  # tooltip offset from the cursor (clear of a big cursor)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_GRAB = 16               # how close (screen px) the cursor must be to grab a corner
_HALF = 8                # half-size of a drawn corner handle
MARGIN = 40              # screen px of background kept around the page, so a box
                         # around the whole page can start outside it

# The only options the window needs: id, label, and the hover tooltip text.
_BUTTONS = [
    ("undo", "Undo", "Remove the last box  (u)"),
    ("clear", "Clear", "Remove all boxes  (r)"),
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


class _Tooltip:
    """A small OS-font tooltip shown beside the cursor while over a widget."""

    def __init__(self, widget, text):
        self.widget, self.text, self.win, self.job = widget, text, None, None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _):
        self.job = self.widget.after(400, self._show)

    def _show(self):
        x = self.widget.winfo_pointerx() + TIP_DX
        y = self.widget.winfo_pointery() + TIP_DY
        self.win = tk.Toplevel(self.widget)
        self.win.wm_overrideredirect(True)
        self.win.wm_geometry(f"+{x}+{y}")
        tk.Label(self.win, text=self.text, font=TIP_FONT, bg="#2b2b2b", fg="white",
                 padx=8, pady=4, relief="solid", borderwidth=1).pack()

    def _hide(self, _=None):
        if self.job:
            self.widget.after_cancel(self.job)
            self.job = None
        if self.win:
            self.win.destroy()
            self.win = None


def _load_drafts(path):
    """{image path: [(x, y, w, h), ...]} from a draft file ("PATH x y w h ...")."""
    drafts = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if len(parts) >= 1 and (len(parts) - 1) % 4 == 0:
                nums = [int(round(float(v))) for v in parts[1:]]
                drafts[parts[0]] = [tuple(nums[i:i + 4]) for i in range(0, len(nums), 4)]
    return drafts


def _save_draft(path, image_path, boxes):
    """Record (boxes) or drop (None) one image's unsaved boxes in the draft file.
    Written to a temp file and renamed, so an interrupt never leaves it torn."""
    if not path:
        return
    key = os.path.abspath(image_path)
    drafts = _load_drafts(path)
    if boxes is None:
        drafts.pop(key, None)
    else:
        drafts[key] = boxes
    if not drafts:
        if os.path.exists(path):
            os.remove(path)
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for k, bs in drafts.items():
            f.write(k + "".join(f" {x} {y} {w} {h}" for x, y, w, h in bs) + "\n")
    os.replace(tmp, path)


def draw(image_path, index=None, total=None, initial=None, draft_file=None):
    """Open a window to draw/edit one box per photo. Returns [(x,y,w,h), ...] in
    source pixels on save, or None if cancelled. When `index`/`total` are given
    the title is "index/total - name.ext".

    The window opens showing the whole page; the mouse wheel zooms around the
    cursor, a right/middle drag pans, `f` shows the whole page again and `w`
    fits the width. The toolbar stays put above the image.

    Drag to add a box; drag a corner handle to resize; drag a box's EDGE to move
    it. Only the edge/corner grabs a box, so a drag starting inside a box makes a
    new (possibly overlapping) box -- letting overlapping sub-images each get one.
    The toolbar buttons (Undo / Clear / Save / Cancel) show a tooltip on hover;
    the same actions are on u / r / Enter / Esc.

    `initial` boxes (source pixels, e.g. what automatic detection found) are
    shown ready to adjust; with none, one box around the whole image. With a
    `draft_file`, every edit is written there at once, so boxes survive an
    interrupted session (Ctrl+C, closed window) and are offered again next
    time; Save or Cancel removes the image's draft.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        print(f"error: could not read image: {image_path}")
        return None
    h, w = img.shape[:2]
    scale = min(1.0, MAX_VIEW / max(h, w))
    buf = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
    vh, vw = buf.shape[:2]
    # Pre-shrunk copies (1, 1/2, 1/4, ...): a frame is resized from the
    # smallest copy still at least as detailed as the zoom, which keeps every
    # redraw a cheap linear resize instead of a full-page area resize.
    pyramid = [buf]
    while max(pyramid[-1].shape[:2]) > 400:
        pyramid.append(cv2.resize(pyramid[-1], None, fx=0.5, fy=0.5,
                                  interpolation=cv2.INTER_AREA))

    # each a [x0, y0, x1, y1] in BUFFER coords: an unsaved draft first, else
    # the given boxes, else one box around the whole image
    start = _load_drafts(draft_file).get(os.path.abspath(image_path)) or initial \
        or [(0, 0, w, h)]
    boxes = [[min(float(vw), x * scale), min(float(vh), y * scale),
              min(float(vw), (x + bw) * scale), min(float(vh), (y + bh) * scale)]
             for x, y, bw, bh in start]

    def source_boxes():
        inv, out = 1.0 / scale, []
        for x0, y0, x1, y1 in (_norm(b) for b in boxes):
            sx0, sy0 = int(round(x0 * inv)), int(round(y0 * inv))
            sx1, sy1 = min(w, int(round(x1 * inv))), min(h, int(round(y1 * inv)))
            out.append((sx0, sy0, sx1 - sx0, sy1 - sy0))
        return out

    def autosave():
        _save_draft(draft_file, image_path, source_boxes())
    # mode: None | 'new' | 'resize' | 'move' | 'pan'
    st = {"mode": None, "idx": -1, "corner": -1, "off": (0, 0), "pan": (0, 0, 0, 0)}
    # view: z = screen px per buffer px; (ox, oy) = buffer point at the
    # canvas's top-left corner
    view = {"z": 1.0, "ox": 0.0, "oy": 0.0, "cw": 1, "ch": 1, "fit": "page"}
    result = {"saved": False}

    name = os.path.basename(image_path)
    root = tk.Tk()
    root.title(f"{index}/{total} - {name}" if index and total else "draw boxes -- " + name)
    bar = tk.Frame(root, bg="#2d2d2d")
    bar.pack(side="top", fill="x")
    canvas = tk.Canvas(root, bg="#3c3c3c", highlightthickness=0, cursor="crosshair")
    canvas.pack(side="top", fill="both", expand=True)

    def act(action):
        if action == "undo" and boxes:
            boxes.pop()
        elif action == "clear":
            boxes.clear()
        elif action in ("save", "cancel", "close"):
            result["saved"] = action == "save"
            result["boxes"] = source_boxes()
            if action != "close":                # closing the window keeps the
                _save_draft(draft_file, image_path, None)  # draft for next time
            root.destroy()
            return
        autosave()
        redraw_boxes()

    for bid, label, tip in _BUTTONS:
        btn = tk.Button(bar, text=label, font=BUTTON_FONT, padx=14, pady=4,
                        command=lambda a=bid: act(a))
        btn.pack(side="left", padx=6, pady=6)
        _Tooltip(btn, tip)
    # open at a screen-friendly size, never narrower than the toolbar
    root.update_idletasks()
    bar_w, bar_h = bar.winfo_reqwidth(), bar.winfo_reqheight()
    s0 = min(1.0, INIT_VIEW / max(vw, vh))
    root.minsize(bar_w, bar_h + 200)
    ww = max(bar_w, int(vw * s0) + 2 * MARGIN)
    wh = max(400, int(vh * s0) + 2 * MARGIN) + bar_h
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{ww}x{wh}+{max(0, (sw - ww) // 2)}+{max(0, (sh - wh) // 2)}")

    def to_buf(sx, sy):
        return sx / view["z"] + view["ox"], sy / view["z"] + view["oy"]

    def to_scr(bx, by):
        return (bx - view["ox"]) * view["z"], (by - view["oy"]) * view["z"]

    def clamp():
        """Keep the image in view, allowing MARGIN px of background beyond its
        edges; centre it along an axis where it is smaller than the window."""
        m = MARGIN / view["z"]
        for o, size, extent in (("ox", view["cw"], vw), ("oy", view["ch"], vh)):
            span = size / view["z"]
            view[o] = (extent - span) / 2 if span >= extent + 2 * m \
                else min(max(view[o], -m), extent - span + m)

    def fit(mode):
        cw, ch = max(1, view["cw"] - 2 * MARGIN), max(1, view["ch"] - 2 * MARGIN)
        view["z"] = cw / vw if mode == "width" else min(cw / vw, ch / vh)
        view["ox"] = view["oy"] = -MARGIN / view["z"]
        view["fit"] = mode
        clamp()
        request_render()

    def zoom_at(factor, sx, sy):
        """Zoom keeping the buffer point under screen (sx, sy) fixed."""
        bx, by = to_buf(sx, sy)
        lo = 0.5 * min(view["cw"] / vw, view["ch"] / vh)   # half of whole-page
        view["z"] = min(ZOOM_MAX, max(lo, view["z"] * factor))
        view["ox"], view["oy"] = bx - sx / view["z"], by - sy / view["z"]
        view["fit"] = None
        clamp()
        request_render()

    photo = {"img": None, "item": None, "pending": False}

    def request_render():
        """Redraw once the event queue is idle: a burst of wheel notches then
        costs one redraw at the final zoom instead of one per notch."""
        if not photo["pending"]:
            photo["pending"] = True
            root.after_idle(render)

    def render():
        """Redraw the visible part of the page at the current zoom."""
        photo["pending"] = False
        cw, ch, z = view["cw"], view["ch"], view["z"]
        bx0, by0 = max(0.0, view["ox"]), max(0.0, view["oy"])
        bx1, by1 = min(vw, view["ox"] + cw / z), min(vh, view["oy"] + ch / z)
        if bx1 > bx0 and by1 > by0:
            lvl = 0                                 # smallest copy with >= detail
            while lvl + 1 < len(pyramid) and z * 2 ** (lvl + 1) <= 1.0:
                lvl += 1
            f = 2 ** lvl
            src = pyramid[lvl]
            ix0, iy0 = int(bx0 / f), int(by0 / f)
            crop = src[iy0:int(np.ceil(by1 / f)), ix0:int(np.ceil(bx1 / f))]
            tw = max(1, int(round(crop.shape[1] * z * f)))
            th = max(1, int(round(crop.shape[0] * z * f)))
            part = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
            if ImageTk is not None:
                photo["img"] = ImageTk.PhotoImage(Image.fromarray(part))
            else:
                photo["img"] = tk.PhotoImage(data=f"P6 {tw} {th} 255 ".encode()
                                             + part.tobytes(), format="PPM")
            sx, sy = to_scr(ix0 * f, iy0 * f)
            pos = (int(round(sx)), int(round(sy)))
            if photo["item"] is None:
                photo["item"] = canvas.create_image(*pos, image=photo["img"], anchor="nw")
                canvas.tag_lower(photo["item"])
            else:
                canvas.coords(photo["item"], *pos)
                canvas.itemconfigure(photo["item"], image=photo["img"], state="normal")
        elif photo["item"] is not None:
            canvas.itemconfigure(photo["item"], state="hidden")
        redraw_boxes()

    def redraw_boxes():
        canvas.delete("box")
        for i, box in enumerate(boxes, 1):
            x0, y0, x1, y1 = _norm(box)
            (sx0, sy0), (sx1, sy1) = to_scr(x0, y0), to_scr(x1, y1)
            col = "#00c800" if i - 1 == st["idx"] and st["mode"] else "#ff2020"
            canvas.create_rectangle(sx0, sy0, sx1, sy1, outline=col, width=2, tags="box")
            canvas.create_text(sx0 + 6, sy0 + 4, text=str(i), anchor="nw", fill=col,
                               font=("TkDefaultFont", 13, "bold"), tags="box")
            for cx, cy in _corners([sx0, sy0, sx1, sy1]):
                canvas.create_rectangle(cx - _HALF, cy - _HALF, cx + _HALF, cy + _HALF,
                                        fill="white", outline="#ff2020", tags="box")

    def on_configure(ev):
        if (ev.width, ev.height) == (view["cw"], view["ch"]):
            return
        view["cw"], view["ch"] = max(1, ev.width), max(1, ev.height)
        if view["fit"]:
            fit(view["fit"])
        else:
            clamp()
            request_render()

    def on_wheel(ev):
        up = ev.num == 4 or getattr(ev, "delta", 0) > 0
        zoom_at(ZOOM_STEP if up else 1 / ZOOM_STEP, ev.x, ev.y)

    def on_press(ev):
        bx, by = to_buf(ev.x, ev.y)
        bx, by = max(0.0, min(bx, float(vw))), max(0.0, min(by, float(vh)))
        r = _GRAB / view["z"]                       # grab distance in buffer px
        hc = _hit_corner(boxes, bx, by, r)
        if hc is not None:
            st.update(mode="resize", idx=hc[0], corner=hc[1])
            return
        he = _hit_edge(boxes, bx, by, r)            # only the border grabs a box
        if he is not None:
            x0, y0, _, _ = _norm(boxes[he])
            st.update(mode="move", idx=he, off=(bx - x0, by - y0))
            return
        boxes.append([bx, by, bx, by])              # interior/empty -> new box
        st.update(mode="new", idx=len(boxes) - 1)

    def on_drag(ev):
        bx, by = to_buf(ev.x, ev.y)
        bx, by = max(0.0, min(bx, float(vw))), max(0.0, min(by, float(vh)))
        if st["mode"] == "new":
            boxes[st["idx"]][2:] = [bx, by]
        elif st["mode"] == "resize":
            box = boxes[st["idx"]]
            box[0 if st["corner"] in (0, 3) else 2] = bx
            box[1 if st["corner"] in (0, 1) else 3] = by
        elif st["mode"] == "move":
            box = boxes[st["idx"]]
            bw, bh = box[2] - box[0], box[3] - box[1]
            ox, oy = st["off"]
            nx = max(0.0, min(bx - ox, vw - bw))
            ny = max(0.0, min(by - oy, vh - bh))
            boxes[st["idx"]] = [nx, ny, nx + bw, ny + bh]
        else:
            return
        redraw_boxes()

    def on_release(_):
        if st["mode"] in ("new", "resize", "move"):
            i = st["idx"]
            boxes[i] = _norm(boxes[i])
            x0, y0, x1, y1 = boxes[i]
            if (x1 - x0) * view["z"] <= 5 or (y1 - y0) * view["z"] <= 5:
                boxes.pop(i)                        # discard degenerate box
            autosave()
        st.update(mode=None, idx=-1, corner=-1)
        redraw_boxes()

    def on_pan_press(ev):
        st.update(mode="pan", pan=(ev.x, ev.y, view["ox"], view["oy"]))

    def on_pan_drag(ev):
        if st["mode"] != "pan":
            return
        x, y, ox, oy = st["pan"]
        view["ox"], view["oy"] = ox - (ev.x - x) / view["z"], oy - (ev.y - y) / view["z"]
        view["fit"] = None
        clamp()
        request_render()

    def on_pan_release(_):
        st.update(mode=None)

    canvas.bind("<Configure>", on_configure)
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    for b in ("2", "3"):
        canvas.bind(f"<ButtonPress-{b}>", on_pan_press)
        canvas.bind(f"<B{b}-Motion>", on_pan_drag)
        canvas.bind(f"<ButtonRelease-{b}>", on_pan_release)
    canvas.bind("<Button-4>", on_wheel)             # X11 wheel
    canvas.bind("<Button-5>", on_wheel)
    canvas.bind("<MouseWheel>", on_wheel)           # Windows / macOS wheel
    keys = {"u": lambda: act("undo"), "r": lambda: act("clear"),
            "s": lambda: act("save"), "Return": lambda: act("save"),
            "q": lambda: act("cancel"), "Escape": lambda: act("cancel"),
            "f": lambda: fit("page"), "w": lambda: fit("width"),
            "plus": lambda: zoom_at(ZOOM_STEP, view["cw"] / 2, view["ch"] / 2),
            "equal": lambda: zoom_at(ZOOM_STEP, view["cw"] / 2, view["ch"] / 2),
            "minus": lambda: zoom_at(1 / ZOOM_STEP, view["cw"] / 2, view["ch"] / 2)}
    root.bind("<Key>", lambda ev: keys.get(ev.keysym, lambda: None)())
    root.protocol("WM_DELETE_WINDOW", lambda: act("close"))
    root.focus_force()
    root.mainloop()

    return result["boxes"] if result["saved"] else None


def _detected(image_path):
    """What uncollage's automatic detection finds -- the editor's starting
    boxes (an empty result falls back to one box around the whole image)."""
    import uncollage
    img = uncollage.read_image(image_path)
    return uncollage.find_subimages(img) if img is not None else None


def run_manual_file(list_path, redo=False):
    """Manual-list mode: `list_path` is uncollage.py's manual-boxes file, one
    image path per line (optionally already followed by boxes). Draw boxes for
    every line that has none (or all, with redo=True) and write the coordinates
    back onto the same lines, preserving comments and order."""
    if not os.path.exists(list_path):
        print(f"error: file not found: {list_path}")
        return 1
    base = os.path.dirname(os.path.abspath(list_path))
    draft_file = list_path + ".draft"
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

    todo = [(item[1], item[2]) for item in parsed
            if item[0] == "img" and (redo or not item[2])]
    if not todo:
        print("all listed images already have boxes (use --redo to redraw).")
        return 0

    results = {}

    def flush():
        """Rewrite the whole list file from parsed + results (called after every
        image so a crash never loses boxes already drawn)."""
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
                    f.write(name + "\n")      # skipped -> stays pending

    total = len(todo)
    for k, (name, existing) in enumerate(todo, 1):
        ip = name if (os.path.isabs(name) or os.path.exists(name)) \
            else os.path.join(base, name)
        print(f"[{k}/{total}] drawing boxes for {ip}  (adjust the boxes; "
              "s/Enter=save, q/Esc=skip)")
        start = [tuple(int(round(float(v))) for v in existing[i:i + 4])
                 for i in range(0, len(existing), 4)] if existing else _detected(ip)
        boxes = draw(ip, k, total, initial=start, draft_file=draft_file)
        if boxes:
            results[name] = boxes
        flush()                               # persist after each image
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
    boxes = draw(image_path, initial=_detected(image_path), draft_file=out + ".draft")
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
