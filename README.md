<p align="center">
<img src="Logo/Logo.png" width="700" title="Uncollage">
</p>

# Uncollage

Separates scanned pages or collages into their individual photos (prints, postcards).
Detection is based on the paper/scanner background around the photos -- any colour, and uneven lighting is followed -- so photos may touch each other and the page edge, and may be rotated or irregularly shaped. Each photo is cropped as an upright rectangle around it (no rotation), with a small margin so no part of it is cut off; neighbouring crops may overlap. Crops are **pixel-identical** to the source.

This toolkit has three programs:

| Program | What it does |
|---|---|
| `uncollage.py` | Main tool: automatic splitting + fallback to a box editor for pages it cannot split. |
| `draw_boxes.py` | Simple GUI to draw and edit bounding boxes for sub-images to be cropped from a collage. Can be used standalone. |
| `sam_uncollage.py` | Optional code to allow splitting using Meta's pretrained Segment Anything model. |

## Requirements

- Core: `opencv-python` (cv2) and `numpy`.
- For `sam_uncollage.py` only: `torch`, `torchvision`, `segment-anything`, and the ViT-B checkpoint `sam_vit_b_01ec64.pth` (saved as `sam_vit_b.pth`).

---

## uncollage.py

```
python uncollage.py INPUT [OUTPUT]
```

`INPUT` may be:
- an **image file**;
- a **folder** → every image in it is processed (non-recursive).
  Files that look like crops this tool already wrote (`<original_image>_<n>`) are skipped, so re-running a folder does not re-crop its own outputs;
- a **text file** → each line is an image path (relative paths based on the text file's folder; blank lines and `#` comments are ignored).

`OUTPUT` is the folder for the crops (created if needed); without it, crops go next to each source image.

Crops are named `<original_image>_1`, `<original_image>_2`, ... numbered row by row (top-to-bottom), left-to-right within a row. They are pixel-identical to the source: a lossless source (PNG, TIFF, BMP) keeps its format, while a lossy one (JPEG, WebP) is saved as **PNG**, because re-encoding it would change the pixels. A crop whose colour channels are exactly equal (a grey photo, or a grey scan stored as RGB) is saved as a one-channel image with the same pixel values, a third of the size; each crop is judged on its own. (On a colour JPEG page a grey photo usually carries a little chroma noise, so it stays RGB to remain pixel-identical.) Bit depth and alpha are kept; EXIF orientation is not applied (the pixels are cropped as stored).

### Options

Normally none are needed.

| Option | Meaning |
|---|---|
| `--touching`, `-t` | Also cut photos that are butted together with **no** gap: along straight shadow seams, and along full-span straight print edges. Off by default, because on ordinary scans it can cut a photo along its own straight horizon or edge. Use it for scans you know are butted composites. |
| `--debug`, `-d` | Also write contact sheets (`uncollage_debug_01.jpg`, ...) into an `uncollage_debug/` subfolder of `OUTPUT` (or of the input's folder): each page as a thumbnail with its boxes numbered like the crop files, 12 pages per sheet. Box colours: red = cropped automatically, magenta = queued for manual boxes (showing what was detected), green = boxes from the manual file, blue = boxes drawn in this run. |
| `--no-draw` | Do not open the box editor for failed pages; just list them in the manual file (for headless/batch use). |
| `--manual FILE` | Path to the manual-boxes file (default `manual_boxes.txt`). |

### How it works

1. Detection runs on a copy scaled down to at most 3000 px (a 1200 dpi A4 scan is ~14000 px), which is faster and ignores film grain; boxes are mapped back to full resolution.
2. **Background:** a smooth quadratic surface is fitted to the page's edge pixels, ignoring photos that run off the edge and dark scanner-lid strips. This handles any paper colour and lighting gradients. If no background surrounds the page (a frameless montage), the page's most frequent colour (its seams) is used instead.
3. A pixel is background when it is smooth, close to that surface, and connected to the page border. "Close" is set from how much the page's own paper strays from the surface, so on clean scanner paper a print's off-white border still counts as part of the print. Pale areas enclosed by a photo (sky, snow) stay part of the photo.
4. Each large foreground blob is a photo. A blob that is not rectangle-shaped (photos touching at a corner, or bridged by the shadow a thick print casts) is eroded until it falls apart into one core per photo; every pixel then goes to its nearest core, so each photo keeps its full extent. The split is kept only if every resulting piece is clearly rectangular.

### Manual-boxes fallback (for pages that can't be split automatically)

Some pages can't be split automatically (e.g. faded, borderless photos on
wallpaper with edges difficult to determine). Uncollage handles these
in one run:

1. **Pass 1** processes every input; clean pages are cropped, and any page whose
   automatic split looks merged/failed is queued.
2. **Pass 2** opens the box editor (`draw_boxes.py`) and user draws one box per
   sub-image for each image (sub-images cropped immediately and box coordinates saved after each image so later runs can reuse the values).

With `--no-draw` (or a headless machine with no display), the second pass is skipped and the queued pages are listed in the manual file to be manually drawn later with:

```
python uncollage.py scans/ crops/ --no-draw   # crops what it can, queues the rest
python draw_boxes.py manual_boxes.txt          # draw boxes for the queued pages
python uncollage.py scans/ crops/              # re-run: cached boxes are used
```

### Examples

```
python uncollage.py page.jpg              # one page, crops beside it
python uncollage.py scans/ crops/         # whole folder -> crops/
python uncollage.py montages/ crops/ -t   # butted composites with no gaps
python uncollage.py list.txt --no-draw    # batch, no GUI
python uncollage.py scans/ crops/ -d      # + contact sheets in crops/uncollage_debug/
```

---

## draw_boxes.py

Draw/edit one box per photo. Two modes:

```
python draw_boxes.py IMG [-o boxes.txt] [--run]     # single image -> boxes file
python draw_boxes.py manual_boxes.txt [--redo]      # fill uncollage.py's list
```

- **Single-image mode** writes the boxes to a file (`-o`, default `boxes.txt`).
  With `--run` it then calls `sam_uncollage.py IMG --boxes <file>` to crop.
- **List mode** takes uncollage.py's manual-boxes file: each image lacking boxes
  is opened in turn and the coordinates you draw are written back onto its line
  (`--redo` also redraws lines that already have boxes).

### Controls

| Action | How |
|---|---|
| New box | drag on empty space (or inside another box — boxes may overlap) |
| Resize | drag a corner handle |
| Move | drag a box's edge |
| Options | toolbar buttons **Undo / Clear / Save / Cancel** (hover for a tooltip) |
| Keys | `u` undo · `r` clear · `s`/`Enter` save · `q`/`Esc` cancel |

Because only a box's edge or corner grabs it, a drag starting inside a box makes
a **new** box — so overlapping sub-images can each get their own box.

The window is **resizable** — drag its border to enlarge it and zoom into the
image. When processing a queue (the fallback or list mode) the title shows
progress as `current/total - name.ext`, and the boxes file is rewritten **after
every image**, so a crash never loses work already done.

---

## sam_uncollage.py (optional, pretrained model)

Splits using Meta's Segment Anything (SAM). Useful when the background-based
method merges faded/borderless prints, though it is not a guaranteed click-free
solution.

```
python sam_uncollage.py IMG [--debug]                      # automatic masks
python sam_uncollage.py IMG --boxes boxes.txt [--debug]    # box-prompted (reliable)
```

| Option | Meaning |
|---|---|
| `--boxes FILE` | Refine one rough box per photo (`x y w h` per line) into the exact region; keeps your box where the print edge is invisible. Most reliable mode. |
| `--checkpoint PATH` | Path to the SAM checkpoint (default `./sam_vit_b.pth`). |
| `--debug` | Also write `<stem>.sam_debug.png` with the chosen boxes drawn on. |

---

## Notes & limitations

- Photos that physically overlap, or butt together along a whole edge with
  **no** gap, border or shadow between them, come out as one crop (see
  `--touching`). Faded, borderless photos whose edges are invisible in the
  scan, and pages where one blob spans the page, are queued for the
  manual-boxes fallback.
- A print border of exactly the paper's tone (white on white, with no shadow
  or edge line) is invisible and may be cropped off; the photo itself is kept.
- An album page or mount card carrying several photos is treated as one
  photo if the sheet's colour is close to the photos'; such pages are usually
  queued.
- `--touching` can't distinguish a real print seam from a photo's own strong
  straight internal edge purely by geometry; the uniformity check mitigates
  this (uniform sky/sea slices are dropped), but it stays opt-in.
- Crops keep each sub-image's original orientation (upright bounding box); a
  rotated sub-image therefore keeps small background corners.
