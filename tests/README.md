# tests

A headless regression suite for `uncollage.py`. Each scenario exercises one
situation the tool handles. There are two parallel suites:

- **synthetic** (`test_uncollage.py`) — tiles are generated patterns; fully
  self-contained, no network.
- **real photos** (`test_uncollage_real.py`) — the same layouts filled with real
  photographs (landscape / nature / city / people), which stress the detector
  the way genuine scans do.

## Run

```
python tests/test_uncollage.py          # synthetic suite (no network)

python tests/fetch_photos.py            # once: download the real-photo pool
python tests/test_uncollage_real.py     # real-photo suite
```

Each prints a PASS/FAIL table and exits non-zero on failure; both also run under
`pytest` if installed. No display is needed. Only `opencv-python` and `numpy`
(the tool's own deps); `fetch_photos.py` additionally needs network access.

## What it covers

| Scenario | Flags | Expected |
|---|---|---|
| `separated` — photos with paper gaps | *(plain)* | 4 crops |
| `rotated` — tilted photos | *(plain)* | 3 crops (upright bbox each) |
| `touching` — butted with a shadow seam | `--rectangular` | 2 crops |
| `grid` — 2×3 touching photos | `--rectangular` | 6 crops |
| `composite` — 2 photos butted, no gap | `--split-lines` | 2 crops |
| `hard` — faded borderless photo ≈ paper | *(plain)* | queued for manual fallback |
| `tinted` — coloured paper, strong lighting gradient | *(plain)* | 4 crops |
| `dark` — mostly-black photos + scanner-lid strip | *(plain)* | 3 crops |
| `corner` — 2 photos meeting at a corner via a shadow | *(plain)* | 2 crops |

(`--rectangular` and `--split-lines` are the `find_subimages` keywords; the
CLI's `--touching` turns on both.)

Plus controls: `touching` without `--rectangular` stays merged;
`--split-lines` does **not** over-split the cleanly separated page; crops of a
JPEG source are PNGs pixel-identical to the decoded source; every crop box
fully contains its photo (padding, never clipping); on a colour page, crops of
grey photos are written as one channel and colour ones stay colour. An end-to-end test runs
`process_image` and checks the actual files written.

## Real-photo suite

Same six layouts, filled with real photographs. Each scenario uses the mode that
**actually works on real content**, which differs from the synthetic suite in
three cases:

| Scenario | Mode | Expected | vs. synthetic |
|---|---|---|---|
| `separated` | *(plain)* | 4 crops | same |
| `rotated` | *(plain)* | 3 crops | same |
| `touching` | `--rectangular` | 2 crops | needs a near-black seam (real content is as dark as a faint shadow) |
| `grid` | *(plain)* | 6 crops | uses thin **paper gaps**, not a dark seam |
| `composite` | *(plain)* | queued for manual | default mode, not `--split-lines` |
| `hard` | *(plain)* | queued for manual | real photo faded to ≈ paper tone |

Plus three **old-print** scenarios built from the same photo pool: each photo
is rendered as an old print -- greyscale or sepia, faded, with a yellowed or
cream border, deckled (zig-zag) edges, slightly askew, casting a soft shadow --
on a scan page with a lighting falloff and a scanner-lid line:

| Scenario | Mode | Expected |
|---|---|---|
| `old_scan` — 5 grey deckled prints on white paper | *(plain)* | 5 crops |
| `old_sepia` — 4 sepia prints on dark-grey album card, strong falloff | *(plain)* | 4 crops |
| `old_corner` — 2 deckled prints meeting at a corner, shadow-bridged | *(plain)* | 2 crops |

For these the true print outlines are known, and a control checks that each
crop holds its **whole** print, border and deckle included (old_scan allows
one miss: a border that the falloff makes exactly the paper's tone).

Two `--touching` paths are pinned as controls: the `--rectangular`
shadow-ridge guillotine cannot fully recover a grid of butted real photos (a
*limitation*: it caps well below 6), while `--split-lines` now splits the real
composite into exactly its 2 photos. For real scans, prefer visible gaps
(plain mode) or the manual-boxes fallback — which is why `--touching` is opt-in.

## Real-collage suite (found online)

Ten genuine photo collages — city "montages" from Wikimedia Commons (freely
licensed) — as a hard, real-world test set. These are not scans on paper: the
photos are butted together with thin (or no) light seams; the suite runs them
in default mode, which does better on them than `--touching`.

```
python tests/fetch_real_collages.py     # once: download images + ATTRIBUTIONS.md
python tests/test_real_collages.py       # golden/characterization table
```

Because there is no ground truth we control, this is a **characterization
(golden) suite**: it pins each collage's current outcome (a crop count, or
`queued` when the page merges to the manual fallback) and asserts it, so any
behavior change is flagged. A `~photos` column (counted by eye) shows how close
the tool is to the real photo count. At pin time: **7/10** split to the exact
count and the other **3/10** queue for manual (tight or dark seams); none lands
on a wrong count.

## Files

- `gen_collages.py` — synthetic collage builders (`SCENARIOS`). The layouts take
  a pluggable tile provider so the real suite can reuse them.
- `test_uncollage.py` — synthetic assertions; standalone or under pytest.
- `make_outputs.py` / `output/` — committed synthetic gallery: per scenario, the
  input collage plus the sub-images uncollage crops (`QUEUED.txt` where a page is
  queued). Regenerate with `python tests/make_outputs.py`.
- `fetch_photos.py` — downloads the real-photo pool into `tests/photos/`
  (gitignored) from fixed seeds, so the pool is reproducible.
- `gen_collages_real.py` — fills the layouts with real photos (`SCENARIOS_REAL`),
  renders the old-print scenarios (`OLD_TRUTH` gives their true boxes) and
  defines the limitation cases.
- `test_uncollage_real.py` — real-photo assertions + limitation controls.
- `make_outputs_real.py` / `output_real/` — committed real-photo gallery.
- `fetch_real_collages.py` — downloads the 10 Commons montages into
  `tests/real_collages/` (gitignored) with an `ATTRIBUTIONS.md`.
- `test_real_collages.py` — the golden/characterization suite over those 10.

Synthetic tiles are textured (gradient + shapes + grain) on purpose, so they
clear the background/texture thresholds and the line-splitter's uniformity checks
— flat colour blocks would be misread as background, unlike real photos.
