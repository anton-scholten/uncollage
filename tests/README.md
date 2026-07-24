# tests

A headless regression suite for `uncollage.py`. It builds small synthetic
collages in memory, each exercising one scenario the tool handles, and asserts
the sub-image counts.

## Run

```
python tests/test_uncollage.py     # prints a PASS/FAIL table, non-zero exit on failure
pytest tests/test_uncollage.py     # same checks under pytest, if installed
```

No display is needed. Only `opencv-python` and `numpy` (the tool's own deps).

## What it covers

| Scenario | Flags | Expected |
|---|---|---|
| `separated` — photos with paper gaps | *(plain)* | 4 crops |
| `rotated` — tilted photos | *(plain)* | 3 crops (upright bbox each) |
| `touching` — butted with a shadow seam | `--rectangular` | 2 crops |
| `grid` — 2×3 touching photos | `--rectangular` | 6 crops |
| `composite` — 2 photos butted, no gap | `--split-lines` | 2 crops |
| `hard` — faded borderless photo ≈ paper | *(plain)* | queued for manual fallback |

Plus negative controls: `touching` without `--rectangular` stays merged, and
`--split-lines` does **not** over-split the cleanly separated page. An
end-to-end test runs `process_image` and checks the actual files written.

## Files

- `gen_collages.py` — deterministic collage builders (`SCENARIOS`). Run it
  directly to dump the input PNGs into `tests/collages/` (gitignored).
- `test_uncollage.py` — the assertions; works standalone or under pytest.
- `make_outputs.py` — renders the committed gallery under `tests/output/`.
- `output/` — one folder per scenario holding the input collage and the
  sub-images uncollage crops from it, so the expected results are visible
  without running anything. The `hard` folder holds only the input plus a
  `QUEUED.txt` marker (that page is queued for the manual fallback, not cropped).
  Regenerate with `python tests/make_outputs.py` after any change that should
  alter the crops.

The tiles are textured (gradient + shapes + grain) on purpose, so they clear the
background/texture thresholds and the line-splitter's uniformity checks — flat
colour blocks would be misread as background, unlike real photos.
