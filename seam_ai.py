#!/usr/bin/env python3
"""
seam_ai.py -- a small, offline, self-contained neural net that detects the
seams (boundaries) between sub-images in a collage / scanned page.

It is deliberately simple: a one-hidden-layer perceptron (implemented in plain
NumPy -- no PyTorch / TensorFlow / scikit-learn, nothing to download) trained on
*synthetic* collages assembled from random crops of whatever example images you
point it at. For each pixel it predicts "is this on a horizontal boundary
between two photos", using a handful of cheap, vectorised local features
(brightness, background-likeness, and how the pixel compares with its neighbours
a few pixels above/below). Vertical seams are handled by running the same model
on the transposed image, so one tiny model covers both directions.

The trained weights live in `seam_model.npz` next to this file. uncollage.py
loads them when run with `--ai` and feeds the predicted seam maps into its
rectangular-mode guillotine splitter.

CLI:
    python seam_ai.py train [img1 img2 ...]   # (re)train; defaults to *.jpg here
    python seam_ai.py show  IMG                # write IMG.seams.png heat-map

Requires: opencv-python (cv2), numpy.
"""

import glob
import os
import sys

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seam_model.npz")

WORK_MIN = 640          # features/inference run at this working scale (px, short side)
OFFSETS = (2, 4, 8, 16, 28)   # vertical neighbour distances used as features
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Features                                                                     #
# --------------------------------------------------------------------------- #
def _bg_color(img):
    bins, step = 16, 16
    q = (img // step).reshape(-1, 3)
    codes = (q[:, 0].astype(np.int32) * bins + q[:, 1]) * bins + q[:, 2]
    peak = np.bincount(codes, minlength=bins ** 3).argmax()
    return img.reshape(-1, 3)[codes == peak].astype(np.float32).mean(axis=0)


def feature_maps(img):
    """Return an (H, W, F) float32 stack of features oriented for HORIZONTAL
    seams (all neighbour comparisons are along the vertical axis)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    bg = _bg_color(img)
    diff = img.astype(np.float32) - bg[None, None, :]
    bgd = np.sqrt((diff * diff).sum(axis=2)) / 255.0   # background distance

    feats = [gray, bgd]
    for d in OFFSETS:
        up = np.roll(gray, -d, axis=0)
        dn = np.roll(gray, d, axis=0)
        feats.append(gray - up)          # darker/brighter than pixel above
        feats.append(gray - dn)          # ... than pixel below
        feats.append(np.abs(up - dn))    # contrast across the pixel
    # local vertical texture: std above vs below (helps tell a boundary from
    # flat content -- one side is often border/background).
    k = 9
    m = cv2.boxFilter(gray, -1, (1, k))
    s = np.sqrt(np.maximum(cv2.boxFilter(gray * gray, -1, (1, k)) - m * m, 0))
    feats.append(np.roll(s, -k, axis=0))
    feats.append(np.roll(s, k, axis=0))
    return np.stack(feats, axis=-1).astype(np.float32)


def _work_image(img):
    h, w = img.shape[:2]
    scale = WORK_MIN / min(h, w) if min(h, w) > WORK_MIN else 1.0
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
    return img


# --------------------------------------------------------------------------- #
# Tiny MLP (NumPy)                                                             #
# --------------------------------------------------------------------------- #
class SeamMLP:
    def __init__(self, w1=None, b1=None, w2=None, b2=None, mu=None, sd=None):
        self.w1, self.b1, self.w2, self.b2, self.mu, self.sd = w1, b1, w2, b2, mu, sd

    @property
    def ready(self):
        return self.w1 is not None

    def _forward(self, x):
        z = (x - self.mu) / self.sd
        h = np.maximum(z @ self.w1 + self.b1, 0.0)
        return 1.0 / (1.0 + np.exp(-(h @ self.w2 + self.b2)))

    def predict_map(self, feat):
        h, w, f = feat.shape
        p = self._forward(feat.reshape(-1, f).astype(np.float32))
        return p.reshape(h, w)

    def save(self, path):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path=MODEL_PATH):
        if not os.path.exists(path):
            return cls()
        d = np.load(path)
        return cls(d["w1"], d["b1"], d["w2"], d["b2"], d["mu"], d["sd"])


def predict_seam_maps(img, model=None):
    """Return (prob_h, prob_w) probability maps in [0, 1] at the image's size.

    prob_h is high on horizontal seams, prob_w high on vertical seams.
    """
    model = model or SeamMLP.load()
    if not model.ready:
        raise RuntimeError("no trained model; run `python seam_ai.py train` first")
    h, w = img.shape[:2]
    small = _work_image(img)
    ph = model.predict_map(feature_maps(small))
    pw = model.predict_map(feature_maps(cv2.transpose(small))).T
    ph = cv2.resize(ph, (w, h))
    pw = cv2.resize(pw, (w, h))
    return ph, pw


# --------------------------------------------------------------------------- #
# Synthetic training data                                                      #
# --------------------------------------------------------------------------- #
def _synth(crops):
    """Build a random collage; return (img, seam_h, seam_v) where the seam masks
    mark horizontal / vertical boundaries between photos (and photo vs paper)."""
    bg = np.array([243, 248, 244]) + RNG.integers(-6, 7, 3)
    H = int(RNG.integers(500, 820)); W = int(RNG.integers(500, 820))
    img = np.clip(np.full((H, W, 3), bg) + RNG.normal(0, 2, (H, W, 3)), 0, 255).astype(np.uint8)
    seam_h = np.zeros((H, W), np.uint8)
    seam_v = np.zeros((H, W), np.uint8)

    rows = int(RNG.integers(1, 4))
    cols = int(RNG.integers(1, 4))
    ys = np.linspace(int(RNG.integers(4, 40)), H - int(RNG.integers(4, 40)), rows + 1).astype(int)
    xs = np.linspace(int(RNG.integers(4, 40)), W - int(RNG.integers(4, 40)), cols + 1).astype(int)

    for r in range(rows):
        for c in range(cols):
            gap = int(RNG.integers(0, 16))           # 0 => photos touch
            y0, y1 = ys[r] + gap, ys[r + 1] - gap
            x0, x1 = xs[c] + gap, xs[c + 1] - gap
            if y1 - y0 < 40 or x1 - x0 < 40:
                continue
            crop = crops[RNG.integers(len(crops))]
            ph = cv2.resize(crop, (x1 - x0, y1 - y0))
            if RNG.random() < 0.6:                    # light photo border
                b = int(RNG.integers(3, 18))
                ph[:b] = ph[-b:] = 240; ph[:, :b] = ph[:, -b:] = 240
            img[y0:y1, x0:x1] = ph
            if RNG.random() < 0.35 and gap < 4:       # overlap shadow on top/left
                img[y0:y0 + 3, x0:x1] = (img[y0:y0 + 3, x0:x1] * 0.45).astype(np.uint8)
            # mark boundaries (a few px thick)
            seam_h[max(0, y0 - 2):y0 + 3, x0:x1] = 1
            seam_h[max(0, y1 - 2):y1 + 3, x0:x1] = 1
            seam_v[y0:y1, max(0, x0 - 2):x0 + 3] = 1
            seam_v[y0:y1, max(0, x1 - 2):x1 + 3] = 1
    return img, seam_h, seam_v


def _load_crops(paths):
    crops = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        for _ in range(6):
            cw, ch = int(RNG.integers(w // 4, w)), int(RNG.integers(h // 4, h))
            x, y = int(RNG.integers(0, w - cw + 1)), int(RNG.integers(0, h - ch + 1))
            crops.append(im[y:y + ch, x:x + cw].copy())
    return crops or [np.full((200, 200, 3), 128, np.uint8)]


def _adam_train(X, y, hidden=24, epochs=160, lr=6e-3, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    n, f = Xs.shape
    w1 = rng.normal(0, np.sqrt(2 / f), (f, hidden)); b1 = np.zeros(hidden)
    w2 = rng.normal(0, np.sqrt(2 / hidden), (hidden, 1)); b2 = np.zeros(1)
    params = [w1, b1, w2, b2]
    ms = [np.zeros_like(p) for p in params]; vs = [np.zeros_like(p) for p in params]
    yb = y.reshape(-1, 1).astype(np.float32)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, 4096):
            b = idx[s:s + 4096]
            xb = Xs[b]
            h_pre = xb @ w1 + b1
            h = np.maximum(h_pre, 0)
            logit = h @ w2 + b2
            p = 1 / (1 + np.exp(-logit))
            g = (p - yb[b]) / len(b)
            gw2 = h.T @ g; gb2 = g.sum(0)
            gh = (g @ w2.T) * (h_pre > 0)
            gw1 = xb.T @ gh; gb1 = gh.sum(0)
            grads = [gw1, gb1, gw2, gb2]
            for i, (pm, gr) in enumerate(zip(params, grads)):
                ms[i] = 0.9 * ms[i] + 0.1 * gr
                vs[i] = 0.999 * vs[i] + 0.001 * (gr * gr)
                pm -= lr * ms[i] / (np.sqrt(vs[i]) + 1e-8)
    return SeamMLP(w1, b1, w2, b2, mu, sd)


def train(paths, n_synth=45):
    crops = _load_crops(paths)
    Xs, Ys = [], []
    for _ in range(n_synth):
        img, sh, sv = _synth(crops)
        # horizontal-seam samples from feats(img); vertical-seam samples from
        # feats(transpose) so both share one model.
        for feat, lab in ((feature_maps(img), sh),
                          (feature_maps(cv2.transpose(img)), cv2.transpose(sv))):
            pos = np.argwhere(lab > 0)
            neg = np.argwhere(lab == 0)
            if len(pos) == 0:
                continue
            k = min(len(pos), 1500)
            pi = pos[RNG.integers(0, len(pos), k)]
            ni = neg[RNG.integers(0, len(neg), k)]
            for arr in (pi, ni):
                Xs.append(feat[arr[:, 0], arr[:, 1]])
            Ys.append(np.ones(k)); Ys.append(np.zeros(k))
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    print(f"training on {len(X)} samples, {X.shape[1]} features ...")
    model = _adam_train(X, Y)
    model.save(MODEL_PATH)
    pred = (model._forward(X).ravel() > 0.5).astype(int)
    print(f"train accuracy {(pred == Y).mean():.3f}; saved {MODEL_PATH}")


def main(argv):
    if len(argv) >= 2 and argv[1] == "train":
        paths = argv[2:] or sorted(glob.glob(os.path.join(os.path.dirname(MODEL_PATH), "*.jpg")))
        train(paths)
        return 0
    if len(argv) == 3 and argv[1] == "show":
        img = cv2.imread(argv[2])
        ph, pw = predict_seam_maps(img)
        heat = np.zeros_like(img)
        heat[..., 2] = (ph * 255).astype(np.uint8)   # red   = horizontal seams
        heat[..., 1] = (pw * 255).astype(np.uint8)   # green = vertical seams
        out = cv2.addWeighted(img, 0.5, heat, 0.9, 0)
        p = argv[2] + ".seams.png"
        cv2.imwrite(p, out); print("wrote", p)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
