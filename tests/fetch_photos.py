#!/usr/bin/env python3
"""
Download the real-photo pool used by gen_collages_real.py into tests/photos/.

The photos are typical collage subjects (landscape / nature / city / people),
fetched from Lorem Picsum (https://picsum.photos, Unsplash-sourced) by a fixed
list of seeds, so the pool is the same every run. The image files are NOT
committed (they are gitignored); run this once before the real-photo suite:

    python tests/fetch_photos.py
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(HERE, "photos")

# Fixed seeds -> a reproducible, varied set of real photographs.
SEEDS = ["valley", "meadow", "tiger", "canyon", "harbor", "skyline",
         "elephant", "lakeside", "forest", "desert", "bridge", "coastline"]
SIZE = (1000, 800)   # (w, h)


def main():
    os.makedirs(PHOTO_DIR, exist_ok=True)
    for i, seed in enumerate(SEEDS, 1):
        dst = os.path.join(PHOTO_DIR, f"{i:02d}_{seed}.jpg")
        if os.path.exists(dst):
            print("have", dst)
            continue
        url = f"https://picsum.photos/seed/{seed}/{SIZE[0]}/{SIZE[1]}"
        print("fetch", url)
        urllib.request.urlretrieve(url, dst)
    print(f"\npool ready in {PHOTO_DIR}")


if __name__ == "__main__":
    main()
