#!/usr/bin/env python3
"""
Download 10 real photo collages (city "montages") from Wikimedia Commons into
tests/real_collages/, and write an ATTRIBUTIONS.md next to them.

These are genuine collages -- grids of rectangular photos butted together with
thin seams -- so they are a realistic, hard test set for uncollage.py (the
dark-photo ones exercise the shadow-seam limitation). All are freely licensed (CC BY-SA / CC0); the images themselves are
gitignored, so run this once before test_real_collages.py:

    python tests/fetch_real_collages.py
"""
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "real_collages")
UA = "uncollage-test-fetch/1.0 (https://github.com/; anton.h.scholten@gmail.com)"
WIDTH = 1000

# slug, Commons File name, license, author/credit
COLLAGES = [
    ("01_bishkek", "Bishkek Montage 2020.png", "CC BY-SA 4.0",
     "Абдырашит Сатылганов, Intergelpo, Vmenkov, Ondřej Žváček et al."),
    ("02_melbourne", "Melbourne City Montage 2019.jpg", "CC BY-SA 4.0", "Ashton 29"),
    ("03_nyc", "NYC Montage 2014 4 - Jleon.jpg", "CC BY-SA 3.0", "Jleon"),
    ("04_quebec", "Quebec City Montage 2016.jpg", "CC BY-SA 4.0", "Judicieux"),
    ("05_saltlake", "Salt Lake City montage 19 July 2011.jpg", "CC BY-SA 3.0",
     "Various (compiled), see file page"),
    ("06_warsaw", "Warsaw montage.JPG", "CC BY-SA 4.0",
     "Foxy1219 (montage), Radek Kołakowski et al."),
    ("07_johannesburg", "Montage Johannesburg.jpg", "CC BY-SA 3.0", "Mr.Jhosimar"),
    ("08_doha", "City montage of Doha2.png", "CC BY-SA 3.0", "Diego Delso"),
    ("09_fukuoka", "Fukuoka City - Montage - 01.JPG", "CC BY-SA 3.0",
     "Composition by JKT-c; photos various"),
    ("10_vyborg", "Vyborg montage 2017.jpg", "CC BY-SA 3.0",
     "A.Savin, Ghirlandajo, Maleevsw, Helga-Tigrokisa"),
]


def _url(name):
    enc = urllib.parse.quote("File:" + name)
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{enc}?width={WIDTH}"


def main():
    os.makedirs(OUT, exist_ok=True)
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", UA)]
    urllib.request.install_opener(opener)

    for slug, name, lic, author in COLLAGES:
        dst = os.path.join(OUT, f"{slug}.jpg")
        if os.path.exists(dst):
            print("have", dst)
            continue
        print("fetch", name)
        urllib.request.urlretrieve(_url(name), dst)

    with open(os.path.join(OUT, "ATTRIBUTIONS.md"), "w", encoding="utf-8") as f:
        f.write("# Real collage test set -- sources & licenses\n\n")
        f.write("All from Wikimedia Commons, freely licensed. Attribution below; "
                "see each file page at https://commons.wikimedia.org/wiki/File:<name>.\n\n")
        f.write("| File | Commons file | License | Credit |\n|---|---|---|---|\n")
        for slug, name, lic, author in COLLAGES:
            f.write(f"| {slug}.jpg | {name} | {lic} | {author} |\n")
    print(f"\npool ready in {OUT}  (+ ATTRIBUTIONS.md)")


if __name__ == "__main__":
    main()
