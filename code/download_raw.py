"""
Re-download the raw dataset archives that code/prep.py consumes.

prep.py itself does NOT download anything -- it only reads from raw/. This
script restores raw/, so the preprocessed pickles in data/ can be rebuilt
from scratch at any time.

    python code/download_raw.py            # skip what is already present
    python code/download_raw.py --force    # re-download everything

Expected total download: ~460 MB. All five sources were verified to return
HTTP 200 on 2026-09-19.
"""
import os
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import RAW  # noqa: E402

SNAP = ("https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/")

# source filename -> (local filename, expected size in bytes)
SOURCES = [
    (SNAP + "reviews_Beauty_5.json.gz",
     "beauty_reviews.json.gz", 44_800_000),
    (SNAP + "meta_Beauty.json.gz",
     "beauty_meta.json.gz", 99_100_000),
    (SNAP + "reviews_Clothing_Shoes_and_Jewelry_5.json.gz",
     "cloth_reviews.json.gz", 48_200_000),
    (SNAP + "meta_Clothing_Shoes_and_Jewelry.json.gz",
     "cloth_meta.json.gz", 280_000_000),
]
ML1M = ("https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ml-1m.zip", 5_900_000)


def fetch(url, dest, expected):
    """Stream a URL to dest, with progress. Skips a plausible existing file."""
    if os.path.exists(dest) and abs(os.path.getsize(dest) - expected) < expected * 0.15:
        print("  = %-28s deja present (%.0f MB)"
              % (os.path.basename(dest), os.path.getsize(dest) / 1e6))
        return True
    print("  > %-28s telechargement..." % os.path.basename(dest), end="", flush=True)
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "hermes-agent"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got, last = 0, 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total and time.time() - last > 5:
                    print(" %d%%" % (100 * got / total), end="", flush=True)
                    last = time.time()
    except Exception as e:
        print(" ECHEC (%s)" % type(e).__name__)
        return False
    size, dt = os.path.getsize(dest), time.time() - t0
    ok = size > 1e6
    print(" %.0f MB en %.0fs%s" % (size / 1e6, dt, "" if ok else "  <-- SUSPECT"))
    return ok


def main():
    force = "--force" in sys.argv
    if force:
        for name in [s[1] for s in SOURCES] + [ML1M[1]]:
            p = os.path.join(RAW, name)
            if os.path.exists(p):
                os.remove(p)
    os.makedirs(RAW, exist_ok=True)

    print("== sources SNAP (Amazon Reviews 2014) ==")
    ok = [fetch(u, os.path.join(RAW, n), s) for u, n, s in SOURCES]

    print("== MovieLens-1M (GroupLens) ==")
    ok.append(fetch(ML1M[0], os.path.join(RAW, ML1M[1]), ML1M[2]))
    zp = os.path.join(RAW, ML1M[1])
    if os.path.exists(zp):
        print("  extraction de ml-1m.zip ...")
        with zipfile.ZipFile(zp) as z:
            z.extractall(RAW)
        print("  -> %s" % os.path.join(RAW, "ml-1m"))

    print()
    if all(ok):
        print("OK -- raw/ est complet. Vous pouvez relancer :")
        print("  python code/prep.py beauty")
        print("  python code/prep.py fashion")
        print("  python code/prep.py movielens")
        return 0
    print("CERTAINS TELECHARGEMENTS ONT ECHOUE -- relancez le script.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
