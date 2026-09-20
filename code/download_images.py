"""Download item images and normalise them to 224x224 RGB (paper Sec. VI-B).

Amazon 2014 `imUrl` links are frequently dead after a decade, so a legacy
fallback (https://images.amazon.com/images/P/{ASIN}...) is attempted before an
item is declared image-less.  Every payload is validated with PIL.  Items that
remain without an image are kept with a missing-modality indicator, which is
exactly the mechanism the model is designed to handle.  Coverage is written
back into data/dataset_stats.json (Table II).
"""
import io
import json
import os
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, IMAGES

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "image/avif,image/webp,image/jpeg,image/png,*/*",
}
LOCK = threading.Lock()
COUNTER = {"ok": 0, "fallback": 0, "fail": 0}


def fetch(url, timeout=8):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    if r.status_code != 200 or len(r.content) < 900:
        return None
    try:
        im = Image.open(io.BytesIO(r.content))
        im.load()
        return im.convert("RGB").resize((224, 224), Image.BICUBIC)
    except Exception:
        return None


def one(args):
    idx, asin, imurl, outdir = args
    dst = os.path.join(outdir, f"{idx}.jpg")
    if os.path.exists(dst):
        with LOCK:
            COUNTER["ok"] += 1
        return True
    im, src = None, "meta"
    if imurl and imurl.startswith("http"):
        try:
            im = fetch(imurl)
        except Exception:
            im = None
    if im is None:
        for tag in ("_SCLZZZZZZZ_", "_SCKZZZZZZZ_", "_SX300_", "_SY300_"):
            try:
                im = fetch(f"https://images.amazon.com/images/P/{asin}.01.{tag}.jpg")
            except Exception:
                im = None
            if im is not None:
                src = "legacy"
                break
    if im is None:
        with LOCK:
            COUNTER["fail"] += 1
        return False
    im.save(dst, "JPEG", quality=92)
    with LOCK:
        COUNTER["ok"] += 1
        if src == "legacy":
            COUNTER["fallback"] += 1
    return True


def run(ds_name, workers=24):
    with open(os.path.join(DATA, f"{ds_name}.pkl"), "rb") as f:
        ds = pickle.load(f)
    items, imurls = ds["items"], ds["imurls"]
    outdir = os.path.join(IMAGES, ds_name)
    os.makedirs(outdir, exist_ok=True)
    jobs = [(i, a, imurls.get(a, ""), outdir) for i, a in enumerate(items)]
    print(f"[{ds_name}] downloading {len(jobs):,} images -> {outdir}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, jobs))
    n_items = len(items)
    have = len([1 for i in range(n_items)
                if os.path.exists(os.path.join(outdir, f"{i}.jpg"))])
    cov = 100.0 * have / n_items
    print(f"[{ds_name}] ok={COUNTER['ok']:,} (legacy fallback {COUNTER['fallback']:,}) "
          f"fail={COUNTER['fail']:,} -> coverage {have:,}/{n_items:,} = {cov:.2f}%")

    sp = os.path.join(DATA, "dataset_stats.json")
    stats = json.load(open(sp)) if os.path.exists(sp) else []
    for s in stats:
        if s["dataset"] == ds_name:
            s["image_coverage_pct"] = round(cov, 2)
            s["images_available"] = have
    json.dump(stats, open(sp, "w"), indent=2)
    json.dump({"coverage_pct": round(cov, 2), "have": have, "n_items": n_items,
               "legacy_fallback": COUNTER["fallback"]},
              open(os.path.join(DATA, f"{ds_name}_imgstats.json"), "w"), indent=2)
    return cov


if __name__ == "__main__":
    run(sys.argv[1], workers=int(sys.argv[2]) if len(sys.argv) > 2 else 24)
