"""
Preprocessing pipeline for LLM-SEMRec (paper Section VI-B).

Implements, per dataset:
  1. iterative 5-core filtering (users AND items with >= 5 interactions)
  2. chronological sorting per user
  3. leave-two-out chronological split (test = last, val = penultimate, train = rest)
  4. item textual profile (Eq. 12) built ONLY from pre-split information
  5. truncation to L at train/inference time
  6. dataset statistics for Table II

SNAP Amazon dumps mix JSON and Python-repr line formats -- both are accepted.
"""
import ast
import gzip
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, RAW

MIN_INTER = 5
MAX_TEXT_CHARS = 600
N_REVIEWS_PER_ITEM = 3


def clean_text(s):
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = s.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s).strip()


def _loads(line):
    try:
        return json.loads(line)
    except Exception:
        pass
    try:
        return ast.literal_eval(line)
    except Exception:
        return None


def iter_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _loads(line)
            if obj is not None:
                yield obj


# ----------------------------------------------------------------------------- loaders
def load_amazon(reviews_path, meta_path):
    print(f"  reading metadata: {os.path.basename(meta_path)}")
    meta = {}
    n_meta = 0
    for m in iter_json_gz(meta_path):
        n_meta += 1
        if n_meta % 200000 == 0:
            print(f"    meta parsed: {n_meta:,}", flush=True)
        asin = m.get("asin")
        if not asin:
            continue
        cats = []
        for c in (m.get("categories") or []):
            if isinstance(c, list):
                cats.extend([clean_text(x) for x in c if isinstance(x, str)])
            elif isinstance(c, str):
                cats.append(clean_text(c))
        desc = m.get("description")
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc)
        seen, cu = set(), []
        for c in cats:
            if c and c not in seen:
                seen.add(c)
                cu.append(c)
        meta[asin] = {
            "title": clean_text(m.get("title")),
            "cats": cu[:8],
            "brand": clean_text(m.get("brand")),
            "price": str(m.get("price") or ""),
            "desc": clean_text(desc)[:MAX_TEXT_CHARS],
            "imUrl": m.get("imUrl") or "",
        }
    print(f"  metadata entries: {len(meta):,}")

    print(f"  reading reviews: {os.path.basename(reviews_path)}")
    interactions = defaultdict(list)
    with gzip.open(reviews_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            r = _loads(line)
            if r is None:
                continue
            u, a = r.get("reviewerID"), r.get("asin")
            ts = r.get("unixReviewTime")
            if not u or not a or ts is None:
                continue
            interactions[u].append(
                (int(ts), a, float(r.get("overall", 0.0)), clean_text(r.get("summary"))))
    print(f"  raw users: {len(interactions):,}  "
          f"raw interactions: {sum(len(v) for v in interactions.values()):,}")
    return interactions, meta


def load_ml1m(path_dir):
    print(f"  reading ML-1M from {path_dir}")
    ratings = []
    with open(os.path.join(path_dir, "ratings.dat"), "r", encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("::")
            if len(p) == 4:
                ratings.append((p[0], p[1], float(p[2]), int(p[3])))
    meta = {}
    with open(os.path.join(path_dir, "movies.dat"), "r", encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("::")
            if len(p) != 3:
                continue
            mid, title, genres = p
            year = ""
            m = re.search(r"\((\d{4})\)\s*$", title)
            if m:
                year = m.group(1)
                title = title[: m.start()].strip()
            meta[mid] = {"title": clean_text(title),
                         "cats": [g.strip() for g in genres.split("|")],
                         "brand": year, "price": "", "desc": "", "imUrl": ""}
    interactions = defaultdict(list)
    for u, a, r, ts in ratings:
        interactions[u].append((ts, a, r, ""))
    print(f"  raw users: {len(interactions):,}  raw interactions: {len(ratings):,}")
    return interactions, meta


# ----------------------------------------------------------------------------- 5-core
def iterative_5core(interactions, min_inter=MIN_INTER, verbose=True):
    cur = {u: list(v) for u, v in interactions.items()}
    for it in range(30):
        ic = Counter()
        for v in cur.values():
            for (_, a, _, _) in v:
                ic[a] += 1
        keep_items = {a for a, c in ic.items() if c >= min_inter}
        nxt = {}
        for u, v in cur.items():
            vv = [x for x in v if x[1] in keep_items]
            if len(vv) >= min_inter:
                nxt[u] = vv
        stable = (set(nxt.keys()) == set(cur.keys())
                  and all(len(nxt[u]) == len(cur[u]) for u in nxt))
        cur = nxt
        if verbose:
            print(f"    5-core iter {it+1}: users={len(cur):,} items={len(keep_items):,}")
        if stable:
            break
    ic = Counter()
    for v in cur.values():
        for (_, a, _, _) in v:
            ic[a] += 1
    items = sorted(a for a, c in ic.items() if c >= min_inter)
    cur = {u: [x for x in v if x[1] in items] for u, v in cur.items()}
    cur = {u: v for u, v in cur.items() if len(v) >= min_inter}
    return cur, items


# ----------------------------------------------------------------------------- build
def build_dataset(name, interactions, meta, out_name=None, min_inter=MIN_INTER):
    print(f"\n=== {name} ===")
    inter, items = iterative_5core(interactions, min_inter)
    item2idx = {a: i for i, a in enumerate(items)}
    user2idx = {u: i for i, u in enumerate(sorted(inter))}
    n_int = sum(len(v) for v in inter.values())
    print(f"  after {min_inter}-core: users={len(inter):,} items={len(items):,} "
          f"interactions={n_int:,}")

    for u in inter:
        inter[u].sort(key=lambda x: x[0])

    # item textual profile -- reviews restricted to the TRAIN portion (no leakage)
    train_rev = defaultdict(list)
    for u, v in inter.items():
        for (ts, a, r, summ) in v[:-2]:
            if summ:
                train_rev[a].append(summ)

    profiles, imurls, titles = {}, {}, {}
    n_text = 0
    for a in items:
        m = meta.get(a, {})
        title = m.get("title") or a
        cat = " > ".join(m.get("cats", []))
        attrs = []
        if m.get("brand"):
            attrs.append(f"brand={m['brand']}")
        if m.get("price"):
            attrs.append(f"price={m['price']}")
        revs = train_rev.get(a, [])[:N_REVIEWS_PER_ITEM]
        parts = [f"Title: {title}"]
        if cat:
            parts.append(f"Category: {cat}")
        if attrs:
            parts.append(f"Attributes: {', '.join(attrs)}")
        if m.get("desc"):
            parts.append(f"Description: {m['desc']}")
        if revs:
            parts.append("Review: " + " | ".join(revs[:3])[:300])
        txt = "; ".join(parts)
        profiles[a] = txt
        imurls[a] = m.get("imUrl", "")
        titles[a] = title
        if len(txt) > 20:
            n_text += 1

    n_bins = 5

    def rating_bin(r):
        return max(0, min(n_bins - 1, int(round(r)) - 1))

    seqs, train_targets, val_targets, test_targets = {}, {}, {}, {}
    for u, v in inter.items():
        arr = [(item2idx[a], rating_bin(r), int(ts)) for (ts, a, r, _) in v]
        if len(arr) < 3:
            continue
        seqs[u] = arr
        train_targets[u] = arr[:-2]
        val_targets[u] = arr[-2]
        test_targets[u] = arr[-1]

    lens = [len(v) for v in seqs.values()]
    stats = {
        "dataset": name,
        "users": len(seqs),
        "items": len(items),
        "interactions": n_int,
        "density_pct": round(100.0 * n_int / (len(seqs) * len(items)), 4),
        "text_coverage_pct": round(100.0 * n_text / len(items), 2),
        "avg_seq_len": round(sum(lens) / len(lens), 2),
        "median_seq_len": int(sorted(lens)[len(lens) // 2]),
        "min_seq_len": int(min(lens)),
        "max_seq_len": int(max(lens)),
        "n_bins": n_bins,
        "n_train_interactions": sum(len(v) for v in train_targets.values()),
    }
    print("  stats:", json.dumps(stats))

    out = {"name": name, "stats": stats, "user2idx": user2idx, "item2idx": item2idx,
           "items": items, "seqs": seqs, "train_targets": train_targets,
           "val_targets": val_targets, "test_targets": test_targets,
           "profiles": profiles, "imurls": imurls, "titles": titles, "meta": meta}
    fn = os.path.join(DATA, f"{out_name or name}.pkl")
    with open(fn, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"  -> {fn}")
    return out


def prep_movielens_zip():
    import zipfile
    z = os.path.join(RAW, "ml-1m.zip")
    d = os.path.join(RAW, "ml-1m")
    if not os.path.isdir(d):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
    return d


def update_stats_file(stats):
    """Merge (not overwrite) per-dataset statistics into data/dataset_stats.json."""
    p = os.path.join(DATA, "dataset_stats.json")
    cur = {}
    if os.path.exists(p):
        try:
            for s in json.load(open(p)):
                cur[s["dataset"]] = s
        except Exception:
            pass
    cur[stats["dataset"]] = stats
    json.dump(list(cur.values()), open(p, "w"), indent=2)
    return list(cur.values())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "beauty"):
        inter, meta = load_amazon(f"{RAW}/beauty_reviews.json.gz", f"{RAW}/beauty_meta.json.gz")
        update_stats_file(build_dataset("amazon_beauty", inter, meta)["stats"])
        del inter, meta
    if which in ("all", "fashion"):
        inter, meta = load_amazon(f"{RAW}/cloth_reviews.json.gz", f"{RAW}/cloth_meta.json.gz")
        update_stats_file(build_dataset("amazon_fashion", inter, meta)["stats"])
        del inter, meta
    if which in ("all", "ml1m"):
        inter, meta = load_ml1m(prep_movielens_zip())
        update_stats_file(build_dataset("movielens_1m", inter, meta)["stats"])
        del inter, meta
    print("\n=== TABLE II DATA ===")
    print(json.dumps(json.load(open(os.path.join(DATA, "dataset_stats.json"))), indent=2))
