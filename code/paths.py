"""Portable project paths -- everything is derived from the repository location."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache")
RAW = os.path.join(ROOT, "raw")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
TABLES = os.path.join(ROOT, "tables")
LOGS = os.path.join(ROOT, "logs")
IMAGES = os.path.join(DATA, "images")

for _d in (DATA, CACHE, RAW, RESULTS, FIGURES, TABLES, LOGS, IMAGES):
    os.makedirs(_d, exist_ok=True)
