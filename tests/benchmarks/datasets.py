"""BEIR dataset loader for HybridDB accuracy benchmarks.

Downloads and caches BEIR datasets (https://github.com/beir-cellar/beir) —
the standard zero-shot IR benchmark used by MTEB and embedding-model papers.

Default dataset: NFCorpus — 3,633 medical documents, 324 test queries, and
graded relevance judgments (0-3), so nDCG is meaningful.

The loader is dependency-free (urllib + zipfile) and caches under
``~/.cache/hybriddb-bench/<name>/``. If the download fails (offline), callers
should skip rather than fail.
"""

import json
import urllib.request
import zipfile
from pathlib import Path

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
CACHE_ROOT = Path.home() / ".cache" / "hybriddb-bench"

# name -> (url, expected files)
BEIR_DATASETS = {
    "nfcorpus": {
        "url": f"{BEIR_BASE}/nfcorpus.zip",
        "files": ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"),
    },
    "scifact": {
        "url": f"{BEIR_BASE}/scifact.zip",
        "files": ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"),
    },
}


def _dataset_dir(name: str) -> Path:
    return CACHE_ROOT / name


def download_beir(name: str = "nfcorpus", timeout: int = 300) -> Path:
    """Download and extract a BEIR dataset into the cache dir. Returns the dir."""
    if name not in BEIR_DATASETS:
        raise ValueError(f"Unknown BEIR dataset: {name!r} (have {sorted(BEIR_DATASETS)})")
    dest = _dataset_dir(name)
    if all((dest / f).exists() for f in BEIR_DATASETS[name]["files"]):
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    url = BEIR_DATASETS[name]["url"]
    zip_path = dest / f"{name}.zip"
    print(f"  downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310 — pinned https URL
    with zipfile.ZipFile(zip_path) as zf:
        # zip contains a top-level dir named after the dataset
        members = zf.namelist()
        prefix = next(
            (m.split("/")[0] for m in members if m.endswith("corpus.jsonl")),
            "",
        )
        for m in members:
            if m.endswith("/") or not m:
                continue
            rel = m[len(prefix) + 1:] if prefix and m.startswith(prefix + "/") else m
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as out:
                out.write(src.read())
    zip_path.unlink(missing_ok=True)
    return dest


def load_beir(name: str = "nfcorpus") -> dict:
    """Load a BEIR dataset into memory.

    Returns:
        {
            "docs": [{"_id": str, "title": str, "text": str}, ...],
            "queries": [{"_id": str, "text": str}, ...],   # test split only
            "qrels": {query_id: {corpus_id: score}},        # test split only
        }
    """
    d = download_beir(name)
    docs = []
    with open(d / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            docs.append({
                "_id": row["_id"],
                "title": row.get("title", ""),
                "text": row.get("text", ""),
            })

    all_queries = {}
    with open(d / "queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            all_queries[row["_id"]] = row["text"]

    qrels: dict[str, dict[str, int]] = {}
    with open(d / "qrels" / "test.tsv") as f:
        next(f)  # header: query-id\tcorpus-id\tscore
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[cid] = int(score)

    # test split queries = those with judgments
    queries = [{"id": qid, "text": all_queries[qid]} for qid in qrels]
    return {"docs": docs, "queries": queries, "qrels": qrels}
