"""BEIR accuracy: bundled uint8 MiniLM vs fp32 Chroma runtime model.

Uses the vendored dynamic-padding loader (scripts/speedup_probe.MiniLM), which
was verified to produce vectors IDENTICAL to Chroma's ONNXMiniLM_L6_V2
(cos 1.0, max abs diff 0.0) but ~6x faster per row because it does not pad
every document to 256 tokens. Identical vectors means the comparison measures
the quantization delta only, not a padding difference.

Reuses tests/benchmarks/test_accuracy._evaluate so numbers are directly
comparable with the published BEIR table.
"""

import tempfile
from pathlib import Path

from tests.benchmarks.test_accuracy import (
    _build_db,
    _evaluate,
    _fmt,
)
from tests.benchmarks.datasets import load_beir

from scripts.speedup_probe import MiniLM


def evaluate(label: str, emb) -> None:
    for dataset in ("nfcorpus", "scifact"):
        data = load_beir(dataset)
        with tempfile.TemporaryDirectory() as tmp:
            db = _build_db(Path(tmp), emb, data)
            for mode in ("semantic", "hybrid"):
                m = _evaluate(db, data["queries"], data["qrels"], mode)
                print(f"{label:6s} {dataset:8s} {mode:8s}: {_fmt(m)}", flush=True)
            db.close()


def main() -> None:
    evaluate("fp32", MiniLM("/tmp/onnx_inspect/onnx").embedding)
    evaluate("uint8", MiniLM("/tmp/onnx_u8/onnx").embedding)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()