"""Validate a dynamic-padded MiniLM loader against Chroma's global-256-padding
embedder: identical vectors, much faster per-row (HybridDB embeds row by row).

Also compares ONNX providers — Chroma defaults to *all* available providers,
which on macOS may pick CoreML and be slow per call.
"""

import os
import sys
import time

import numpy as np
from tokenizers import Tokenizer

import onnxruntime as ort


FP32 = "/tmp/onnx_inspect/onnx"


class MiniLM:
    """Vendored equivalent of chroma's ONNXMiniLM_L6_V2 forward pass, with
    dynamic padding instead of a global 256-token pad. Mean pooling uses the
    attention mask, so padded positions contribute nothing and the resulting
    vectors should be identical."""

    def __init__(self, model_dir: str, providers: list[str] | None = None):
        tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tok.enable_truncation(max_length=256)
        tok.enable_padding(pad_id=0, pad_token="[PAD]")   # pad to batch max, not 256
        self._tokenizer = tok
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._providers = providers or ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=self._providers,
            sess_options=opts,
        )

    def embedding(self, text: str) -> np.ndarray:
        enc = self._tokenizer.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        out = self._sess.run(None, {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids),
        })[0]
        expanded = np.broadcast_to(np.expand_dims(mask, -1), out.shape)
        pooled = (out * expanded).sum(1) / np.clip(expanded.sum(1), 1e-9, None)
        v = pooled[0].astype(np.float32)
        n = np.linalg.norm(v)
        return v / (n if n > 0 else 1e-12)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    sys.path.insert(0, "/Users/eddy/Developer/Python/HybridDB")
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    class CHROMA(ONNXMiniLM_L6_V2):
        DOWNLOAD_PATH = "/tmp/onnx_inspect"

    docs = [
        "hello world memory entry",
        "quarterly roadmap for the platform team",
        "The transformer architecture revolutionised NLP in 2017.",
        "word " * 600,
    ]

    chroma_ef = CHROMA()
    t0 = time.perf_counter()
    chroma_vecs = chroma_ef(docs)
    chroma_t = time.perf_counter() - t0

    cpu = MiniLM(FP32, providers=["CPUExecutionProvider"])
    avail = ort.get_available_providers()
    print("available ONNX providers:", avail)
    print(f"chroma (default providers) 4 docs : {chroma_t:.3f}s")
    t0 = time.perf_counter()
    cpu_vecs = [cpu.embedding(d) for d in docs]
    cpu_t = time.perf_counter() - t0
    print(f"vendored CPU, dynamic pad, 4 docs : {cpu_t:.3f}s  ({chroma_t / cpu_t:.1f}x faster)")

    sims = [cos(chroma_vecs[i], cpu_vecs[i]) for i in range(len(docs))]
    diffs = [abs(float(x) - float(y))
             for a, b in zip(chroma_vecs, cpu_vecs) for x, y in zip(a, b)]
    print("provider=CPU vs chroma default  :", "cos mean", round(sum(sims) / len(sims), 6),
          "min", round(min(sims), 6), "max abs diff", round(max(diffs), 6))

    if any("CoreML" in p for p in avail):
        coreml = MiniLM(FP32, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        t0 = time.perf_counter()
        cml_vecs = [coreml.embedding(d) for d in docs]
        cml_t = time.perf_counter() - t0
        s2 = [cos(chroma_vecs[i], cml_vecs[i]) for i in range(len(docs))]
        print(f"vendored CoreML, 4 docs          : {cml_t:.3f}s",
              "cos mean", round(sum(s2) / len(s2), 6), "min", round(min(s2), 6))