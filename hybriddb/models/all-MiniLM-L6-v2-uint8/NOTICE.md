# Bundled model: all-MiniLM-L6-v2 (uint8, per-channel)

This directory ships the default embedding engine for HybridDB, so the
"embedded, local, no internet required" promise holds without a first-run
download and without the lossy hash fallback (see
`docs/notes/2026-09-28-chromadb-rust-bindings-segfault.md`).

| file | size | origin |
|---|---|---|
| `onnx/model.onnx` | 22.9 MB | per-channel uint8 dynamic quantization of the fp32 model |
| `onnx/tokenizer.json` | 695 KB | copy of the tokenizer in Chroma's bundle |
| `onnx/tokenizer_config.json`, `special_tokens_map.json`, `config.json`, `vocab.txt` | <1 KB | needed by `ONNXMiniLM_L6_V2` layout checks and by this loader's provenance |

## Provenance

- fp32 source: `https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz`
- fp32 archive SHA-256: `913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3`
- fp32 `model.onnx` SHA-256:
  `4f148ba8ae9c2c7fbee4af2b132db8d06c6a6545b47fc83bbb98c3d22b8393e6`
- bundle (this directory) `model.onnx` SHA-256:
  `beafa0f3fce2421733d982dcdbc6f97f6976bce8526717646ca2e129cac23d2b`
- quantization: `onnxruntime.quantization.quantize_dynamic(..., weight_type=QUInt8, per_channel=True, reduce_range=True)`

## Licence and attribution

- `all-MiniLM-L6-v2` — © UKPLab, `sentence-transformers`, Apache-2.0
- ONNX conversion — © chroma-core/`onnx-embedding`, Apache-2.0
- `onnxruntime` / `tokenizers` — MIT, already required by `chromadb`

Copies of the licences are not included here; see the upstream repositories.
Redistribution under Apache-2.0 requires keeping the notices above.