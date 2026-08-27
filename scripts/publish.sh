#!/bin/bash
# Publish HybridDB to PyPI. Prompts securely for the API token, saves it to
# .env, and uploads the built artifacts in dist/.
#
# Usage: ./scripts/publish.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "$(ls dist/*.whl 2>/dev/null)" ]; then
    echo "dist/ has no wheel — run: uv build"
    exit 1
fi

read -r -s -p "Paste PyPI token (hidden): " TOKEN
echo

if [[ ! "$TOKEN" =~ ^pypi- ]]; then
    echo "error: token should start with 'pypi-'"
    exit 1
fi

# update .env in place
python3 - "$TOKEN" <<'PY'
import sys
tok = sys.argv[1]
lines = open(".env").read().splitlines()
out, found = [], False
for line in lines:
    if line.startswith("PYPI_TOKEN="):
        out.append("PYPI_TOKEN=" + tok)
        found = True
    else:
        out.append(line)
if not found:
    out.append("PYPI_TOKEN=" + tok)
open(".env", "w").write("\n".join(out) + "\n")
print(f".env updated (token length {len(tok)})")
PY

UV_PUBLISH_TOKEN="$TOKEN" uv publish dist/*
echo "published."