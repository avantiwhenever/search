#!/usr/bin/env bash
set -euo pipefail

# Fetches the pre-exported ONNX weights + tokenizer for the embedding model
# (BAAI/bge-small-en-v1.5, via Xenova's ONNX export) into models/.
# https://huggingface.co/Xenova/bge-small-en-v1.5

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"

EMBEDDING_REPO="Xenova/bge-small-en-v1.5"
EMBEDDING_DIR="${MODELS_DIR}/bge-small-en-v1.5"
EMBEDDING_BASE="https://huggingface.co/${EMBEDDING_REPO}/resolve/main"

mkdir -p "${EMBEDDING_DIR}"

echo "Downloading ${EMBEDDING_REPO}..."
curl -fsSL "${EMBEDDING_BASE}/onnx/model.onnx" -o "${EMBEDDING_DIR}/model.onnx"
curl -fsSL "${EMBEDDING_BASE}/tokenizer.json" -o "${EMBEDDING_DIR}/tokenizer.json"
curl -fsSL "${EMBEDDING_BASE}/config.json" -o "${EMBEDDING_DIR}/config.json"

echo "Done."
du -h "${EMBEDDING_DIR}"/*
