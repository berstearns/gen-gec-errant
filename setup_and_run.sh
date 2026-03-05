#!/usr/bin/env bash
# ============================================================
# Setup & Run with uv
# ============================================================
set -e

echo "=== Step 1: Create venv and install dependencies ==="
uv venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

uv pip install -e .

echo ""
echo "=== Step 2: Download spaCy model ==="
uv run python -m spacy download en_core_web_sm