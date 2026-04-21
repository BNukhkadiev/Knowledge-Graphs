#!/usr/bin/env bash
# Evaluate a two-stage checkpoint on a DLCC-style tab-separated split.
#
# Usage:
#   ./scripts/dbpedia/eval_checkpoint.sh <checkpoint.pt> <test.txt>
#
# Example:
#   ./scripts/dbpedia/eval_checkpoint.sh output/dbpedia_run/rdf2vec_final.pt \
#     v1/dbpedia/tc07/species/500/train_test/test.txt

set -euo pipefail

CKPT="${1:?checkpoint .pt path}"
TEST="${2:?test.txt path}"

uv run src/evaluate_embeddings.py "${TEST}" -c "${CKPT}"
