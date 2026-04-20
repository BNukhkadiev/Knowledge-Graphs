#!/usr/bin/env bash
# Generate RDF2Vec-style random walks for every entity listed in
# v1/dbpedia/tc01/**/{positives,negatives}.txt, scanning the large DBpedia graph once.
#
# Default walk hyperparams match this repo's "usual" settings:
#   mode=jrdf2vec-duplicate-free, depth=4, walks_per_entity=100
#
# Output goes under: output/dbpedia_tc01_walks/<datetime>/<domain>/<size>/
#
# Usage:
#   ./scripts/dbpedia/generate_tc01_walks.sh
#   DEPTH=4 WALKS_PER_ENTITY=100 ./scripts/dbpedia/generate_tc01_walks.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TC01_ROOT="${TC01_ROOT:-${REPO_ROOT}/v1/dbpedia/tc01}"
GRAPH_NT="${GRAPH_NT:-${REPO_ROOT}/dbpedia_graph/graph.nt}"

DEPTH="${DEPTH:-4}"
WALKS_PER_ENTITY="${WALKS_PER_ENTITY:-100}"
TOKEN_FORMAT="${TOKEN_FORMAT:-angled}"
THREADS="${THREADS:-}"
SEED="${SEED:-}"

RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/output/dbpedia_tc01_walks/${RUN_DATETIME}}"
mkdir -p "${OUT_ROOT}"

if [ ! -f "${GRAPH_NT}" ]; then
  echo "Missing graph file: ${GRAPH_NT}" >&2
  exit 1
fi

if [ ! -d "${TC01_ROOT}" ]; then
  echo "Missing TC01 root folder: ${TC01_ROOT}" >&2
  exit 1
fi

echo "TC01 root:      ${TC01_ROOT}"
echo "Graph:          ${GRAPH_NT}"
echo "Output root:    ${OUT_ROOT}"
echo "Depth:          ${DEPTH}"
echo "Walks/entity:   ${WALKS_PER_ENTITY}"
echo "Token format:   ${TOKEN_FORMAT}"

ARGS=(--tc01-root "${TC01_ROOT}" --graph "${GRAPH_NT}" --out-root "${OUT_ROOT}" --depth "${DEPTH}" --walks-per-entity "${WALKS_PER_ENTITY}" --token-format "${TOKEN_FORMAT}")
if [ -n "${THREADS}" ]; then
  ARGS+=(--threads "${THREADS}")
fi
if [ -n "${SEED}" ]; then
  ARGS+=(--seed "${SEED}")
fi

uv run src/dbpedia/tc01_walks.py "${ARGS[@]}"

echo "Done. Walks written under: ${OUT_ROOT}"

