#!/usr/bin/env bash
# End-to-end RDF2Vec: random walks → Word2Vec (Gensim) → node-classification eval.
# All artifacts go under output/${OUTPUT_DIR}/<datetime>/.

set -euo pipefail

# Experiment folder under output/ — edit the default, or run: OUTPUT_DIR=my_run ./scripts/e2e.sh
# Each run uses a new YYYYMMDD_HHMMSS subfolder so repeated runs do not overwrite previous outputs.
OUTPUT_DIR="${OUTPUT_DIR:-tc02_rdf2vec}"
RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUT="${REPO_ROOT}/output/${OUTPUT_DIR}/${RUN_DATETIME}"
mkdir -p "${OUT}"

# Dataset paths (relative to repo root)
GRAPH_NT="${REPO_ROOT}/v1/synthetic_ontology/tc02/synthetic_ontology/graph.nt"
TEST_TXT="${REPO_ROOT}/v1/synthetic_ontology/tc02/synthetic_ontology/1000/train_test/test.txt"

WALKS_FILE="${OUT}/walks.txt"
CHECKPOINT="${OUT}/rdf2vec_word2vec.pt"
RUN_LOG="${OUT}/run.log"

# Capture full stdout and stderr of the rest of the script into RUN_LOG (and still print to terminal).
exec > >(tee "${RUN_LOG}") 2>&1

echo "Output directory: ${OUT}"
echo "Full log (stdout+stderr): ${RUN_LOG}"

echo "== 1/3 Random walks → ${WALKS_FILE}"
uv run src/walk/random_walks.py "${GRAPH_NT}" "${WALKS_FILE}" \
  --mode "jrdf2vec-duplicate-free" \
  --depth 4 \
  --walks-per-entity 100

echo "== 2/3 Train Word2Vec (RDF2Vec) → ${CHECKPOINT}"
uv run src/train/train_word2vec.py "${WALKS_FILE}" \
  --architecture skipgram \
  --dim 100 \
  --window 5 \
  --epochs 5 \
  --loss-every-steps 100 \
  -o "${CHECKPOINT}"

echo "== 3/3 Evaluate"
uv run src/evaluate_embeddings.py "${TEST_TXT}" -c "${CHECKPOINT}"

echo "Done. Artifacts under: ${OUT}"
echo "  full log:       ${RUN_LOG}"
echo "  walks:          ${WALKS_FILE}"
echo "  checkpoint:     ${CHECKPOINT}"
echo "  loss CSV/PNG:   ${OUT}/rdf2vec_word2vec_loss.csv, ${OUT}/rdf2vec_word2vec_loss.png"
echo "  step loss CSV/PNG: ${OUT}/rdf2vec_word2vec_loss_steps.csv, ${OUT}/rdf2vec_word2vec_loss_steps.png"
