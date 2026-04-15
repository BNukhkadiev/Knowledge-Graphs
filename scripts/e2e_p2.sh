#!/usr/bin/env bash
# End-to-end RDF2Vec with MASCHInE protograph P2 pretraining:
#   ontology → P1+P2 NT → walks on P2 → instance walks → Word2Vec (pretrain + finetune) → eval.
# All artifacts go under output/${OUTPUT_DIR}/<datetime>/.
#
# Mirrors scripts/e2e_p1.sh (same walk/train hyperparameters; TC via $TC) but pretrains on P2
# (--mode p2) instead of P1.

set -euo pipefail

# Test case id (folder under v1/synthetic_ontology/<TC>/...). Override: TC=tc11 ./scripts/e2e_p2.sh
TC="${TC:-tc12}"
OUTPUT_DIR="${OUTPUT_DIR:-${TC}_rdf2vec_p2}"
RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUT="${REPO_ROOT}/output/${OUTPUT_DIR}/${RUN_DATETIME}"
mkdir -p "${OUT}"

# Dataset paths (same layout as e2e_no_pretrain.sh / e2e_p1.sh)
BASE_ONTO="${REPO_ROOT}/v1/synthetic_ontology/${TC}/synthetic_ontology"
GRAPH_NT="${BASE_ONTO}/graph.nt"
ONTOLOGY_NT="${BASE_ONTO}/ontology.nt"
TEST_TXT="${BASE_ONTO}/1000/train_test/test.txt"

PROT_P1_NT="${OUT}/protograph_p1.nt"
PROT_P2_NT="${OUT}/protograph_p2.nt"
WALKS_P2="${OUT}/walks_p2.txt"
WALKS_INSTANCE="${OUT}/walks_instance.txt"
CHECKPOINT="${OUT}/rdf2vec_final.pt"
RUN_LOG="${OUT}/run.log"

# Optional overrides (defaults align with e2e_no_pretrain.sh: depth 4, 100 walks/entity, dim 100, window 5)
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
LOSS_EVERY_STEPS_PRETRAIN="${LOSS_EVERY_STEPS_PRETRAIN:-1}"
LOSS_EVERY_STEPS_FINETUNE="${LOSS_EVERY_STEPS_FINETUNE:-100}"
if [ -n "${LOSS_EVERY_STEPS:-}" ]; then
  LOSS_STEP_ARGS=(--loss-every-steps "${LOSS_EVERY_STEPS}")
else
  LOSS_STEP_ARGS=(
    --loss-every-steps-pretrain "${LOSS_EVERY_STEPS_PRETRAIN}"
    --loss-every-steps-finetune "${LOSS_EVERY_STEPS_FINETUNE}"
  )
fi

exec > >(tee "${RUN_LOG}") 2>&1

echo "Output directory: ${OUT}"
echo "Full log (stdout+stderr): ${RUN_LOG}"

echo "== 1/5 Build protograph P1 + P2 → ${PROT_P1_NT}, ${PROT_P2_NT}"
uv run protograph \
  --schema "${ONTOLOGY_NT}" \
  --kg "${GRAPH_NT}" \
  --out-dir "${OUT}"

echo "== 2/5 Random walks on P2 → ${WALKS_P2}"
uv run src/walk/random_walks.py "${PROT_P2_NT}" "${WALKS_P2}" \
  --mode "jrdf2vec-duplicate-free" \
  --depth 4 \
  --walks-per-entity 100

echo "== 3/5 Random walks on instance KG → ${WALKS_INSTANCE}"
uv run src/walk/random_walks.py "${GRAPH_NT}" "${WALKS_INSTANCE}" \
  --mode "jrdf2vec-duplicate-free" \
  --depth 4 \
  --walks-per-entity 100

echo "== 4/5 Train Word2Vec (P2 pretrain + instance finetune) → ${CHECKPOINT}"
uv run src/train/train_word2vec.py \
  --mode p2 \
  --pretrain-walks "${WALKS_P2}" \
  --instance-walks "${WALKS_INSTANCE}" \
  --ontology "${ONTOLOGY_NT}" \
  --out-dir "${OUT}" \
  --architecture skipgram \
  --dim 100 \
  --window 5 \
  --pretrain-epochs "${PRETRAIN_EPOCHS}" \
  --finetune-epochs "${FINETUNE_EPOCHS}" \
  "${LOSS_STEP_ARGS[@]}" \
  -o "${CHECKPOINT}"

echo "== 5/5 Evaluate"
uv run src/evaluate_embeddings.py "${TEST_TXT}" -c "${CHECKPOINT}"

echo "Done. Artifacts under: ${OUT}"
echo "  full log:            ${RUN_LOG}"
echo "  protograph P1:       ${PROT_P1_NT}"
echo "  protograph P2:       ${PROT_P2_NT}"
echo "  walks (P2):          ${WALKS_P2}"
echo "  walks (instance):    ${WALKS_INSTANCE}"
echo "  pretrained gensim:   ${OUT}/rdf2vec_pretrained.model"
echo "  checkpoint (.pt):    ${CHECKPOINT}"
echo "  loss PNGs:           ${OUT}/pretrain_per_epoch.png, ${OUT}/protograph_per_epoch.png"
echo "  step loss PNGs:      ${OUT}/pretrain_per_step.png, ${OUT}/protograph_per_step.png"
