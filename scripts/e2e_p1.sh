#!/usr/bin/env bash
# End-to-end RDF2Vec with MASCHInE protograph P1 pretraining:
#   ontology → P1 NT → walks on P1 → instance walks → Word2Vec (pretrain + finetune) → eval.
# All artifacts go under output/${OUTPUT_DIR}/<datetime>/.
#
# Mirrors scripts/e2e_no_pretrain.sh (same walk/train hyperparameters where applicable; TC via $TC)
# but uses --mode p1 instead of a single-corpus train.

set -euo pipefail

# Test case id (folder under v1/synthetic_ontology/<TC>/...). Override: TC=tc11 ./scripts/e2e_p1.sh
TC="${TC:-tc12}"
OUTPUT_DIR="${OUTPUT_DIR:-${TC}_rdf2vec_p1}"
RUN_DATETIME="$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUT="${REPO_ROOT}/output/${OUTPUT_DIR}/${RUN_DATETIME}"
mkdir -p "${OUT}"

# Dataset paths (same layout as e2e_no_pretrain.sh)
BASE_ONTO="${REPO_ROOT}/v1/synthetic_ontology/${TC}/synthetic_ontology"
GRAPH_NT="${BASE_ONTO}/graph.nt"
ONTOLOGY_NT="${BASE_ONTO}/ontology.nt"
TEST_TXT="${BASE_ONTO}/1000/train_test/test.txt"

PROT_P1_NT="${OUT}/protograph_p1.nt"
WALKS_P1="${OUT}/walks_p1.txt"
WALKS_INSTANCE="${OUT}/walks_instance.txt"
if [[ -n "${PRECOMPUTED_INSTANCE_WALKS:-}" ]]; then
  WALKS_INSTANCE="${PRECOMPUTED_INSTANCE_WALKS}"
fi
CHECKPOINT="${OUT}/rdf2vec_final.pt"
RUN_LOG="${OUT}/run.log"

# Optional overrides (defaults align with e2e_no_pretrain.sh: depth 4, 100 walks/entity, dim 100, window 5)
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-5}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
# Step-loss batch interval: small P1 pretrain corpus → dense logging (default 1); instance finetune → sparser (default 100).
# If LOSS_EVERY_STEPS is set, it applies to both stages (--loss-every-steps), matching the old single-flag behavior.
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

echo "== 1/5 Build protograph P1 → ${PROT_P1_NT}"
uv run protograph \
  --schema "${ONTOLOGY_NT}" \
  --kg "${GRAPH_NT}" \
  --out-dir "${OUT}"

echo "== 2/5 Random walks on P1 → ${WALKS_P1}"
uv run src/walk/random_walks.py "${PROT_P1_NT}" "${WALKS_P1}" \
  --mode "jrdf2vec-duplicate-free" \
  --depth 4 \
  --walks-per-entity 100

if [[ -n "${PRECOMPUTED_INSTANCE_WALKS:-}" ]]; then
  echo "== 3/5 Skip instance walks (PRECOMPUTED_INSTANCE_WALKS=${WALKS_INSTANCE})"
else
  echo "== 3/5 Random walks on instance KG → ${WALKS_INSTANCE}"
  uv run src/walk/random_walks.py "${GRAPH_NT}" "${WALKS_INSTANCE}" \
    --mode "jrdf2vec-duplicate-free" \
    --depth 4 \
    --walks-per-entity 100
fi

echo "== 4/5 Train Word2Vec (P1 pretrain + instance finetune) → ${CHECKPOINT}"
uv run src/train/train_word2vec.py \
  --mode p1 \
  --pretrain-walks "${WALKS_P1}" \
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
echo "  walks (P1):          ${WALKS_P1}"
echo "  walks (instance):    ${WALKS_INSTANCE}"
echo "  pretrained gensim:   ${OUT}/rdf2vec_pretrained.model"
echo "  checkpoint (.pt):    ${CHECKPOINT}"
echo "  loss PNGs:           ${OUT}/pretrain_per_epoch.png, ${OUT}/protograph_per_epoch.png"
echo "  step loss PNGs:      ${OUT}/pretrain_per_step.png, ${OUT}/protograph_per_step.png"
