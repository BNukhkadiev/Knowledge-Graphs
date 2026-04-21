#!/usr/bin/env bash
# Offline DBpedia → protograph → walks → two-stage RDF2Vec (p2 example).
#
# Usage:
#   ./scripts/dbpedia/run_from_dumps.sh <entity_list.txt> [<more_entity_lists>...]
#
# Required env:
#   DBPEDIA_ONTOLOGY        Path to dbpedia.owl (RDFLib-readable)
#   DBPEDIA_MAPPING         mapping-based object properties .nt or .nt.bz2
#   DBPEDIA_INSTANCE_TYPES  instance types .nt or .nt.bz2
#
# Optional env:
#   OUT, HOPS, TRAINING_MODE, PRETRAIN_EPOCHS, FINETUNE_EPOCHS, DIM, DEPTH, WALKS_PER_ENTITY

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <entities.txt> [more_entities.txt ...]" >&2
  exit 1
fi

: "${DBPEDIA_ONTOLOGY:?Set DBPEDIA_ONTOLOGY}"
: "${DBPEDIA_MAPPING:?Set DBPEDIA_MAPPING}"
: "${DBPEDIA_INSTANCE_TYPES:?Set DBPEDIA_INSTANCE_TYPES}"

OUT="${OUT:-output/dbpedia_run}"
HOPS="${HOPS:-1}"
MODE="${TRAINING_MODE:-p2}"
PRE="${PRETRAIN_EPOCHS:-3}"
FT="${FINETUNE_EPOCHS:-3}"
DIM="${DIM:-128}"
DEPTH="${DEPTH:-4}"
WPE="${WALKS_PER_ENTITY:-100}"

mkdir -p "$OUT"

uv run dbpedia-build export-schema --ontology "${DBPEDIA_ONTOLOGY}" -o "${OUT}/schema_protograph.nt"

uv run dbpedia-build filter-instance \
  --entities "$@" \
  --mapping-based "${DBPEDIA_MAPPING}" \
  --hops "${HOPS}" \
  -o "${OUT}/instance.nt"

uv run dbpedia-build filter-types \
  --entities "$@" \
  --instance-types "${DBPEDIA_INSTANCE_TYPES}" \
  -o "${OUT}/types_seed.nt"

uv run dbpedia-build merge-maschine-ontology \
  "${OUT}/schema_protograph.nt" "${OUT}/types_seed.nt" \
  -o "${OUT}/ontology_maschine.nt"

if [[ "${SKIP_KG:-0}" == "1" ]]; then
  uv run protograph --schema "${OUT}/schema_protograph.nt" --out-dir "${OUT}"
else
  uv run protograph \
    --schema "${OUT}/schema_protograph.nt" \
    --kg "${OUT}/instance.nt" \
    --out-dir "${OUT}"
fi

if [[ "${MODE}" == "p1" ]]; then
  PROTO_NT="${OUT}/protograph_p1.nt"
  WALKS_PRE="${OUT}/walks_p1.txt"
else
  PROTO_NT="${OUT}/protograph_p2.nt"
  WALKS_PRE="${OUT}/walks_p2.txt"
fi

uv run src/walk/random_walks.py "${PROTO_NT}" "${WALKS_PRE}" \
  --mode jrdf2vec-duplicate-free --depth "${DEPTH}" --walks-per-entity "${WPE}" --threads 8 --seed 42

WALKS_INST="${OUT}/walks_instance.txt"
WALK_THREADS="${WALK_THREADS:-8}"
WALK_SEED="${WALK_SEED:-42}"
uv run src/walk/random_walks.py "${OUT}/instance.nt" "${WALKS_INST}" \
  --mode jrdf2vec-duplicate-free \
  --depth "${DEPTH}" \
  --walks-per-entity "${WPE}" \
  --threads "${WALK_THREADS}" \
  --seed "${WALK_SEED}"

uv run src/train/train_word2vec.py "${WALKS_INST}" \
  --mode "${MODE}" \
  --pretrain-walks "${WALKS_PRE}" \
  --ontology "${OUT}/ontology_maschine.nt" \
  --out-dir "${OUT}" \
  -o rdf2vec_final.pt \
  --architecture skipgram \
  --dim "${DIM}" \
  --window 5 \
  --negative 5 \
  --min-count 1 \
  --lr 0.025 \
  --pretrain-epochs "${PRE}" \
  --finetune-epochs "${FT}" \
  --workers 0 \
  --maschine-init \
  --loss-every-steps 0

echo "Artifacts under ${OUT}"
