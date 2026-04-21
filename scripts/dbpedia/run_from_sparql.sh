#!/usr/bin/env bash
# SPARQL-based instance graph + two-stage training (no mapping dump).
#
# Usage:
#   ./scripts/dbpedia/run_from_sparql.sh <entity_list.txt> [<more_entity_lists>...]
#
# Required env (unless SCHEMA_NT is set):
#   DBPEDIA_ONTOLOGY   Path to ontology for export-schema
# Optional:
#   SCHEMA_NT   If set, copy this file as schema instead of export-schema
#   OUT, ENDPOINT, TRAINING_MODE

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <entities.txt> [more_entities.txt ...]" >&2
  exit 1
fi

OUT="${OUT:-output/dbpedia_sparql_run}"
ENDPOINT="${ENDPOINT:-https://dbpedia.org/sparql}"
MODE="${TRAINING_MODE:-p2}"

mkdir -p "$OUT"

if [[ -n "${SCHEMA_NT:-}" ]]; then
  cp "${SCHEMA_NT}" "${OUT}/schema_protograph.nt"
else
  : "${DBPEDIA_ONTOLOGY:?Set DBPEDIA_ONTOLOGY or SCHEMA_NT}"
  uv run dbpedia-build export-schema --ontology "${DBPEDIA_ONTOLOGY}" -o "${OUT}/schema_protograph.nt"
fi

uv run dbpedia-build fetch-sparql \
  --entities "$@" \
  --endpoint "${ENDPOINT}" \
  --batch-size 20 \
  --delay 0.6 \
  -o "${OUT}/instance.nt"

uv run dbpedia-build filter-types \
  --entities "$@" \
  --instance-types "${OUT}/instance.nt" \
  -o "${OUT}/types_from_construct.nt"

uv run dbpedia-build merge-maschine-ontology \
  "${OUT}/schema_protograph.nt" "${OUT}/types_from_construct.nt" \
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
  --mode jrdf2vec-duplicate-free --depth 4 --walks-per-entity 50 --threads 4 --seed 42

WALKS_INST="${OUT}/walks_instance.txt"
uv run src/walk/random_walks.py "${OUT}/instance.nt" "${WALKS_INST}" \
  --mode jrdf2vec-duplicate-free --depth 4 --walks-per-entity 50 --threads 4 --seed 42

uv run src/train/train_word2vec.py "${WALKS_INST}" \
  --mode "${MODE}" \
  --pretrain-walks "${WALKS_PRE}" \
  --ontology "${OUT}/ontology_maschine.nt" \
  --out-dir "${OUT}" \
  -o rdf2vec_final.pt \
  --architecture skipgram \
  --dim 128 \
  --window 5 \
  --negative 5 \
  --min-count 1 \
  --pretrain-epochs 2 \
  --finetune-epochs 2 \
  --workers 0 \
  --maschine-init \
  --loss-every-steps 0

echo "Done. Artifacts under ${OUT}"
