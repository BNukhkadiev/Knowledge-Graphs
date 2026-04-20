#!/usr/bin/env bash
# Build instance.nt for DBpedia TC01 (people / dbo:child) via public endpoint CONSTRUCT.
#
# TC01 entity definitions match the illustrative SELECTs in:
#   scripts/dbpedia/dlcc_table2_dbpedia_sparql.sparql  (dbo:Person + dbo:child)
# Gold lists live under v1/dbpedia/tc01/people/<N>/.
#
# Usage:
#   ./scripts/dbpedia/build_tc01_instance_sparql.sh [N]
# N defaults to 50 (smallest bundled split). Other values: 500, 5000.
#
# Env:
#   OUT, ENDPOINT — same semantics as run_from_sparql.sh

set -euo pipefail

N="${1:-50}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENT="${ROOT}/v1/dbpedia/tc01/people/${N}"
OUT="${OUT:-${ROOT}/output/dbpedia_tc01_people_${N}}"
ENDPOINT="${ENDPOINT:-https://dbpedia.org/sparql}"

if [[ ! -d "$ENT" ]]; then
  echo "missing entity dir: $ENT" >&2
  exit 1
fi

mkdir -p "$OUT"

uv run dbpedia-build fetch-sparql \
  --entities \
    "${ENT}/positives.txt" \
    "${ENT}/negatives.txt" \
    "${ENT}/negatives_hard.txt" \
    "${ENT}/train_test/train.txt" \
    "${ENT}/train_test/test.txt" \
    "${ENT}/train_test_hard/train.txt" \
    "${ENT}/train_test_hard/test.txt" \
  --endpoint "${ENDPOINT}" \
  --batch-size 15 \
  --delay 0.7 \
  -o "${OUT}/instance.nt"

echo "Wrote ${OUT}/instance.nt"
