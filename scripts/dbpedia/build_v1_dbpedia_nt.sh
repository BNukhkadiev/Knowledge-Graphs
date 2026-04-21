#!/usr/bin/env bash
# Build v1/dbpedia/dbpedia.nt: every resource URI under v1/dbpedia/**/*.txt plus k-hop
# context from a DBpedia mapping-based object-properties dump.
#
# Usage:
#   export DBPEDIA_MAPPING=/path/to/mappingbased_objects_wkd_uris_en.nt.bz2
#   ./scripts/dbpedia/build_v1_dbpedia_nt.sh
#
# Optional env:
#   V1_ROOT       (default: v1/dbpedia)
#   OUT           (default: v1/dbpedia/dbpedia.nt)
#   HOPS          (default: 2)
#   CACHE_ENTITIES  If set (path), write sorted URI list there when scanning v1
#   AUGMENT_SPARQL  If set to 1, run SPARQL for seeds missing as subject after dump extract

set -euo pipefail

: "${DBPEDIA_MAPPING:?Set DBPEDIA_MAPPING to mapping-based .nt or .nt.bz2}"

ROOT="${V1_ROOT:-v1/dbpedia}"
OUT="${OUT:-v1/dbpedia/dbpedia.nt}"
HOPS="${HOPS:-2}"

EXTRA=()
if [[ -n "${CACHE_ENTITIES:-}" ]]; then
  EXTRA+=(--cache-entities "${CACHE_ENTITIES}")
fi
if [[ "${AUGMENT_SPARQL:-0}" == "1" ]]; then
  EXTRA+=(--augment-with-sparql)
fi

uv run dbpedia-build build-v1-instance \
  --v1-root "${ROOT}" \
  --mapping-based "${DBPEDIA_MAPPING}" \
  --hops "${HOPS}" \
  -o "${OUT}" \
  "${EXTRA[@]}"

echo "Wrote ${OUT}"
