#!/usr/bin/env bash
# Run an e2e script for multiple test cases (TCs) sequentially.
# Each run uses TC=<id>; per-run logs stay under output/<OUTPUT_DIR>/<datetime>/run.log inside the repo.
#
# Usage:
#   ./scripts/run_batch_e2e.sh e2e_no_pretrain.sh tc01 tc02 tc11
#   ./scripts/run_batch_e2e.sh e2e_p1.sh tc12
#
# Env:
#   E2E_SCRIPT  — if set, first positional can be omitted: E2E_SCRIPT=e2e_no_pretrain.sh ./scripts/run_batch_e2e.sh tc01 tc02

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF' >&2
Usage:
  run_batch_e2e.sh <e2e_script> <tc> [tc ...]
  E2E_SCRIPT=<e2e_script> run_batch_e2e.sh <tc> [tc ...]

Examples:
  ./scripts/run_batch_e2e.sh e2e_no_pretrain.sh tc01 tc02 tc11
  ./scripts/run_batch_e2e.sh e2e_p1.sh tc12

e2e_script is a basename under scripts/ (e.g. e2e_no_pretrain.sh) or a path to the script.
EOF
}

resolve_e2e_script() {
  local name="$1"
  if [[ "${name}" == */* ]]; then
    [[ -f "${name}" ]] || { echo "error: not a file: ${name}" >&2; exit 1; }
    printf '%s' "$(cd "$(dirname "${name}")" && pwd)/$(basename "${name}")"
  else
    [[ -f "${SCRIPT_DIR}/${name}" ]] || { echo "error: no script ${SCRIPT_DIR}/${name}" >&2; exit 1; }
    printf '%s' "${SCRIPT_DIR}/${name}"
  fi
}

if [[ -n "${E2E_SCRIPT:-}" ]]; then
  E2E_PATH="$(resolve_e2e_script "${E2E_SCRIPT}")"
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi
  TCS=("$@")
else
  if [[ $# -lt 2 ]]; then
    usage
    exit 1
  fi
  E2E_PATH="$(resolve_e2e_script "$1")"
  shift
  TCS=("$@")
fi

if [[ ${#TCS[@]} -eq 0 ]]; then
  usage
  exit 1
fi

total=${#TCS[@]}
bar_width=24

# Pretty progress to stderr so stdout can be redirected without losing the bar.
progress_line() {
  local current="$1" total="$2" tc="$3" label="$4"
  local pct=$(( current * 100 / total ))
  local filled=$(( current * bar_width / total ))
  local empty=$(( bar_width - filled ))
  local bar
  bar="$(printf '%*s' "${filled}" '' | tr ' ' '█')$(printf '%*s' "${empty}" '' | tr ' ' '░')"
  printf '[%s] %3d%%  %d/%d  %-6s  %s\n' "${bar}" "${pct}" "${current}" "${total}" "${tc}" "${label}" >&2
}

batch_start_ts="$(date +%s)"

echo >&2 ""
echo >&2 "━━ Batch e2e ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo >&2 "  script:  ${E2E_PATH}"
echo >&2 "  TCs:     ${TCS[*]} (${total} total)"
echo >&2 "  repo:    ${REPO_ROOT}"
echo >&2 "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo >&2 ""

failed=()
idx=0
for tc in "${TCS[@]}"; do
  idx=$((idx + 1))

  # Suppress console output from each e2e (they still tee to output/.../run.log).
  set +e
  (cd "${REPO_ROOT}" && TC="${tc}" bash "${E2E_PATH}") >/dev/null 2>&1
  rc=$?
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    progress_line "${idx}" "${total}" "${tc}" "OK"
  else
    progress_line "${idx}" "${total}" "${tc}" "FAIL (${rc})"
    failed+=("${tc}")
  fi
done

elapsed=$(( $(date +%s) - batch_start_ts ))
echo >&2 ""
echo >&2 "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ ${#failed[@]} -eq 0 ]]; then
  printf >&2 "  Done: %d/%d succeeded in %ds\n" "${total}" "${total}" "${elapsed}"
  echo >&2 "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
else
  printf >&2 "  Finished with failures: %d/%d ok, %d failed in %ds\n" \
    "$(( total - ${#failed[@]} ))" "${total}" "${#failed[@]}" "${elapsed}"
  echo >&2 "  Failed TCs: ${failed[*]}"
  echo >&2 "  (Check output/<run>/run.log under each TC’s OUTPUT_DIR for details.)"
  echo >&2 "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 1
fi
