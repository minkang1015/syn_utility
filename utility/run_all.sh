#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

DATA_ROOT_REL="./datas"
OUTDIR_REL="./outputs"
OUTDIR_ABS="${ROOT_DIR}/outputs"

DATASETS=("rossmann_subsampled" "walmart_subsampled")
MODES=("tmtr" "tatr")
MODELS=("REALTABFORMER" "RCTGAN" "CLAVADDPM" "RelDiff" "RGCLD" "SDV")

# ✅ 배열 원소는 공백으로 분리해야 함
RUNS=(1)
SAMPLES=("sample1")
SEEDS=(1 2 3)

FILTERING_OPTIONS=("true" "false")

# dataset|mode 별 설정
declare -A CFG_DATES
declare -A CFG_BIN
declare -A CFG_RATIO_MAX
declare -A CFG_GRAN

# Rossmann
CFG_DATES["rossmann_subsampled|tmtr"]="7,14"
CFG_BIN["rossmann_subsampled|tmtr"]="0.1"
CFG_RATIO_MAX["rossmann_subsampled|tmtr"]="1.0"
CFG_GRAN["rossmann_subsampled|tmtr"]="10"

CFG_DATES["rossmann_subsampled|tatr"]="7,14"
CFG_BIN["rossmann_subsampled|tatr"]="0.1"
CFG_RATIO_MAX["rossmann_subsampled|tatr"]="1.0"
CFG_GRAN["rossmann_subsampled|tatr"]="10"

# Walmart
CFG_DATES["walmart_subsampled|tmtr"]="1,2"
CFG_BIN["walmart_subsampled|tmtr"]="0.1"
CFG_RATIO_MAX["walmart_subsampled|tmtr"]="1.0"
CFG_GRAN["walmart_subsampled|tmtr"]="10"

CFG_DATES["walmart_subsampled|tatr"]="1,2"
CFG_BIN["walmart_subsampled|tatr"]="0.1"
CFG_RATIO_MAX["walmart_subsampled|tatr"]="1.0"
CFG_GRAN["walmart_subsampled|tatr"]="10"

TEST_RATIO="0.2"

CONFIG_DIR="${PROJECT_DIR}/configs/generated"
LOG_DIR="${OUTDIR_ABS}/logs"
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${OUTDIR_ABS}"

# sanity check
if [[ ! -d "${ROOT_DIR}/datas" ]]; then
  echo "[ERROR] datas folder not found at: ${ROOT_DIR}/datas"
  exit 1
fi

TOTAL=0
FAIL=0

pushd "${ROOT_DIR}" >/dev/null

for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    key="${dataset}|${mode}"

    dates="${CFG_DATES[$key]:-}"
    bin="${CFG_BIN[$key]:-}"
    ratio_max="${CFG_RATIO_MAX[$key]:-}"
    gran="${CFG_GRAN[$key]:-}"

    if [[ -z "${dates}" || -z "${bin}" || -z "${ratio_max}" || -z "${gran}" ]]; then
      echo "[SKIP] Missing config for key='${key}'"
      continue
    fi

    for method in "${MODELS[@]}"; do
      for run in "${RUNS[@]}"; do
        for sample in "${SAMPLES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            for FILTERING in "${FILTERING_OPTIONS[@]}"; do
            
              TOTAL=$((TOTAL+1))

              safe_dataset="${dataset//[^a-zA-Z0-9._-]/_}"
              safe_method="${method//[^a-zA-Z0-9._-]/_}"
              safe_mode="${mode//[^a-zA-Z0-9._-]/_}"
              safe_run="${run//[^a-zA-Z0-9._-]/_}"
              safe_sample="${sample//[^a-zA-Z0-9._-]/_}"
              safe_seed="${seed//[^a-zA-Z0-9._-]/_}"

              cfg_name="${safe_dataset}__${safe_method}__${safe_mode}__run${safe_run}__${safe_sample}__seed${safe_seed}__filt${FILTERING}.json"
              
              cfg="${CONFIG_DIR}/${cfg_name}"
              log="${LOG_DIR}/${safe_dataset}__${safe_method}__${safe_mode}__run${safe_run}__${safe_sample}__seed${safe_seed}__filt${FILTERING}.log"

              # JSON 생성
              cat > "${cfg}" <<EOF
{
  "data_root": "${DATA_ROOT_REL}",
  "outdir": "${OUTDIR_REL}",

  "dataset": "${dataset}",
  "method": "${method}",
  "run": "${run}",
  "sample": "${sample}",

  "mode": "${mode}",
  "seed": ${seed},
  "test_ratio": ${TEST_RATIO},
  "granularity": ${gran},
  "filtering": ${FILTERING},

  "dates": "${dates}",
  "bin": ${bin},
  "ratio_max": ${ratio_max}
}
EOF

              echo "============================================================"
              echo "[RUN ${TOTAL}] dataset=${dataset} mode=${mode} method=${method} run=${run} sample=${sample} seed=${seed}"
              echo "             filtering=${FILTERING}"
              echo "  cfg=${cfg}"
              echo "  log=${log}"
              echo "============================================================"

              if python -m utility.run --config "${cfg}" 2>&1 | tee "${log}"; then
                echo "  -> OK"
              else
                FAIL=$((FAIL+1))
                echo "  -> FAIL"
              fi
              echo
            done # End filtering loop
          done # End seed loop
        done
      done
    done
  done
done

popd >/dev/null

echo "[DONE] total=${TOTAL} fail=${FAIL}"
if [[ "${FAIL}" -ne 0 ]]; then
  exit 1
fi