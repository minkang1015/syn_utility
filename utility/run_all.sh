#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

DATA_ROOT_REL="./datas"
OUTDIR_REL="./outputs"
OUTDIR_ABS="${ROOT_DIR}/outputs"

DATASETS=("rossmann_subsampled"  "walmart_subsampled")  # "rossmann_subsampled"  "walmart_subsampled" "berka" "freddiemac"
MODES=("tmtr" "tatr")   # "tmtr" "tatr"
MODELS=("CLAVADDPM" "RCTGAN" "RelDiff" "RGCLD" "SDV") # "CLAVADDPM" "RCTGAN" "REALTABFORMER" "RelDiff" "RGCLD" "SDV"

RUNS=(1)  # 수정하지 않아도 됨
SAMPLES=("sample1") # 수정하지 않아도 됨    # 합성 데이터 디렉토리가 ./datas/synthetic/DATASETS/MODELS/RUNS/SAMPLES/xxx.csv 식으로 존재해야함
SEEDS=(1 2 3) # seed 설정 1 2 3                # (예시: ./datas/synthetic/ptbxl/RCTGAN/1/sample1/records.csv)
FILTERING_OPTIONS=("true" "false")  # "true" "false"

declare -A CFG_DATES
declare -A CFG_BIN
declare -A CFG_RATIO_MAX
declare -A CFG_GRAN

# 1. Rossmann
CFG_DATES["rossmann_subsampled"]="7,14"
CFG_BIN["rossmann_subsampled"]="0.1"
CFG_RATIO_MAX["rossmann_subsampled"]="1.0"
CFG_GRAN["rossmann_subsampled"]="10"

# 2. Walmart
CFG_DATES["walmart_subsampled"]="1,2"
CFG_BIN["walmart_subsampled"]="0.1"
CFG_RATIO_MAX["walmart_subsampled"]="1.0"
CFG_GRAN["walmart_subsampled"]="10"

# 3. PTB-XL
CFG_DATES["ptbxl"]="5,10,15,20"
CFG_BIN["ptbxl"]="0.1"
CFG_RATIO_MAX["ptbxl"]="1.0"
CFG_GRAN["ptbxl"]="10"

# 4. Freddie Mac
CFG_DATES["freddiemac"]="1,2"
CFG_BIN["freddiemac"]="0.1"
CFG_RATIO_MAX["freddiemac"]="0.9"
CFG_GRAN["freddiemac"]="10"

# 5. Fannie Mae
CFG_DATES["fanniemae"]="1,2"
CFG_BIN["fanniemae"]="0.1"
CFG_RATIO_MAX["fanniemae"]="1.0"
CFG_GRAN["fanniemae"]="10"

# 6. Berka
CFG_DATES["berka"]="5,10,20"
CFG_BIN["berka"]="0.1"
CFG_RATIO_MAX["berka"]="1.0"
CFG_GRAN["berka"]="10"

TEST_RATIO="0.2"

CONFIG_DIR="${PROJECT_DIR}/configs/generated"
LOG_DIR="${OUTDIR_ABS}/logs"
mkdir -p "${CONFIG_DIR}" "${LOG_DIR}" "${OUTDIR_ABS}"

if [[ ! -d "${ROOT_DIR}/datas" ]]; then
  echo "[ERROR] datas folder not found at: ${ROOT_DIR}/datas"
  exit 1
fi

TOTAL=0
FAIL=0

pushd "${ROOT_DIR}" >/dev/null

for dataset in "${DATASETS[@]}"; do
  dates="${CFG_DATES[$dataset]:-}"
  bin="${CFG_BIN[$dataset]:-}"
  ratio_max="${CFG_RATIO_MAX[$dataset]:-}"
  gran="${CFG_GRAN[$dataset]:-}"

  if [[ -z "${dates}" ]]; then
    echo "[SKIP] Missing config for dataset='${dataset}'"
    continue
  fi

  for mode in "${MODES[@]}"; do
    for method in "${MODELS[@]}"; do
      for run in "${RUNS[@]}"; do
        for sample in "${SAMPLES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            for filtering in "${FILTERING_OPTIONS[@]}"; do
            
              TOTAL=$((TOTAL+1))

              safe_dataset="${dataset//[^a-zA-Z0-9._-]/_}"
              safe_method="${method//[^a-zA-Z0-9._-]/_}"
              safe_mode="${mode//[^a-zA-Z0-9._-]/_}"

              cfg_name="${safe_dataset}__${safe_method}__${safe_mode}__run${run}__${sample}__seed${seed}__filt${filtering}.json"
              cfg="${CONFIG_DIR}/${cfg_name}"
              log="${LOG_DIR}/${cfg_name%.json}.log"

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
  "filtering": ${filtering},
  "dates": "${dates}",
  "bin": ${bin},
  "ratio_max": ${ratio_max}
}
EOF

              echo "============================================================"
              echo "[RUN ${TOTAL}] dataset=${dataset} mode=${mode} method=${method} seed=${seed} filtering=${filtering}"
              echo "  cfg=${cfg}"
              echo "============================================================"

              if python -m utility.run --config "${cfg}" 2>&1 | tee "${log}"; then
                echo "  -> OK"
              else
                FAIL=$((FAIL+1))
                echo "  -> FAIL"
              fi
              echo
            done
          done
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