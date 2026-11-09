#!/usr/bin/env bash
set -Eeuo pipefail

AIC_DRIVE_REPO="${AIC_DRIVE_REPO:-/content/drive/MyDrive/AIC_HCMC_26/AIC-HCMC-26}"
AIC_WORK_ROOT="${AIC_WORK_ROOT:-/content/aic26}"
AIC_CODE_DIR="${AIC_CODE_DIR:-${AIC_WORK_ROOT}/AIC-HCMC-26}"
HF_HOME="${HF_HOME:-${AIC_WORK_ROOT}/huggingface}"
export HF_HOME PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="${AIC_OUTPUT_DIR:-${AIC_DRIVE_REPO}/outputs}"
OUTPUT_ZIP="${AIC_OUTPUT_ZIP:-${OUTPUT_DIR}/SOTUYEN1_kis_baseline_submission.zip}"
CHECKPOINT_DIR="${AIC_CHECKPOINT_DIR:-${OUTPUT_DIR}/SOTUYEN1_kis_baseline_checkpoint}"
OLD_CHECKPOINT_DIR="${AIC_OLD_CHECKPOINT_DIR:-${OUTPUT_DIR}/SOTUYEN1_a100_checkpoint}"
PROVENANCE_DIR="${AIC_PROVENANCE_DIR:-${OUTPUT_DIR}/SOTUYEN1_kis_baseline_provenance}"
REVIEW_DIR="${AIC_REVIEW_DIR:-${AIC_WORK_ROOT}/review_kis_baseline}"
LOG_PATH="${AIC_LOG_PATH:-${OUTPUT_DIR}/SOTUYEN1_kis_baseline_run.log}"

if [[ ! -d "$AIC_CODE_DIR/.git" ]]; then
  echo "RUN FAILED: code checkout not found at $AIC_CODE_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$PROVENANCE_DIR" "$REVIEW_DIR"

# Preserve the already generated non-KIS work; this run intentionally
# regenerates only the 20 KIS queries into a separate checkpoint directory.
for name in \
  query-p1-15-qa.csv \
  query-p1-16-trake.csv \
  query-p1-17-qa.csv \
  query-p1-3-qa.csv \
  query-p1-9-qa.csv
do
  if [[ ! -f "$OLD_CHECKPOINT_DIR/$name" ]]; then
    echo "RUN FAILED: missing reusable checkpoint $OLD_CHECKPOINT_DIR/$name" >&2
    exit 1
  fi
  if [[ ! -f "$CHECKPOINT_DIR/$name" ]]; then
    cp "$OLD_CHECKPOINT_DIR/$name" "$CHECKPOINT_DIR/$name"
  fi
done

cd "$AIC_CODE_DIR"
echo "KIS baseline commit=$(git rev-parse --short HEAD) reused_non_kis=5 output=$OUTPUT_ZIP"

python scripts/preflight.py \
  --repo-root . \
  --kis-profile fast \
  --vlm-mode bf16 \
  --skip-model

time python -u scripts/generate_submission.py \
  --queries-dir query/SOTUYEN1-bo-de-thi \
  --output "$OUTPUT_ZIP" \
  --repo-root . \
  --device cuda \
  --vlm-mode bf16 \
  --kis-profile fast \
  --kis-baseline-simple \
  --disable-kis-ocr \
  --kis-candidate-budget 1000 \
  --kis-video-budget 8 \
  --kis-local-frame-budget 48 \
  --kis-query-variant-limit 4 \
  --frame-neighborhood-count 7 \
  --vqa-top-k 100 \
  --vqa-qwen-candidate-budget 96 \
  --trake-top-k 100 \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --resume \
  --review-output-dir "$REVIEW_DIR" \
  --provenance-dir "$PROVENANCE_DIR" \
  --max-rows 100 2>&1 | tee "$LOG_PATH"

python scripts/validate_submission.py \
  --zip "$OUTPUT_ZIP" \
  --queries-dir query/SOTUYEN1-bo-de-thi

echo "READY TO UPLOAD: $OUTPUT_ZIP"
ls -lh "$OUTPUT_ZIP"
