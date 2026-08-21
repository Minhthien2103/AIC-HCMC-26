#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-standard}"
AIC_DRIVE_REPO="${AIC_DRIVE_REPO:-/content/drive/MyDrive/AIC_HCMC_26/AIC-HCMC-26}"
AIC_WORK_ROOT="${AIC_WORK_ROOT:-/content/aic26}"
AIC_CODE_DIR="${AIC_CODE_DIR:-${AIC_WORK_ROOT}/AIC-HCMC-26}"
HF_HOME="${HF_HOME:-${AIC_WORK_ROOT}/huggingface}"
export HF_HOME PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="${AIC_OUTPUT_DIR:-${AIC_DRIVE_REPO}/outputs}"
OUTPUT_ZIP="${AIC_OUTPUT_ZIP:-${OUTPUT_DIR}/SOTUYEN1_a100_submission.zip}"
CHECKPOINT_DIR="${AIC_CHECKPOINT_DIR:-${OUTPUT_DIR}/SOTUYEN1_a100_checkpoint}"
PROVENANCE_DIR="${AIC_PROVENANCE_DIR:-${OUTPUT_DIR}/SOTUYEN1_a100_provenance}"
REVIEW_DIR="${AIC_REVIEW_DIR:-${AIC_WORK_ROOT}/review}"
LOG_PATH="${AIC_LOG_PATH:-${OUTPUT_DIR}/SOTUYEN1_a100_run.log}"

case "$PROFILE" in
  standard)
    KIS_VLM_TOP_K=64
    KIS_VIDEO_BUDGET=16
    KIS_LOCAL_FRAME_BUDGET=48
    KIS_QUERY_VARIANT_LIMIT=3
    NEIGHBORHOOD_COUNT=7
    TRAKE_TOP_VIDEOS=5
    TRAKE_EVENT_TOP_K=16
    TRAKE_QWEN_PER_EVENT=4
    ;;
  emergency)
    KIS_VLM_TOP_K=32
    KIS_VIDEO_BUDGET=12
    KIS_LOCAL_FRAME_BUDGET=32
    KIS_QUERY_VARIANT_LIMIT=2
    NEIGHBORHOOD_COUNT=5
    TRAKE_TOP_VIDEOS=4
    TRAKE_EVENT_TOP_K=12
    TRAKE_QWEN_PER_EVENT=3
    ;;
  *)
    echo "Usage: $0 [standard|emergency]" >&2
    exit 2
    ;;
esac

if [[ ! -d "$AIC_CODE_DIR/.git" ]]; then
  echo "RUN FAILED: execute scripts/colab_sotuyen1_setup.sh first" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR" "$PROVENANCE_DIR" "$REVIEW_DIR"
cd "$AIC_CODE_DIR"

echo "Profile=$PROFILE commit=$(git rev-parse --short HEAD) completed_checkpoints=$(find "$CHECKPOINT_DIR" -maxdepth 1 -name '*.csv' | wc -l)/25"
echo "Output=$OUTPUT_ZIP"

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
  --disable-kis-ocr \
  --kis-candidate-budget 1000 \
  --kis-vlm-top-k "$KIS_VLM_TOP_K" \
  --kis-video-budget "$KIS_VIDEO_BUDGET" \
  --kis-local-frame-budget "$KIS_LOCAL_FRAME_BUDGET" \
  --kis-query-variant-limit "$KIS_QUERY_VARIANT_LIMIT" \
  --frame-neighborhood-count "$NEIGHBORHOOD_COUNT" \
  --vqa-top-k 100 \
  --trake-top-k 100 \
  --trake-top-videos "$TRAKE_TOP_VIDEOS" \
  --trake-event-top-k "$TRAKE_EVENT_TOP_K" \
  --trake-qwen-per-event "$TRAKE_QWEN_PER_EVENT" \
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
