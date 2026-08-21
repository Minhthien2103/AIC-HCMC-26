#!/usr/bin/env bash
set -Eeuo pipefail

# Stage the old Google Drive assets onto Colab's local SSD while keeping code
# on the requested Git branch. Override these variables only if Drive uses a
# different layout from MyDrive/AIC_HCMC_26/AIC-HCMC-26.
AIC_DRIVE_REPO="${AIC_DRIVE_REPO:-/content/drive/MyDrive/AIC_HCMC_26/AIC-HCMC-26}"
AIC_WORK_ROOT="${AIC_WORK_ROOT:-/content/aic26}"
AIC_CODE_DIR="${AIC_CODE_DIR:-${AIC_WORK_ROOT}/AIC-HCMC-26}"
AIC_ASSET_ROOT="${AIC_ASSET_ROOT:-${AIC_WORK_ROOT}/assets}"
AIC_BRANCH="${AIC_BRANCH:-aic2026-submission}"
AIC_REPO_URL="${AIC_REPO_URL:-https://github.com/Minhthien2103/AIC-HCMC-26.git}"
HF_HOME="${HF_HOME:-${AIC_WORK_ROOT}/huggingface}"
export HF_HOME

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "SETUP FAILED: missing $1" >&2
    exit 1
  fi
}

copy_file_atomic() {
  local source="$1"
  local destination="$2"
  if [[ -f "$destination" ]] && [[ "$(stat -c %s "$source")" == "$(stat -c %s "$destination")" ]]; then
    return
  fi
  mkdir -p "$(dirname "$destination")"
  cp "$source" "${destination}.part"
  mv "${destination}.part" "$destination"
}

link_asset() {
  local source="$1"
  local destination="$2"
  if [[ -L "$destination" ]] && [[ "$(readlink -f "$destination")" == "$(readlink -f "$source")" ]]; then
    return
  fi
  if [[ -e "$destination" || -L "$destination" ]]; then
    echo "SETUP FAILED: $destination exists and is not the expected symlink" >&2
    exit 1
  fi
  ln -s "$source" "$destination"
}

require_path "$AIC_DRIVE_REPO"
require_path "$AIC_DRIVE_REPO/keyframes.zip"
require_path "$AIC_DRIVE_REPO/data/map-keyframes"
require_path "$AIC_DRIVE_REPO/indexes/faiss_clip.index"
require_path "$AIC_DRIVE_REPO/indexes/metadata.parquet"
require_path "$AIC_DRIVE_REPO/indexes/objects.parquet"
require_path "$AIC_DRIVE_REPO/SOTUYEN1-bo-de-thi.zip"

mkdir -p "$AIC_WORK_ROOT"
if [[ ! -d "$AIC_CODE_DIR/.git" ]]; then
  git clone --depth 1 --branch "$AIC_BRANCH" "$AIC_REPO_URL" "$AIC_CODE_DIR"
else
  if [[ -n "$(git -C "$AIC_CODE_DIR" status --porcelain)" ]]; then
    echo "SETUP FAILED: local Colab checkout has uncommitted changes: $AIC_CODE_DIR" >&2
    exit 1
  fi
  git -C "$AIC_CODE_DIR" fetch origin "$AIC_BRANCH"
  git -C "$AIC_CODE_DIR" checkout "$AIC_BRANCH"
  git -C "$AIC_CODE_DIR" pull --ff-only origin "$AIC_BRANCH"
fi

echo "Installing the deadline-safe runtime (Paddle/OCR and ViT-H are intentionally omitted)..."
python -m pip install -q \
  "accelerate>=0.34.0" \
  "deep-translator>=1.11.4" \
  "faiss-cpu>=1.8.0" \
  "huggingface_hub>=0.24.0" \
  "open_clip_torch>=2.24.0" \
  "Pillow>=10.0.0" \
  "polars>=0.20.0" \
  "sentencepiece" \
  "transformers>=4.45.0,<5.0.0"

mkdir -p "$AIC_ASSET_ROOT/data" "$AIC_ASSET_ROOT/indexes"
KEYFRAME_MARKER="$AIC_ASSET_ROOT/data/.keyframes-complete"
if [[ ! -f "$KEYFRAME_MARKER" ]]; then
  echo "Extracting only keyframes/* from the 28.4 GB Drive archive onto local SSD..."
  mkdir -p "$AIC_ASSET_ROOT/data/keyframes"
  unzip -q -o "$AIC_DRIVE_REPO/keyframes.zip" "keyframes/*" -d "$AIC_ASSET_ROOT/data"
  KEYFRAME_COUNT="$(find "$AIC_ASSET_ROOT/data/keyframes" -type f -name '*.jpg' | wc -l)"
  if (( KEYFRAME_COUNT < 170000 )); then
    echo "SETUP FAILED: extracted only $KEYFRAME_COUNT JPEG keyframes" >&2
    exit 1
  fi
  touch "$KEYFRAME_MARKER"
fi

if [[ ! -d "$AIC_ASSET_ROOT/data/map-keyframes" ]]; then
  echo "Copying frame maps from Drive..."
  cp -a "$AIC_DRIVE_REPO/data/map-keyframes" "$AIC_ASSET_ROOT/data/map-keyframes"
fi

for name in faiss_clip.index metadata.parquet objects.parquet; do
  copy_file_atomic "$AIC_DRIVE_REPO/indexes/$name" "$AIC_ASSET_ROOT/indexes/$name"
done

mkdir -p "$AIC_CODE_DIR/data" "$AIC_CODE_DIR/query"
link_asset "$AIC_ASSET_ROOT/data/keyframes" "$AIC_CODE_DIR/data/keyframes"
link_asset "$AIC_ASSET_ROOT/data/map-keyframes" "$AIC_CODE_DIR/data/map-keyframes"
link_asset "$AIC_ASSET_ROOT/indexes" "$AIC_CODE_DIR/indexes"

QUERY_DIR="$AIC_CODE_DIR/query/SOTUYEN1-bo-de-thi"
mkdir -p "$QUERY_DIR"
unzip -q -o -j "$AIC_DRIVE_REPO/SOTUYEN1-bo-de-thi.zip" '*.txt' -d "$QUERY_DIR"
QUERY_COUNT="$(find "$QUERY_DIR" -maxdepth 1 -type f -name '*.txt' | wc -l)"
if [[ "$QUERY_COUNT" != "25" ]]; then
  echo "SETUP FAILED: expected 25 query files, found $QUERY_COUNT" >&2
  exit 1
fi

# Reuse either cache layout visible in the user's Drive screenshot. The final
# snapshot_download calls repair a partial cache and fetch only missing files.
LOCAL_HUB="$HF_HOME/hub"
mkdir -p "$LOCAL_HUB"
for model_dir in \
  models--Qwen--Qwen2-VL-7B-Instruct \
  models--facebook--mbart-large-50-many-to-many-mmt; do
  if [[ ! -d "$LOCAL_HUB/$model_dir" ]]; then
    for drive_hub in \
      "$AIC_DRIVE_REPO/huggingface/hub" \
      "$AIC_DRIVE_REPO/huggingface" \
      "$AIC_DRIVE_REPO/hf-cache/hub" \
      "$AIC_DRIVE_REPO/hf-cache"; do
      if [[ -d "$drive_hub/$model_dir/snapshots" ]]; then
        echo "Copying cached $model_dir from Drive to local SSD..."
        cp -a "$drive_hub/$model_dir" "$LOCAL_HUB/$model_dir"
        break
      fi
    done
  fi
done

echo "Verifying/downloading Qwen2-VL and mBART model snapshots..."
python - <<'PY'
from huggingface_hub import snapshot_download

for model in (
    "Qwen/Qwen2-VL-7B-Instruct",
    "facebook/mbart-large-50-many-to-many-mmt",
):
    print(snapshot_download(model, max_workers=8))
PY

echo "Warming the exact OpenCLIP ViT-B/32 checkpoint used by the FAISS index..."
cd "$AIC_CODE_DIR"
python - <<'PY'
from src.online_pipeline.query_encoder import QueryEncoder

QueryEncoder("ViT-B-32-quickgelu", "openai", device="cpu")
print("OpenCLIP cache: OK")
PY

echo "SETUP COMPLETE"
echo "Code: $AIC_CODE_DIR"
echo "Commit: $(git -C "$AIC_CODE_DIR" rev-parse --short HEAD)"
echo "Queries: $QUERY_COUNT"
echo "Keyframes: $(find "$AIC_ASSET_ROOT/data/keyframes" -type f -name '*.jpg' | wc -l)"
df -h /content | tail -n 1
