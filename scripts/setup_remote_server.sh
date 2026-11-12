#!/usr/bin/env bash
# ==============================================================================
# Remote Data & Media Server Setup Script for AIC-HCMC-26
# Server Target: CPU Server 300GB (192.168.28.112)
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Starting Remote Media Server Setup on 192.168.28.112 ..."
echo "=========================================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "[1/6] Creating directories..."
mkdir -p data/keyframes
mkdir -p data/map-keyframes
mkdir -p data/clip-features-32
mkdir -p data/objects
mkdir -p indexes
mkdir -p downloads

echo "[2/6] Setting up Python environment..."
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown, polars, faiss-cpu, numpy, tqdm..."
    pip3 install --upgrade pip
    pip3 install gdown polars numpy faiss-cpu tqdm
fi

echo "[3/6] Downloading Datasets from Google Drive..."
# Batch 1 Google Drive ID: 1F_l-itCAkSRImn2j0yQo7oUhL4xrFG0R
if [ ! -f "downloads/batch1.zip" ]; then
    echo "Downloading Batch 1 (~10-20GB)..."
    python3 -m gdown "1F_l-itCAkSRImn2j0yQo7oUhL4xrFG0R" -O downloads/batch1.zip --fuzzy || true
fi

# Batch 2 Google Drive ID: 1GKPSHfyZGpN1yK0gcbzBfydZWtZC_guU
if [ ! -f "downloads/batch2.zip" ]; then
    echo "Downloading Batch 2 (~10-20GB)..."
    python3 -m gdown "1GKPSHfyZGpN1yK0gcbzBfydZWtZC_guU" -O downloads/batch2.zip --fuzzy || true
fi

echo "[4/6] Extracting Archives into data/..."
# Function to extract zip if exists
extract_archive() {
    local zip_file="$1"
    if [ -f "${zip_file}" ]; then
        echo "Extracting ${zip_file}..."
        unzip -q -o "${zip_file}" -d data/ || echo "Warning: extraction reported non-zero status"
    fi
}

extract_archive "downloads/batch1.zip"
extract_archive "downloads/batch2.zip"

# Reorganize if extracted into nested folders
echo "Normalizing dataset paths..."
if [ -d "data/data-batch-1" ]; then
    cp -rn data/data-batch-1/* data/ 2>/dev/null || true
fi
if [ -d "data/data-batch-2" ]; then
    cp -rn data/data-batch-2/* data/ 2>/dev/null || true
fi

echo "[5/6] Building Unified Offline Indexes for Batch 1 + Batch 2..."
echo "-> 1. Building metadata.parquet..."
python3 src/offline_pipeline/metadata_loader.py

echo "-> 2. Building faiss_clip.index..."
python3 src/offline_pipeline/build_index.py

echo "-> 3. Building objects.parquet..."
python3 src/offline_pipeline/objects_loader.py || true

echo "Indexes built successfully in ${PROJECT_DIR}/indexes:"
ls -lh indexes/

echo "[6/6] Starting Static HTTP Media Server on Port 8007..."
# Kill any previous server on port 8007
fuser -k 8007/tcp 2>/dev/null || true

# Serve the 'data' directory directly so both keyframes/ and map-keyframes/ are accessible:
# http://192.168.28.112:8007/keyframes/L21_V001/001.jpg
# http://192.168.28.112:8007/map-keyframes/L21_V001.csv
nohup python3 -m http.server 8007 --directory "${PROJECT_DIR}/data" > media_server.log 2>&1 &
SERVER_PID=$!

echo "=========================================================="
echo " Media server is RUNNING in background (PID: ${SERVER_PID})"
echo " Accessible at: http://192.168.28.112:8007"
echo " Keyframes URL: http://192.168.28.112:8007/keyframes/{video_id}/{frame}.jpg"
echo " Map CSV URL:   http://192.168.28.112:8007/map-keyframes/{video_id}.csv"
echo ""
echo " To sync updated indexes back to your local RTX 4060 PC, run this on your PC:"
echo "   scp -r 24521668@192.168.28.112:${PROJECT_DIR}/indexes/* ./indexes/"
echo "=========================================================="
