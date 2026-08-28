# %% [markdown]
# # AIC HCMC 2026 — chạy SOTUYEN2 hoàn toàn trên Kaggle
#
# Notebook này dùng Zilliz cho visual, subtitle và object detection; ảnh
# Qwen/OCR được đọc trực tiếp từ các Kaggle input đã giải nén. Notebook **không
# copy 29 GB ảnh**: nó chỉ tạo symlink theo từng video trong `/kaggle/temp`.
#
# Trước khi chạy:
#
# 1. Chọn **Add Input** và attach đủ bốn nguồn:
#    - `hoanghieu26/aic-hcmc-26-dake-keyframes`
#    - `hoanghieu26/aic-hcmc-2026-dake-l26-b-5`
#    - output của `khoanguyen2024/notebook743841c790`
#    - `hoanghieu26/sotuyen2`
# 2. Trong **Add-ons → Secrets**, tạo `ZILLIZ_URI` và `ZILLIZ_TOKEN` (hoặc
#    dùng tên cũ `MILVUS_URI`, `MILVUS_TOKEN`), rồi bật quyền truy cập cho
#    notebook. Không dán token vào cell.
# 3. Bật **Internet** để clone GitHub, kết nối Zilliz và tải model. GPU chưa cần
#    cho setup/preflight; cell chạy 30 query mới bắt buộc GPU. T4 dùng `4bit`,
#    A100 tự chọn `bf16`.
#
# Canonical `metadata.parquet` được xuất lại từ scalar fields của collection
# visual, không tải embedding 512 chiều và không cần Google Drive.

# %%
from pathlib import Path
import os
import re
import subprocess
import sys

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_TEMP = Path("/kaggle/temp")
if not KAGGLE_INPUT.is_dir() or not KAGGLE_WORKING.is_dir():
    raise RuntimeError("Notebook này phải được chạy trong Kaggle Notebook")
KAGGLE_TEMP.mkdir(parents=True, exist_ok=True)

top_level_inputs = sorted(path.name for path in KAGGLE_INPUT.iterdir())
print("Attached input roots:", len(top_level_inputs))
for name in top_level_inputs:
    print(" -", name)

# %% [markdown]
# ## 1. Clone đúng branch `dev` và cài dependencies
#
# Repo nằm trên `/kaggle/temp` để model cache, source code và symlink không bị
# đóng gói vào output khi Save Version. Khi chạy lại cell, code được đồng bộ về
# `origin/dev`.

# %%
REPO_URL = "https://github.com/Minhthien2103/AIC-HCMC-26.git"
REPO_DIR = KAGGLE_TEMP / "AIC-HCMC-26"

if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "--branch", "dev", "--single-branch", REPO_URL, str(REPO_DIR)],
        check=True,
    )
elif (REPO_DIR / ".git").is_dir():
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", "dev"], check=True)
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "checkout", "-B", "dev", "origin/dev"],
        check=True,
    )
else:
    raise RuntimeError(f"{REPO_DIR} tồn tại nhưng không phải Git repository")

commit = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "log", "-1", "--oneline"], text=True
).strip()
print("Code:", commit)

# %%
KAGGLE_PACKAGES = [
    "pymilvus>=2.5.0,<3.0.0",
    "polars>=0.20.0",
    "pyarrow",
    "faiss-cpu>=1.8.0",
    "open_clip_torch>=2.24.0",
    "transformers>=4.45.0,<5.0.0",
    "accelerate>=0.34.0",
    "sentencepiece",
    "bitsandbytes>=0.46.1",
    "sentence-transformers>=3.0.0",
    "deep-translator>=1.11.4",
    "opencv-python-headless>=4.8.0",
    "Pillow>=10.0.0",
    "tqdm>=4.66.0",
]
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *KAGGLE_PACKAGES],
    check=True,
)
print("Dependencies ready.")

# %% [markdown]
# ## 2. Nạp Zilliz Secrets
#
# URI và token được lấy từ Kaggle Secrets, chỉ lưu trong biến môi trường của
# phiên chạy và không được in ra output.

# %%
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()


def first_secret(*names):
    for name in names:
        try:
            value = secrets.get_secret(name)
        except Exception:
            value = None
        if value:
            return value
    return None


os.environ["ZILLIZ_URI"] = first_secret("ZILLIZ_URI", "MILVUS_URI") or ""
os.environ["ZILLIZ_TOKEN"] = first_secret("ZILLIZ_TOKEN", "MILVUS_TOKEN") or ""
if not os.environ["ZILLIZ_URI"] or not os.environ["ZILLIZ_TOKEN"]:
    raise RuntimeError(
        "Thiếu Kaggle Secrets ZILLIZ_URI/ZILLIZ_TOKEN "
        "(hoặc MILVUS_URI/MILVUS_TOKEN). Hãy tạo secret rồi bật quyền notebook."
    )
print("Đã nạp URI và token từ Kaggle Secrets; giá trị không được hiển thị.")

# %% [markdown]
# ## 3. Tự tìm 30 query và khai báo bốn đường dẫn runtime
#
# Không cần sửa mount path của Kaggle. Cell tìm 30 file `*_kis.txt`, `*_qa.txt`,
# `*_trake.txt` trong input `sotuyen2` và dựng một thư mục query chuẩn bằng
# symlink.

# %%
RUNTIME_ROOT = KAGGLE_TEMP / "aic_sotuyen2_runtime"
OUTPUT_DIR = KAGGLE_WORKING / "aic_sotuyen2_output"
QUERY_DIR = RUNTIME_ROOT / "queries"
METADATA_PATH = OUTPUT_DIR / "metadata.parquet"
KEYFRAMES_DIR = RUNTIME_ROOT / "keyframes"
for directory in (RUNTIME_ROOT, QUERY_DIR, KEYFRAMES_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

query_name = re.compile(r"(?:^|[_-])(kis|qa|trake)\.txt$", re.IGNORECASE)
query_files = []
for current, directories, files in os.walk(KAGGLE_INPUT):
    if Path(current).name.lower() == "keyframes":
        directories.clear()
        continue
    for name in files:
        if query_name.search(name):
            query_files.append(Path(current) / name)

sotuyen2_files = [path for path in query_files if "sotuyen2" in str(path).lower()]
if len(sotuyen2_files) == 30:
    query_files = sotuyen2_files
if len(query_files) != 30:
    raise RuntimeError(
        f"Phải tìm thấy đúng 30 query, hiện có {len(query_files)}. "
        "Kiểm tra input hoanghieu26/sotuyen2 đã được attach."
    )
if len({path.name for path in query_files}) != 30:
    raise RuntimeError("Tên query bị trùng giữa các Kaggle input")
for source in sorted(query_files):
    target = QUERY_DIR / source.name
    if target.is_symlink():
        if target.resolve() == source.resolve():
            continue
        target.unlink()
    elif target.exists():
        raise RuntimeError(f"Không ghi đè file query runtime: {target}")
    target.symlink_to(source)

sys.path.insert(0, str(REPO_DIR))
from src.submission.query_parser import load_query_specs

specs = load_query_specs(queries_dir=QUERY_DIR)
type_counts = {
    kind: sum(spec.query_type == kind for spec in specs)
    for kind in ("kis", "qa", "trake")
}
if len(specs) != 30:
    raise RuntimeError(f"Parser chỉ đọc được {len(specs)}/30 query")

print("QUERY_DIR     =", QUERY_DIR)
print("METADATA_PATH =", METADATA_PATH)
print("KEYFRAMES_DIR =", KEYFRAMES_DIR)
print("OUTPUT_DIR    =", OUTPUT_DIR)
print("Query pack:", type_counts)

# %% [markdown]
# ## 4. Xuất canonical metadata từ Zilliz
#
# Chỉ lấy `pk`, `video_id`, `frame_name`, `frame_idx`, `timestamp` của
# collection visual. Field `embedding` không nằm trong truy vấn, vì vậy không
# tải vector xuống Kaggle.

# %%
subprocess.run(
    [
        sys.executable,
        str(REPO_DIR / "scripts" / "export_zilliz_visual_metadata.py"),
        "--output",
        str(METADATA_PATH),
    ],
    cwd=REPO_DIR,
    check=True,
)

# %% [markdown]
# ## 5. Ghép ba nguồn keyframe bằng symlink và kiểm tra toàn bộ coverage
#
# Cell quét mọi layout `Lxx_Vxxx/keyframes`, chọn nguồn chứa đúng `frame_name`
# trong metadata và chỉ tạo link. Nếu một video bị chia giữa hai input, cell tạo
# overlay bằng symlink từng file cho riêng video đó. Thiếu dù chỉ một tên ảnh
# canonical thì cell dừng.

# %%
KEYFRAME_MANIFEST = OUTPUT_DIR / "keyframe_sources.json"
subprocess.run(
    [
        sys.executable,
        str(REPO_DIR / "scripts" / "prepare_kaggle_keyframes.py"),
        "--input-root",
        str(KAGGLE_INPUT),
        "--metadata-path",
        str(METADATA_PATH),
        "--output-root",
        str(KEYFRAMES_DIR),
        "--manifest",
        str(KEYFRAME_MANIFEST),
    ],
    cwd=REPO_DIR,
    check=True,
)

# %% [markdown]
# ## 6. Preflight — phải hiện `Setup validation: PASSED`
#
# Bước này chưa tải MobileCLIP hay Qwen. Nó kiểm tra ba collection, search visual
# + subtitle, object schema, canonical mapping, keyframe mẫu và cấu trúc 30
# query. Có thể chạy khi chưa bật GPU.

# %%
os.environ["AIC_RETRIEVAL_BACKEND"] = "zilliz"
os.environ["AIC_METADATA_PATH"] = str(METADATA_PATH)
os.environ["AIC_KEYFRAMES_DIR"] = str(KEYFRAMES_DIR)
os.environ["HF_HOME"] = str(RUNTIME_ROOT / "hf-cache")
os.environ["TORCH_HOME"] = str(RUNTIME_ROOT / "torch-cache")

subprocess.run(
    [
        sys.executable,
        str(REPO_DIR / "scripts" / "validate_zilliz_setup.py"),
        "--metadata-path",
        str(METADATA_PATH),
        "--keyframes-dir",
        str(KEYFRAMES_DIR),
        "--queries-dir",
        str(QUERY_DIR),
    ],
    cwd=REPO_DIR,
    check=True,
)

# %% [markdown]
# ## 7. Chọn chế độ GPU
#
# Giữ `VLM_MODE_OVERRIDE = None` để tự chọn: A100 → `bf16`; T4/L4 → `4bit`.
# Muốn ép thủ công thì đặt thành `'4bit'` hoặc `'bf16'`.

# %%
import torch

VLM_MODE_OVERRIDE = None  # None | "4bit" | "bf16"
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    AUTO_VLM_MODE = "bf16" if "A100" in GPU_NAME.upper() else "4bit"
    VLM_MODE = VLM_MODE_OVERRIDE or AUTO_VLM_MODE
    if VLM_MODE == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{GPU_NAME} không hỗ trợ BF16; hãy dùng 4bit")
    print("GPU:", GPU_NAME)
    print("VLM mode:", VLM_MODE)
else:
    GPU_NAME = None
    VLM_MODE = VLM_MODE_OVERRIDE or "4bit"
    print("Chưa có GPU. Setup/preflight vẫn hoàn tất; hãy bật GPU trước cell chạy query.")

# %% [markdown]
# ## 8. Chạy toàn bộ 30 query
#
# Cell đầu tiên tải MobileCLIP và Qwen2-VL nên sẽ lâu hơn. `--resume` bỏ qua
# những query đã có checkpoint trong cùng phiên Kaggle. Kết quả cuối nằm trong
# `/kaggle/working/aic_sotuyen2_output`; chọn **Save Version** để lưu output của
# notebook.

# %%
if not torch.cuda.is_available():
    raise RuntimeError(
        "Hãy bật Accelerator GPU trong Kaggle Settings rồi chạy lại cell GPU và cell này"
    )

RESULT_ZIP = OUTPUT_DIR / "SOTUYEN2_submission.zip"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PROVENANCE_DIR = OUTPUT_DIR / "provenance"
run_command = [
    sys.executable,
    str(REPO_DIR / "scripts" / "generate_submission.py"),
    "--queries-dir",
    str(QUERY_DIR),
    "--output",
    str(RESULT_ZIP),
    "--retrieval-backend",
    "zilliz",
    "--metadata-path",
    str(METADATA_PATH),
    "--keyframes-dir",
    str(KEYFRAMES_DIR),
    "--checkpoint-dir",
    str(CHECKPOINT_DIR),
    "--provenance-dir",
    str(PROVENANCE_DIR),
    "--resume",
    "--device",
    "cuda",
    "--vlm-mode",
    VLM_MODE,
    "--max-rows",
    "100",
]
subprocess.run(run_command, cwd=REPO_DIR, check=True)
print("Hoàn tất:", RESULT_ZIP)

# %%
if not RESULT_ZIP.is_file():
    raise FileNotFoundError(RESULT_ZIP)
print("Submission ZIP:", RESULT_ZIP, f"({RESULT_ZIP.stat().st_size / 1024:.1f} KiB)")
print("Checkpoint CSVs:", len(list(CHECKPOINT_DIR.glob("*.csv"))))
print("Metadata mapping:", METADATA_PATH)
print("Keyframe source manifest:", KEYFRAME_MANIFEST)
