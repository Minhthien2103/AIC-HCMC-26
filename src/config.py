import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    """Return an optional environment path without resolving it at import time."""
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else default

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"

# ── Data paths (confirmed from actual directory structure) ────────────────────
VIDEOS_DIR        = DATA_DIR / "raw_videos"                 # Full videos
KEYFRAMES_DIR     = _env_path("AIC_KEYFRAMES_DIR", DATA_DIR / "keyframes")
CLIP_FEATURES_DIR = DATA_DIR / "clip-features-32"           # LXX_Vxxx.npy, one per video, [N, 512]
MAP_KEYFRAMES_DIR = DATA_DIR / "map-keyframes"              # LXX_Vxxx.csv (n, pts_time, fps, frame_idx)
OBJECTS_DIR       = DATA_DIR / "objects"                    # LXX_Vxxx/NNN.json
MEDIA_INFO_DIR    = DATA_DIR / "media-info"                 # LXX_Vxxx.json (TBC)

# ── Index paths ───────────────────────────────────────────────────────────────
FAISS_INDEX_PATH  = INDEX_DIR / "faiss_clip.index"
METADATA_PATH     = _env_path("AIC_METADATA_PATH", INDEX_DIR / "metadata.parquet")
OBJECTS_PATH      = INDEX_DIR / "objects.parquet"
MEDIA_INFO_PATH   = INDEX_DIR / "media_info.json"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Shot Segmentation Constants
SHOT_MIN_LENGTH = 10            # Minimum frames for a valid shot
OPTICAL_FLOW_THRESHOLD = 5.0    # Threshold for global motion filter to suppress shaky cuts

# Embedding Models Config
EMBEDDING_BATCH_SIZE = 32

# ── CLIP model (MUST match the model used to create the .npy features) ────────
# OpenCLIP's ``openai`` checkpoint was trained with QuickGELU.  The plain
# ViT-B-32 architecture defaults to GELU and emits a mismatch warning; use the
# architecture alias that preserves the checkpoint's embedding space.
CLIP_MODEL_NAME = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_DIM = 512

# The two production Zilliz collections were generated with this exact
# OpenCLIP model/checkpoint pair. A query encoded with the legacy ViT-B/32
# checkpoint is also 512-dimensional, but belongs to a different embedding
# space and therefore produces meaningless neighbours.
ZILLIZ_CLIP_MODEL_NAME = os.getenv("AIC_ZILLIZ_MODEL", "MobileCLIP-S2")
ZILLIZ_CLIP_PRETRAINED = os.getenv("AIC_ZILLIZ_PRETRAINED", "datacompdr")
ZILLIZ_CLIP_DIM = 512

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 100
OBJECT_CONF_THRESH = 0.3
CLIP_WEIGHT = 0.7
OBJECT_BOOST_WEIGHT = 0.3

# ── KIS GPU retrieval / re-ranking ─────────────────────────────────────────
# These values are intentionally configurable rather than hidden inside the
# task implementation, so trial submissions can be reproduced and ablated.
KIS_CANDIDATES_PER_VARIANT = 1000
KIS_RRF_K = 60

# Keep the dual-retrieval settings used by the local KIS pipeline available
# alongside the dev-branch defaults.

# ── KIS dual-retrieval assets ──────────────────────────────────────────────
# All values are exposed as CLI defaults rather than semantic query rules.
VITH_MODEL_NAME = "ViT-H-14"
VITH_PRETRAINED = "laion2b_s32b_b79k"
VITH_INDEX_PATH = INDEX_DIR / "faiss_vith.index"
VITH_FEATURES_PATH = INDEX_DIR / "vith_features.f16.npy"
MEDIA_TEXT_INDEX_PATH = INDEX_DIR / "media_e5.index"
MEDIA_TEXT_RECORDS_PATH = INDEX_DIR / "media_e5_records.json"
MEDIA_INFO_ARCHIVE_URL = "https://aic-data.ledo.io.vn/media-info-aic25-b1.zip"
E5_MODEL_NAME = "intfloat/multilingual-e5-base"

KIS_DUAL_CANDIDATE_BUDGET = 1000
KIS_OCR_CANDIDATE_BUDGET = 384
KIS_TEXT_FRAMES_PER_VIDEO = 48  # compatibility alias for local frame budget
KIS_REVIEW_TOP_K = 20
KIS_QWEN_RERANK_TOP_K = 192
KIS_VIDEO_BUDGET = 24
KIS_LOCAL_FRAME_BUDGET = 48
KIS_QUERY_VARIANT_LIMIT = 4
FRAME_NEIGHBORHOOD_COUNT = 7
KIS_VLM_TOP_K = 100
KIS_VLM_PROMOTE_MIN_SCORE = 2

# ── Temporal search ───────────────────────────────────────────────────────────
TEMPORAL_WINDOW = 20
TEMPORAL_KEYWORDS = [
    "before", "after", "then", "while", "during", "next", "followed by",
    "truoc", "sau", "tiep theo", "trong khi", "sau do", "roi",
    "trước", "tiếp theo", "sau đó", "rồi",
]

# VQA Pipeline Config (Improvements #3, #5, #7, #8, #9, #11, #12)
VQA_CANDIDATES = 100                # FAISS candidates per query before RRF
VQA_RERANK_K = 15                   # Top-K candidates passed to VLM for answer
VQA_PARAPHRASE_N = 5               # Number of paraphrases for RRF
VQA_VERIFICATION_BOOST = 0.15      # Score boost for VLM-verified frames (Improvement #5)
VQA_RRF_K = 60                     # RRF constant k
SEMANTIC_OBJ_THRESHOLD = 0.55      # MiniLM cosine similarity threshold (Improvement #9)
SEMANTIC_OBJ_TOP_K = 3             # Top-K OpenImages labels per noun phrase
VQA_MAX_ANSWER_CHARS = 100
VQA_ENABLE_EXTERNAL_SEARCH = False
VQA_MAX_CANDIDATES = 100
VQA_VERIFY_CANDIDATES = True
VQA_VIDEO_BUDGET = 12
VQA_LOCAL_FRAME_BUDGET = 48
VQA_QWEN_CANDIDATE_BUDGET = 24

# TRAKE is evaluated as a sequence, but only explicit temporal relations in
# the query are constraints.  Events otherwise may occur at arbitrary frame
# positions in a video.
TRAKE_EVENT_TOP_K = 16
TRAKE_QWEN_PER_EVENT = 6


def keyframe_path(video_id: str, keyframe_name: str) -> Path:
    """Resolve a keyframe against the current checkout, never stale Parquet paths."""
    return KEYFRAMES_DIR / str(video_id) / f"{str(keyframe_name).removesuffix('.jpg')}.jpg"

# TRAKE
TRAKE_VLM_TOP_K = 30
TRAKE_VLM_PROMOTE_MIN_SCORE = 2

# ── Production retrieval / Zilliz Cloud ─────────────────────────────────
# Credentials intentionally have no in-repository fallback. In Colab, read
# them from Colab Secrets and expose them as environment variables.
RETRIEVAL_BACKEND = os.getenv("AIC_RETRIEVAL_BACKEND", "zilliz").strip().lower()
ZILLIZ_URI = os.getenv("ZILLIZ_URI", "").strip()
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN", "").strip()
ZILLIZ_VISUAL_COLLECTION = os.getenv(
    "ZILLIZ_VISUAL_COLLECTION", "aic_visual_mobileclip_s2_v1"
)
ZILLIZ_TEXT_COLLECTION = os.getenv(
    "ZILLIZ_TEXT_COLLECTION", "aic_text_mobileclip_s2_v1"
)
ZILLIZ_OBJECT_COLLECTION = os.getenv(
    "ZILLIZ_OBJECT_COLLECTION", "aic_object_detection_v1"
)
ZILLIZ_VECTOR_FIELD = os.getenv("ZILLIZ_VECTOR_FIELD", "embedding")
ZILLIZ_VISUAL_PK_FIELD = os.getenv("ZILLIZ_VISUAL_PK_FIELD", "pk")
ZILLIZ_TEXT_PK_FIELD = os.getenv("ZILLIZ_TEXT_PK_FIELD", "pk")
ZILLIZ_OBJECT_PK_FIELD = os.getenv("ZILLIZ_OBJECT_PK_FIELD", "pk")
ZILLIZ_METRIC_TYPE = os.getenv("ZILLIZ_METRIC_TYPE", "COSINE").upper()

# Legacy ReCap collection name; it is unrelated to the three production
# retrieval collections above.
MILVUS_COLLECTION = "aic_captions"

# ── RAG (Text-based semantic search) ────────────────────────────────────  
SUPABASE_URL   = "https://your-project.supabase.co"
SUPABASE_KEY   = "your-anon-key"  # For vector similarity search
SUPABASE_TOKEN = "your-service-role-key"  # For FTS queries (NOT needed if using stored tsvector)

# The table we created in init_db.py
RAG_TABLE_NAME = "aic_captions"
SUPABASE_KEY = "your-supabase-anon-key"
