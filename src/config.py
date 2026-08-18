from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"

# ── Data paths (confirmed from actual directory structure) ────────────────────
VIDEOS_DIR        = DATA_DIR / "raw_videos"                 # Full videos
KEYFRAMES_DIR     = DATA_DIR / "keyframes"                  # LXX_Vxxx/NNN.jpg
CLIP_FEATURES_DIR = DATA_DIR / "clip-features-32"           # LXX_Vxxx.npy, one per video, [N, 512]
MAP_KEYFRAMES_DIR = DATA_DIR / "map-keyframes"              # LXX_Vxxx.csv (n, pts_time, fps, frame_idx)
OBJECTS_DIR       = DATA_DIR / "objects"                    # LXX_Vxxx/NNN.json
MEDIA_INFO_DIR    = DATA_DIR / "media-info"                 # LXX_Vxxx.json (TBC)

# ── Index paths ───────────────────────────────────────────────────────────────
FAISS_INDEX_PATH  = INDEX_DIR / "faiss_clip.index"
METADATA_PATH     = INDEX_DIR / "metadata.parquet"
OBJECTS_PATH      = INDEX_DIR / "objects.parquet"
MEDIA_INFO_PATH   = INDEX_DIR / "media_info.json"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Shot Segmentation Constants
SHOT_MIN_LENGTH = 10            # Minimum frames for a valid shot
OPTICAL_FLOW_THRESHOLD = 5.0    # Threshold for global motion filter to suppress shaky cuts

# Embedding Models Config
EMBEDDING_BATCH_SIZE = 32

# ── CLIP model (MUST match the model used to create the .npy features) ────────
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_DIM = 512

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 100
OBJECT_CONF_THRESH = 0.3
CLIP_WEIGHT = 0.7
OBJECT_BOOST_WEIGHT = 0.3

# ── Temporal search ───────────────────────────────────────────────────────────
TEMPORAL_WINDOW = 20
TEMPORAL_KEYWORDS = [
    "before", "after", "then", "while", "during", "next", "followed by",
    "truoc", "sau", "tiep theo", "trong khi", "sau do", "roi",
    "trước", "tiếp theo", "sau đó", "rồi",
]

# VQA Pipeline Config (Improvements #3, #5, #7, #8, #9, #11, #12)
VQA_CANDIDATES = 30                # FAISS candidates per query before RRF
VQA_RERANK_K = 5                   # Top-K candidates passed to VLM for answer
VQA_PARAPHRASE_N = 2               # Number of paraphrases for RRF
VQA_VERIFICATION_BOOST = 0.15      # Score boost for VLM-verified frames (Improvement #5)
VQA_RRF_K = 60                     # RRF constant k
SEMANTIC_OBJ_THRESHOLD = 0.55      # MiniLM cosine similarity threshold (Improvement #9)
SEMANTIC_OBJ_TOP_K = 3             # Top-K OpenImages labels per noun phrase
VQA_MAX_ANSWER_CHARS = 100
VQA_ENABLE_EXTERNAL_SEARCH = False
VQA_MAX_CANDIDATES = 100
VQA_VERIFY_CANDIDATES = True


def keyframe_path(video_id: str, keyframe_name: str) -> Path:
    """Resolve a keyframe against the current checkout, never stale Parquet paths."""
    return KEYFRAMES_DIR / str(video_id) / f"{str(keyframe_name).removesuffix('.jpg')}.jpg"
