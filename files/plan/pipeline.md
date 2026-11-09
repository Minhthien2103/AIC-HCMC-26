# Full Pipeline — AIC-HCMC 2026 Retrieval System

This document describes the entire system from raw data to a final submission string,
using the same two-phase structure: **Offline** (run once) and **Online** (every query).

---

## The Big Picture

The system has one fundamental job: given a Vietnamese text description of a video
moment, find the correct keyframe from a database of 177,321 frames across 873 videos.

It solves this by combining three signals:
1. **Semantic similarity** — does the image *look like* what the text describes? (CLIP)
2. **Object presence** — does the image actually *contain* the objects mentioned? (Object Filter)
3. **Temporal ordering** — for multi-event queries, do events happen in the *right order*? (Temporal Search)

---

## Phase 1: Offline Ingestion (Run Once)

This phase happens before any user searches. It transforms raw data files into fast, searchable indexes.

---

### 1.1 — Map-Keyframes CSV → Master Metadata Table
**File:** `src/offline_pipeline/metadata_loader.py`
**Output:** `indexes/metadata.parquet`

The organizers provided one `.csv` file per video (873 total). Each CSV maps every
keyframe to its position in the original video file.

For every row in every CSV, we store:

| Column | Meaning |
|---|---|
| `video_id` | e.g. `L29_V003` |
| `keyframe_name` | e.g. `299` (the filename) |
| `frame_id` | The **original MP4 frame index** — this is what you submit to the competition |
| `pts_time` | The timestamp in seconds |
| `faiss_idx` | A global counter (0, 1, 2 … 177320) that links each row to a row in the FAISS index |
| `keyframe_path` | Absolute path to the JPEG file on disk |

All 873 CSVs are concatenated into a single Parquet file. This is the central registry
that every other component reads from.

> **Critical:** `frame_id` (from `frame_idx` in the CSV) is **not** the keyframe filename.
> Keyframe `299.jpg` maps to `frame_id = 17932` in the original MP4. You always submit `17932`, never `299`.

---

### 1.2 — CLIP Feature Files (.npy) → FAISS Index
**File:** `src/offline_pipeline/build_index.py`
**Output:** `indexes/faiss_clip.index`

The organizers also pre-computed a CLIP visual embedding for every keyframe and saved
them as `.npy` files (one per video). Each embedding is a 512-dimensional float32 vector
that encodes *what the image looks like* in CLIP's semantic space.

We load all 873 `.npy` files, concatenate them into one giant matrix of shape
`[177321, 512]`, L2-normalize every row, and hand the whole matrix to FAISS.

FAISS builds an `IndexFlatIP` (Inner Product) index — a structure that can find the
closest vectors to any query in milliseconds, even across 177,000 entries.

**The key constraint:** These vectors were computed by the organizers using `ViT-B-32`.
We cannot change them. The quality ceiling of every CLIP-based search is set by whatever
`ViT-B-32` understood about each frame when it encoded it.

---

### 1.3 — Object JSON Files → Object Store
**File:** `src/offline_pipeline/objects_loader.py`
**Output:** `indexes/objects.parquet`

For every keyframe, the organizers ran an OpenImages object detector and saved the
results as `.json` files. Each file lists up to 100 detected objects with their confidence
scores and bounding boxes.

We parse all of these JSONs and flatten them into a single Parquet table:
one row per detected object instance, with columns `video_id`, `keyframe_name`,
`faiss_idx`, `object_label`, and `confidence`.

We discard any detection with `confidence < 0.3` to remove noise.
This leaves 1,122,384 clean object detections across all available videos.

**The key constraint:** The object labels come from the OpenImages vocabulary (~600 nouns).
Colors and adjectives are not object classes. `"Red shirt"` does not exist in this database —
only `"Shirt"` does.

---

## Phase 2: Online Query Pipeline (Every Search)

This phase runs every time the user presses Search. It processes the query and ranks the 177,321 frames.

---

### 2.1 — Vietnamese → English Translation
**Component:** `deep_translator.GoogleTranslator` (inside each task class)

CLIP's text encoder was trained almost entirely on English text. Feeding it Vietnamese
produces garbage vectors. Every query is therefore translated to English first.

If translation fails (network error), we fall back to the original query text silently
and continue — a bad vector is still better than a crash.

---

### 2.2 — Text → CLIP Vector
**File:** `src/online_pipeline/query_encoder.py`

The translated English query is passed through the CLIP text encoder (`ViT-B-32`, `openai` weights).

This produces one 512-dimensional vector that represents *the meaning of the text*.

The key property: this vector lives in the **same semantic space** as the image vectors in the FAISS index. A text about a person speaking outdoors will naturally be close to image vectors of frames showing a person speaking outdoors.

---

### 2.3 — Temporal Detection (Branch)
**File:** `src/online_pipeline/temporal_search.py`

Before searching, the system checks if the query contains temporal keywords such as
`"then"`, `"after"`, `"sau đó"`, `"tiếp theo"`, etc.

**If NO temporal keywords:** proceed directly to FAISS search (Step 2.4).

**If temporal keywords found:** split the query into sub-events (e.g. `"A then B"` → `["A", "B"]`).
Encode each sub-event independently. Retrieve top-K candidates for each.
Then find videos where sub-event A appears in an earlier keyframe than sub-event B.
Score these valid chronological chains highly. Fall back to standard FAISS search if no chains are found.

---

### 2.4 — FAISS Search → Top-K Candidates
**File:** `src/online_pipeline/retrieval.py`

The query vector is compared against all 177,321 image vectors in the FAISS index using
inner product (which equals cosine similarity because all vectors are L2-normalized).

The top-K most similar frames are returned as a list of dicts, each containing the full
metadata row (video_id, frame_id, score, keyframe_path, etc.) joined from `metadata.parquet`.

At this point, the score is a pure cosine similarity: **0.0 = completely unrelated, 1.0 = perfect match.**

---

### 2.5 — Object Filter / Boost
**File:** `src/online_pipeline/object_filter.py`

The system scans the translated query for words that match known OpenImages labels
using a Regex with simple plural support (e.g. `"trees"` → matches label `"Tree"`).

If matches are found (e.g. `["Shirt", "Tree"]`), the system looks up each candidate
frame in `objects.parquet`. Any frame that contains at least one of the required objects
(at confidence ≥ 0.3) receives a flat bonus of `+0.3` to its score.

**Current formula:**
```
final_score = clip_cosine_score + 0.3  (if ≥ 1 required object found in frame)
final_score = clip_cosine_score + 0.0  (if no required objects found)
```

The results are re-sorted by `final_score` descending and returned.

**Known limitation:** The boost is a flat constant. A frame with 5 matching objects
scores the same as a frame with 1. Proportional scoring is a planned tuning item.

---

### 2.6 — Results Displayed in UI
**File:** `app.py`

The top-K frames are displayed in a 3-column grid. For each frame:
- The keyframe JPEG is shown (path rebuilt dynamically from `config.KEYFRAMES_DIR` to survive project folder renames)
- The `video_id` and `frame_id` are shown
- The final score is shown
- The matched objects (if any) are listed
- A submission string (`L29_V003, 17932`) is shown in a copyable code block

---

## Task-Specific Heads

### KIS-T (Known Item Search — Textual)
**Input:** A Vietnamese text description.
**Pipeline:** Translate → Encode → Temporal Check → FAISS → Object Boost → Display top-K.
**Output:** `video_id, frame_id` for each result.

### VQA (Visual Question Answering)
**Input:** A question about a video moment.
**Pipeline:** Translate → Encode → FAISS (top-5) → Object Boost → for each candidate, pass the keyframe JPEG + question to `Qwen2-VL-7B` → return the VLM answer.
**Output:** `video_id, frame_id, answer` for each candidate.

### TRAKE (Temporal Retrieval of Anchoring Keyframes)
**Phase A — Find the video:** Encode the overall video description → FAISS top-500 → group by `video_id` → take the max frame score per video → rank videos → pick the top candidates.
**Phase B — Find each event:** For each selected video, encode each event description → search only within that video's keyframes → pick the best frame per event.
**Temporal constraint:** Verify `frame_id_1 < frame_id_2 < ... < frame_id_N`.
**Output:** `video_id, frame_id_1, frame_id_2, ...`

---

## Data Flow Diagram

```
[User Query (Vietnamese)]
        │
        ▼
[GoogleTranslator → English]
        │
        ▼
[CLIP Text Encoder → 512-dim vector]
        │
        ├──[Temporal keywords?]──YES──► [Split → encode each event → find chronological chains]
        │                                                    │
        NO                                                   │
        │                                                    │
        ▼                                                    ▼
[FAISS IndexFlatIP Search → top-K candidates] ◄─────────────┘
        │
        ▼
[Object Filter: Regex → labels → lookup objects.parquet → +0.3 boost]
        │
        ▼
[Re-sort by final_score]
        │
        ▼
[Streamlit UI: images + submission strings]
```

---

## Key Files Reference

| File | Role |
|---|---|
| `src/config.py` | All paths and tunable constants |
| `src/offline_pipeline/metadata_loader.py` | CSV → `metadata.parquet` |
| `src/offline_pipeline/build_index.py` | `.npy` files → `faiss_clip.index` |
| `src/offline_pipeline/objects_loader.py` | JSON files → `objects.parquet` |
| `src/online_pipeline/query_encoder.py` | Text → CLIP vector |
| `src/online_pipeline/retrieval.py` | FAISS search + metadata join |
| `src/online_pipeline/object_filter.py` | Object presence scoring |
| `src/online_pipeline/temporal_search.py` | Temporal keyword detection + chronological ranking |
| `src/online_pipeline/vlm_pipeline.py` | Qwen2-VL-7B inference (VQA + TRAKE event extraction) |
| `src/tasks/kis_t.py` | KIS-T task orchestrator |
| `src/tasks/vqa.py` | VQA task orchestrator |
| `src/tasks/trake.py` | TRAKE task orchestrator |
| `main_ingest.py` | Entry point for offline ingestion |
| `app.py` | Streamlit UI |
