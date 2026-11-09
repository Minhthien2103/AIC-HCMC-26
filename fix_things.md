# Fix Things — Testing Issues Log

This file tracks issues found during manual testing. Each issue has a description,
root cause, and proposed fix to be implemented in the tuning phase.

---

## Issue #1: Wrong Frame Ranked — Color Attribute Not Understood

**Query:** `"Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh."`

**Expected:** Frames of the woman in a **red shirt** (e.g. `L29_V003/300.jpg`, `301.jpg`)

**Actual:** System correctly found video `L29_V003` but ranked frames of a **white-shirt man with a microphone** higher (e.g. `299.jpg`)

**Root Cause:**
- `ViT-B-32` (the provided CLIP model) suffers from weak color/attribute binding. It sees "shirt + person" but cannot distinguish "red shirt" from "white shirt".
- The translated phrase "speaker giving speech" strongly matches the man-with-mic frames, overriding the color signal.

**Proposed Fix:**
- Add a **VLM reranker** step after CLIP retrieval. After getting the top-N frames, send each to `Qwen2-VL` with a yes/no question: *"Does this image show a person wearing a red shirt?"* Boost frames that answer yes.
- Long-term: Replace `ViT-B-32` features with `SigLIP` or `InternVL` embeddings which are much stronger at attribute binding.

---

## Issue #2: Object Regex Cannot Extract Color-Modified Nouns

**Query:** `"...mặc áo đỏ..."` → translated: `"...wearing a red shirt..."`

**Expected:** Extract `"red shirt"` as the filter object

**Actual:** Extracts only `"Shirt"` (the base OpenImages class). `"Red"` is ignored because it is not an object class in the OpenImages vocabulary.

**Root Cause:**
- Our Regex simply matches query words against the OpenImages label list. "Red shirt" is not an OpenImages label — only "Shirt" is.
- Even if we extracted "red shirt", the `objects.parquet` database has no "red shirt" entries because the object detector was never trained to detect colors.

**Proposed Fix:**
- This is an inherent limitation of the OpenImages-based detector. The fix belongs at the **retrieval** level (VLM reranking), not the object filter level. Object filter should remain focused on base nouns.
- No change needed to the Regex itself.

---


## Improvement #3: Dynamic Object Confidence Threshold

**Decision:** Move from hard ingestion-time filter to late filter in `object_filter.py` at query time.

**Current behavior:** Objects with `confidence < 0.3` are discarded permanently during `main_ingest.py`. The threshold is hard-coded and cannot be adjusted without re-ingesting.

**Planned behavior:** Store ALL detections (confidence >= 0.0) in `objects.parquet`. At query time, `object_filter.py` applies a configurable threshold that can be tuned per-query context (e.g. stricter for common objects like "person", looser for rare objects where even a weak detection is meaningful).

**Implementation notes:**
- Remove the `confidence < 0.3` filter from `src/offline_pipeline/objects_loader.py`.
- Re-run `main_ingest.py` once to rebuild `objects.parquet` with full detections.
- Add a `conf_threshold` parameter to `ObjectFilter.filter_candidates()`, defaulting to `config.OBJECT_CONF_THRESH` (0.3) for backward compatibility.
- Parquet size will increase but remains manageable (~3-4x current size).

---

## Improvement #4: YOLO Re-Detection Pass (Offline)

**Decision:** Run one YOLO inference pass over all keyframes and MERGE results into `objects.parquet` alongside the provided OpenImages detections.

**Rationale:** The provided OpenImages detector has 600 classes but is weaker on fine-grained attributes. YOLO (COCO-80) is faster and stronger on common objects (person, car, chair, etc.) with better bounding box precision. The two vocabularies overlap partially but are complementary.

**Vocabulary mapping required (COCO -> OpenImages):**
Key overlapping classes that need deduplication:
- `"person"` -> `"Person"`
- `"car"` -> `"Car"`
- `"chair"` -> `"Chair"`
- `"bottle"` -> `"Bottle"`
- `"laptop"` -> `"Laptop"`
(Full mapping table to be created in `src/offline_pipeline/yolo_vocab_map.py`)

**Merge strategy:**
- Add a `source` column to `objects.parquet`: `"openimages"` or `"yolo"`.
- For overlapping classes (same label, same frame), keep BOTH rows. At query time, a frame gets the object boost if it appears in either source.
- Do NOT replace the provided detections - they cover 600 classes which YOLO cannot match.

**Implementation notes:**
- New script: `src/offline_pipeline/yolo_detector.py` using `ultralytics` (`pip install ultralytics`).
- Recommended model: `yolov8m.pt` or `yolov11m.pt` (good accuracy/speed balance).
- Run on GPU (RTX 4060) - ~873 videos x ~200 keyframes = ~175k images. At ~5ms/image, total ~15 minutes.
- Output: appended rows to `objects.parquet` with `source="yolo"`.

---

## Improvement #5: VLM Reranker (Top-5 to Top-10 Only)

**Decision:** After CLIP retrieval returns top-K candidates, pass ONLY the top 5-10 to `Qwen2-VL` for visual verification. Do NOT run VLM on the full top-100.

**Rationale:** Running Qwen2-VL-7B on 100 frames would take ~5-10 minutes per query, which is unusable in a competition. On 5-10 frames, it takes ~10-30 seconds — acceptable for a high-stakes query.

**How it works:**
1. CLIP + Object Filter returns top-100, sorted by score.
2. Take top `rerank_k` (default: 5) frames.
3. For each frame, ask Qwen2-VL a targeted yes/no verification question auto-generated from the query (e.g. "Is there a person wearing a red shirt in this image? Answer yes or no.").
4. Frames that answer "yes" get a score boost and rise to the top.
5. The remaining top-100 are returned unchanged below them.

**Implementation notes:**
- New method: `VLMPipeline.verify(image_path, question) -> bool` (yes/no only, max 5 tokens).
- New method: `KIStask.rerank(candidates, query, rerank_k=5)` that calls `verify` for each.
- Add `rerank_k` slider to the Streamlit sidebar (0 = disabled, 5 = default).
- Setting `rerank_k=0` completely skips VLM and keeps current behavior for fast queries.
- This directly addresses Issue #1 (color attribute not understood by CLIP).

---

## Improvement #6: Region-Level CLIP Embeddings for Top-K Reranking

**Decision:** Implement as an online (query-time) operation on top-K frames only. No offline precomputation needed.

**The idea:** Frame-level CLIP embeddings encode the entire scene, diluting fine-grained signals like object color. Cropping the detected object regions and encoding them separately gives CLIP a much cleaner, more focused view.

**Revised architecture (solves Con 3 — no fusion problem):**
1. FAISS frame-level search returns top-100 candidates (as today).
2. For top-K frames (e.g. top-20): load the keyframe JPEG, crop each relevant object region using the bounding boxes already stored in `objects.parquet`, encode each crop with CLIP.
3. Compute `region_score = cosine_similarity(query_vector, crop_vector)`.
4. Merge: `final_score = frame_score + beta * region_score` for the top-K, re-sort.
5. Return re-ranked top-K at the top, followed by the remaining candidates.

**Why this is clean:**
- No offline precomputation needed -> zero extra storage cost.
- No second FAISS index -> no fusion hyperparameter problem.
- Bounding boxes already exist in `objects.parquet`.
- Query-time cost: ~20 frames x ~5 objects = 100 crops x ~5ms = ~0.5 seconds. Acceptable.

**Scope:** Apply only to object classes that carry discriminative attribute information:
- Priority classes: `"Person"`, `"Clothing"`, `"Shirt"`, `"Car"`, `"Vehicle"` (objects where color/style matters most for queries).
- Skip high-noise classes like `"Tree"`, `"Building"`, `"Sky"` where cropping adds little signal.

**Implementation notes:**
- New method: `QueryEncoder.encode_image_crop(image_path, bbox) -> np.ndarray` (bbox as [ymin, xmin, ymax, xmax] normalized 0-1, already stored in objects.parquet).
- New method: `RetrievalEngine.rerank_with_crops(candidates, query_vector, top_k, priority_classes) -> list[dict]`.
- Add `region_rerank_k` parameter (default: 20) alongside `rerank_k` for VLM.
- Weight `beta` to be tuned in Phase 3 (start with 0.2).

**Phase:** Phase 3 consideration. Implement after VLM reranker (Improvement #5) is verified working.

---

## Improvement #7: Query Expansion via Qwen2.5 Paraphrase (Phase 3)

**Decision:** Implement as Google Translate + Qwen2.5 paraphrase, average-pooled. Skip NLLB.

**The idea:** Instead of translating once, translate with Google then ask Qwen2.5 to rephrase the English output in a way more aligned with how CLIP was trained to describe scenes. Encode both versions and average their vectors before FAISS search.

**Why NOT full three-way ensemble (Google + NLLB + Qwen2.5):**
- NLLB adds a large model dependency (600MB-3.3B) and latency for marginal gain over Google Translate on well-formed Vietnamese queries.
- The weak link is CLIP attribute binding, not translation accuracy. This improvement matters more once upgraded to SigLIP/InternVL in Phase 3+.

**Recommended design:**
1. `google_translation = GoogleTranslator(vi -> en).translate(query)`
2. `paraphrase = Qwen2.5("Rephrase for a visual scene retrieval system: {google_translation}")`
3. `v1 = CLIP.encode_text(google_translation)`
4. `v2 = CLIP.encode_text(paraphrase)`
5. `query_vector = (v1 + v2) / 2` (average pool, NOT max pool)
6. Search FAISS with `query_vector` as normal.

**Why average, not max pool:**
- Max pooling takes the most extreme value per dimension across vectors, which can create an unnatural composite that does not correspond to any real semantic point in CLIP space.
- Average pooling stays within the convex hull of the original vectors, remaining semantically coherent.

**Implementation notes:**
- Make it optional via `config.USE_QUERY_EXPANSION = False` (off by default, toggleable in UI).
- Adds ~1-2 seconds per query for Qwen2.5 inference (text-only, fast).
- Qwen2.5 is already available locally via the VLM pipeline — no new model download needed.

**Phase:** Phase 3. Implement after VLM reranker and region crop reranking are verified.

---

## Improvement #7: Query Expansion & Translation Ensemble (Google + Qwen2.5 Paraphrase)

**Decision:** Implement a lightweight 2-way query ensemble: Google Translate baseline + 1 Qwen2.5 descriptive paraphrase, followed by vector average pooling. Skip local NLLB.

**Rationale:** 
- Adding NLLB introduces substantial model weight and latency without meaningful translation quality gain over Google Translate on standard Vietnamese.
- However, generating a descriptive rephrasing using Qwen2.5 (e.g., transforming `"speaker in red shirt"` into `"a person presenting on stage wearing a red top with outdoor background"`) broadens semantic coverage across CLIP's training distribution.

**Ensemble Architecture:**
1. Translate Vietnamese query -> English ($Q_1$) via Google Translate.
2. Generate one descriptive paraphrase ($Q_2$) via Qwen2.5 prompt: `"Paraphrase this video search query into a descriptive visual caption: {Q_1}"`.
3. Encode both: $v_1 = \text{CLIP}(Q_1)$, $v_2 = \text{CLIP}(Q_2)$.
4. Fuse embeddings using **Average Pooling** followed by L2-normalization:
   $$v_{\text{fused}} = \text{normalize}\left(\frac{v_1 + v_2}{2}\right)$$
   *(Note: Avoid max pooling as it distorts the unit hypersphere geometry).*

**Phase:** Phase 3 / advanced tuning. Recommended especially after upgrading to modern vision backbones (SigLIP/InternVL).

---

## Improvement #8: LLM Temporal Decomposition (Replacing Regex)

**Decision:** Replace the fragile Regex temporal keyword matcher with an LLM-based query analyzer. We will merge this mission with the Query Expansion task (Improvement #7) into a single Qwen 2.5 prompt to avoid double-latency.

**The Problem:** The current Regex only catches explicit keywords ("sau đó", "then"). A query like *"Người đàn ông bước vào phòng, cởi áo khoác, ngồi xuống ghế"* is highly sequential but fails the regex because it relies on commas.

**The Solution (Single-Pass JSON):**
Instead of two LLM calls, ask Qwen 2.5 to return a JSON containing both the paraphrase AND the sub-events:
```json
{
  "reasoning": "The query describes three distinct actions happening in order...",
  "paraphrase": "A man enters a room, takes off his coat, and sits on a chair.",
  "temporal_sub_events": [
    "A man enters a room",
    "A man takes off his coat",
    "A man sits on a chair"
  ]
}
```

**Handling the 7B Model Quality Trade-off:**
Asking a 7B model to do multiple tasks at once can degrade quality on complex queries. To guarantee high quality, we will use two strict Prompt Engineering techniques:
1. **Few-Shot Prompting:** The system prompt will include 3 hardcoded examples of complex queries and their perfect JSON outputs. This anchors the 7B model's behavior.
2. **Chain-of-Thought (CoT):** The model MUST generate a `"reasoning"` key first before generating the paraphrase or events. Forcing a 7B model to "think out loud" dramatically improves its logical accuracy.

If testing reveals the 7B model still struggles, the fallback is to split it into two parallel API calls (since local 7B inference is fast enough to handle parallel batching).

---

**Update to Improvement #8: Interleaved Chain-of-Thought JSON Schema**
Based on testing, the JSON schema will be structured with interleaved reasoning keys to maximize the 7B model's attention. Generating the reasoning for step A *immediately before* the output for step A prevents the model from losing context. 

Additionally, it will generate multiple paraphrases per sub-event to support robust RRF fusion:

```json
{
  "temporal_reasoning": "Câu này có 3 hành động: bước vào, cởi áo, ngồi xuống — xảy ra tuần tự.",
  "sub_events": [
    "a man enters the room", 
    "he takes off his jacket", 
    "he sits down on a chair"
  ],
  "paraphrase_reasoning": "Mỗi sub-event có thể diễn đạt theo nhiều cách (e.g. takes off vs removes) để tăng recall.",
  "paraphrases": {
    "event_0": ["a guy walking into a room", "male entering indoor space"],
    "event_1": ["he removes his coat", "taking off outer garment"],
    "event_2": ["he takes a seat", "man sitting down"]
  }
}
```
*(Note: Enforcing this exact key order in the Few-Shot examples is critical. If the model generates outputs before reasoning, the CoT effect is ruined).*

## Improvement #9: LLM Object Extraction & Dynamic Score Weighting

**Decision 1: Eliminate Regex for Object Extraction**
Since we are already passing every query to Qwen 2.5 (Improvement #8), we will add an `"objects_required"` key to the interleaved JSON schema. The LLM is vastly superior to Regex at identifying core physical nouns and can map them to the OpenImages/YOLO vocabulary intelligently.

*Updated LLM Schema addition:*
```json
  "object_reasoning": "Vật thể chính cần tìm là 'người đàn ông' (Person) và 'ghế' (Chair).",
  "objects_required": ["Person", "Chair"]
```

**Decision 2: Dynamic Object Score Weighting**
The proposed object boost formula is mathematically sound:
`object_boost = (matched_objects / required_objects) * avg_confidence`
`final_score = clip_cosine_score * (1 - w) + object_boost * w`

However, if a query is abstract (e.g., *"một khung cảnh yên bình vào buổi sáng sớm"*) and the LLM extracts `objects_required: []`:
- If we apply the formula, `object_boost` is 0.
- `final_score` becomes `clip_cosine_score * (1 - w)`, effectively penalizing the frame for a lack of objects when none were requested.

**Solution:** The weight `w` must act as a dynamic toggle:
- If `len(objects_required) == 0`: set `w = 0.0` -> `final_score = clip_cosine_score` (pure CLIP).
- If `len(objects_required) > 0`: set `w = 0.3` (configurable) and apply the interpolation formula.
This perfectly balances object-heavy queries and abstract scene queries without artificially suppressing scores.

---

**Update to Improvement #9: Semantic Similarity Mapping (Replacing LLM & Fuzzy Matching)**
Based on the risk of overloading the 7B model with a 3rd mission (which also requires strict adherence to a 600-word vocabulary), we will NOT use the LLM for object extraction.
Additionally, edit-distance (RapidFuzz) fails on synonyms (e.g., 'automobile' -> 'Car', 'puppy' -> 'Dog').

Instead, we will use a fast, deterministic semantic NLP approach:
1. Take the English baseline query (from Google Translate).
2. Run standard Noun-Phrase (NP) extraction using a fast library like spaCy (e.g., extracting "red shirt", "automobile", "puppy").
3. Encode each extracted noun-phrase AND the 600 OpenImages labels using a lightweight sentence embedding model (e.g., all-MiniLM-L6-v2). Note: The 600 labels are encoded ONCE and cached in memory.
4. Compute cosine similarity between the noun-phrase vector and the 600 label vectors -> take the highest scoring label.
5. Keep the top-3 candidates + their similarity scores (don't hardcode a threshold immediately, allow dynamic filtering based on score drops).

**Why this is superior:**
- **Zero Hallucination:** The LLM cannot accidentally invent classes that aren't in OpenImages.
- **Catches Synonyms:** all-MiniLM-L6-v2 maps synonyms to the same vector space, fixing the RapidFuzz weakness.
- **Low Latency:** NER + MiniLM embedding takes ~10-20 milliseconds on CPU.
- **Cognitive Offloading:** It frees the 7B LLM to focus 100% of its attention on Paraphrasing and Temporal Reasoning, ensuring high quality on those complex tasks.

## Improvement #10: TRAKE Phase A (Contextualized Per-Event Search)

**The Problem:** In TRAKE, Phase A currently encodes the entire video description as one massive string. This forces CLIP to try and find a single keyframe that contains *all* the events at once, which is impossible for sequential actions. 
Alternatively, searching raw sub-events (e.g., "giậm nhảy" / "jumps") loses the subject context, causing CLIP to return random jumping people from unrelated videos.

**Decision:** Implement **Per-Event Search with Balanced Global Context Injection**, using a variance-penalized scoring formula.

**How it works:**
1. Use Qwen 2.5 to extract the "Global Context" (the setting, the characters) and the "Sub-Events" (the actions).
2. **Context Balancing:** Before encoding, prepend the Global Context to every Sub-Event. To prevent the Global Context from dominating CLIP's attention mechanism, it must be kept extremely brief (e.g., 2-3 words like "Red shirt man:"). The Sub-Event action must account for the largest proportion of the final text.
   - *Balanced String:* "Red shirt man jumps"
3. Run FAISS search separately for each contextualized event to find the max score for each event within a given video.
4. **Variance-Penalized Scoring:** Instead of a simple sum or mean (which allows one super-matching event to carry a video where other events are missing), use a formula that rewards consistency across all events:
   $$Score_{video} = mean(scores) - \lambda \times std(scores)$$
   *(Note: $\lambda$ is a tunable hyperparameter, e.g., 0.5. A high standard deviation means the video has a great frame for Event 1 but a terrible frame for Event 2, so its score is heavily penalized).*
5. Take the top 5-10 videos with the highest penalized scores to pass into Phase B.

**Why this is superior:**
It perfectly aligns with the goal of finding complete sequences. It ensures CLIP remembers *who* is performing short actions, while the mathematical penalty rigorously filters out videos that only contain partial matches.

---

## Improvement #11: VQA — Query Split (Scene Retrieval vs VLM Question)

**Problem:** The current VQA sends the full question (e.g., "What is the man in the red shirt holding?") directly to FAISS. CLIP tries to match "holding" and "what" as visual concepts, degrading retrieval quality.

**Fix:** Use LLM to split the query into two parts before processing:
- **Retrieval part** → FAISS: "A man wearing a red shirt"
- **Question part** → VLM: "What is he holding in his hand?"

This means FAISS finds the right frame, and VLM answers the right question.

---

## Improvement #12: VQA — Knowledge-Grounded Query Rewriting (External Lookup)

**Scope: VQA only.**

**Problem:** Queries may contain implicit factual references that no keyframe contains verbatim.
- Simple: "tổng thống Mỹ" → no frame says "US president", but frames show "Donald Trump"
- Hard: "nước có GDP cao nhất" → no frame mentions this, but frames show the US flag

**Fix:** Add a fact-resolution step before FAISS search, VQA-only:
1. LLM (Qwen 2.5) checks: *"Does this query require external fact lookup?"*
2. If YES → call a 3rd-party search API (Google Search API or Wikipedia API) to resolve the fact.
3. Rewrite the query with the resolved fact: *"nước có GDP cao nhất"* → *"United States economy"*
4. Feed the rewritten query to normal FAISS + VLM pipeline.
5. If NO → proceed directly to paraphrasing (Improvement #8).

---
## Improvement #13: VQA - Answer Precision & Object Recognition (To Do)

**Problem:** The VLM (Qwen2-VL-7B) struggles to accurately identify the specific shapes of held objects in certain contexts. For example, when a presenter is holding a script book/card, the VLM often guesses generically (e.g., 'wallet', 'thing', 'book') due to low resolution or shape ambiguity.

**Fix:** We need to improve the answer quality. Potential solutions to explore later:
- **Few-shot prompting for object types:** Give the VLM examples of common broadcasting objects (script cards, microphones, clickers).
- **Cropping/Zooming:** Dynamically crop the image around the person's hands (using bounding boxes from YOLO/OpenImages) before passing the frame to the VLM to increase pixel density on the object.

---

# Implementation Status Tracking

- [x] **Improvement #8:** LLM Query Expansion (Implemented in vlm_pipeline.py)
- [x] **Improvement #9:** Semantic Similarity Mapping (Implemented via semantic_object_filter.py with all-MiniLM-L6-v2)
- [ ] **Improvement #10:** TRAKE Phase A - Contextualized Per-Event Search (Pending)
- [x] **Improvement #11:** VQA - Query Split - Scene Retrieval vs VLM Question (Implemented in analyze_vqa_query)
- [x] **Improvement #12:** VQA - Knowledge-Grounded Query Rewriting (Implemented via _external_search fact lookup)
- [ ] **Improvement #13:** VQA - Answer Precision & Object Recognition (Pending)


---

# Analysis: Real Competition Query Gaps (data/test/)

After reviewing all 24 real competition queries (14 KIS, 3 QA, 3 TRAKE + 4 unassigned), the following critical gaps were identified between our current pipeline and what the competition actually requires.

## Improvement #14: CLIP Token Limit on Long KIS Descriptions

**Problem:** CLIP's text encoder has a hard 77-token limit. Many KIS queries are 150-300+ characters long (e.g., `query-p1-12-kis` describing a multi-step pancake decorating scene). Any text beyond ~77 tokens is silently truncated, causing the retrieval to miss critical details from the end of the description.

**Fix:**
- Split long descriptions into individual sentences.
- Run a separate FAISS search for each sentence.
- Merge all result lists with RRF fusion.
- This is already partially handled by paraphrasing, but we do NOT currently split sentences from the original query.

**Priority: HIGH** — affects nearly all KIS queries with 2+ sentences.

---

## Improvement #15: QA / VQA - OCR-Mode for Text-Reading Queries

**Problem:** All 3 real QA queries (`p1-15`, `p1-19`, `p1-22`) require reading **on-screen text** from the frame:
- `p1-15`: Name of a village (xã) shown on a banner during a charity event.
- `p1-19`: Two poetry verses inscribed in a temple shrine.
- `p1-22`: Name of a dish shown on a recipe card with "200g thịt nạc xay".

Our current `answer_question` prompt says "clearly visible information" but does not explicitly instruct the VLM to **read and transcribe** text in the image. The VLM tends to describe the scene instead.

**Fix:** Detect OCR intent from the `vlm_question` (keywords: "tên là gì?", "câu thơ", "tiêu đề", "ghi gì", "đọc được", "nội dung"). When detected, switch to an OCR-mode prompt:
> "Read and transcribe EXACTLY any text visible in this image that is relevant to the question. Do not describe the image. Output only the text you can read."

**Priority: CRITICAL** — 100% of QA queries require this.

---

## Improvement #16: Knowledge-Grounded Retrieval for KIS Queries

**Problem:** Several KIS queries contain implicit factual references that must be resolved before FAISS search:
- `p1-23`: "loài động vật biển nguy hiểm... phim của Steven Spielberg năm 1975" → must resolve to **great white shark** (Jaws).
- `p1-21`: "Đại học ở thành phố Lausanne... cơ chế bay của bọ" → must resolve to **EPFL** and **insect biomechanics robotics**.
- `p1-8`: "lễ hội ẩm thực Nhật Bản lớn nhất thế giới" → need to resolve the festival's common name.

Our external search (Improvement #12) exists only for **VQA**. KIS queries go straight to FAISS with no fact resolution.

**Fix:** Apply the same `needs_fact_lookup` → external search → rewrite pipeline to KIS queries, not just VQA. Route through `analyze_vqa_query` or a dedicated `analyze_kis_query` method.

**Priority: HIGH** — affects at least 3 KIS queries.

---

## Improvement #17: Multi-Scene KIS — Identify the "Key Frame" Scene

**Problem:** Several KIS queries describe **multiple shots/scenes** in sequence but we only need ONE representative frame (the key scene):
- `p1-11`: Describes first a man arranging cardboard pieces, THEN the shadow art result on a wall. The key frame is likely the shadow portrait.
- `p1-14`: Describes first a sand sculpture of athletes, then multiple other sculptures, then colored smoke. There is one KEY scene we must target.
- `p1-12`: Multi-step cooking (place plate → add panna cotta → drizzle → add fruit). The key frame is likely the final plated dish.

Currently we search the entire description as one blob. We waste CLIP tokens on irrelevant transitional sentences.

**Fix:** In `analyze_vqa_query` (or a KIS-specific analyzer), add a `key_scene` field:
> "key_scene: The single most visually distinctive moment that would be easiest to find in one keyframe."
Use `key_scene` as the PRIMARY retrieval string, and the full description for secondary RRF queries.

**Priority: MEDIUM** — significantly improves KIS precision.

---

## Improvement #18: Camera Angle / Framing Awareness

**Problem:** Two cycling queries explicitly describe **camera angle**:
- `p1-24`: "góc quay trực diện từ trên cao xuống" (aerial straight-down shot, 3 cyclists in a line)
- `p1-25`: "góc quay từ flycam trên cao" (flycam aerial view, overtaking sequence)

CLIP does not inherently distinguish aerial vs. ground-level shots well. If we include "aerial view" or "drone shot" in the retrieval description it helps, but we are not currently doing this.

**Fix:** In the retrieval description generation, explicitly preserve camera angle descriptors: "aerial view", "top-down shot", "close-up", "wide shot". Add standard English equivalents in paraphrases.

**Priority: MEDIUM** — affects cycling/sports queries.

---

## Improvement #19: TRAKE - Precise Moment Detection (First-Contact Frames)

**Problem:** All 3 real TRAKE queries ask for the **exact first frame** of a very specific sub-event:
- `p1-4`: "Khoảnh khắc đầu tiên miếng mang tây đầu tiên tiếp xúc với dầu trong chảo" (first frame asparagus touches oil).
- `p1-16`: "Khoảnh khắc đầu tiên 4 chân hoàn toàn chạm đất" (first frame all 4 lion paws touch ground).
- `p1-18`: "Khoảnh khắc đầu tiên thấy cắt nấm" (first frame mushroom is cut).

Our current TRAKE just retrieves a keyframe near each event — it does not attempt to find the precise **transition frame** (e.g., the exact frame when asparagus first contacts oil vs. 5 frames before it).

**Fix:** For TRAKE, after finding the best candidate video, do a dense scan of neighboring keyframes around the best match and use a lightweight VLM verification pass to find the true "first contact" frame. This is related to Improvement #10 but focused on frame-level precision.

**Priority: HIGH** — current TRAKE results will likely be off by many frames.

---

## Summary: Priority Order for Remaining Work

| # | Improvement | Type | Priority |
|---|---|---|---|
| #15 | OCR-Mode for QA text-reading | VQA | CRITICAL |
| #19 | TRAKE precise moment detection | TRAKE | HIGH |
| #14 | CLIP 77-token limit - sentence splitting | KIS | HIGH |
| #16 | Fact-lookup for KIS queries | KIS | HIGH |
| #17 | Multi-scene KIS - key frame identification | KIS | MEDIUM |
| #18 | Camera angle / framing awareness | KIS | MEDIUM |
| #13 | VQA answer object precision (held object ID) | VQA | LOW |
| #10 | TRAKE Phase A - contextualized per-event | TRAKE | MEDIUM |


---

**Update to Improvement #15: OCR via Pre-computed DeepSolo + Vietnamese PARSeq**

The team preprocessing pipeline already runs **DeepSolo** (text detection) + **fine-tuned Vietnamese PARSeq** (text recognition) on keyframes. This is far superior to asking Qwen2-VL to read text.

**Revised Architecture:**

1. **Offline (preprocessing):** DeepSolo detects text bounding boxes. PARSeq reads each bounding box. Store results as JSON alongside each keyframe, e.g.:
   `data/ocr/L22_V018/174.json`
   ```json
   {"texts": ["Chương trình: Trao kinh phí hỗ trợ...", "FANA", "Khánh Hòa"]}
   ```

2. **Online (QA query):**
   - Detect OCR intent from `vlm_question` (keywords: "tên", "câu thơ", "ghi gì", "đọc được", "tiêu đề", "nội dung", "là gì").
   - If OCR intent detected:
     - Retrieve top-K candidate frames via FAISS (normal flow).
     - For each candidate, **load its pre-computed OCR JSON** instead of running VLM inference.
     - Use fuzzy/semantic match between question keywords and OCR texts to find the best answer.
     - Return the matched OCR text as the answer.
   - If NOT OCR intent: proceed with normal VLM `answer_question`.

3. **Fallback:** If OCR JSON is missing for a frame (not yet processed), fall back to VLM with an explicit OCR-mode prompt.

**Why this is better than VLM-based OCR:**
- PARSeq is trained specifically for Vietnamese text — handles diacritics correctly.
- DeepSolo handles rotated/curved text on banners, signs, and shrines reliably.
- Pre-computed = zero additional latency at query time.
- VLMs like Qwen2-VL frequently hallucinate text or skip small characters.

**Implementation tasks:**
- [ ] Confirm OCR output format/location from preprocessing team.
- [ ] Build `OcrIndex` class: loads OCR JSONs lazily per video, same pattern as `ObjectFilter`.
- [ ] Add `ocr_intent_detection()` helper to `vlm_pipeline.py` or `vqa.py`.
- [ ] In `VQATask.execute()`: branch on OCR intent before calling `answer_question`.
