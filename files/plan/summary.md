## AI Challenge HCMC 2026 — Task Overview & TycheVid Reference System (updated)

### 0. Data characteristics for 2026 — Sousveillance video

This is a major shift from the TycheVid 2024 dataset and changes which modules matter most.

| Characteristic | Implication for system design |
|---|---|
| **Big data / large volume** | Storage and indexing must scale — keyframe compression (WebP) and shot-based sampling from TycheVid remain relevant |
| **Unstructured** | No pre-existing metadata structure to lean on; everything must be derived from raw video |
| **Shaky / rotating camera angle** (wearable, not tripod) | Shot-boundary detection (AutoShot) and object detection will be noisier than 2024's news-style footage — expect more false shot cuts, motion blur, and detection dropouts. May need frame stabilization or motion-aware smoothing before running embeddings/OCR |
| **Constantly changing lighting** | Indoor/outdoor transitions common (e.g., market to kitchen). Embedding models sensitive to lighting (CLIP-family) may need augmentation-aware retrieval or lighting-invariant preprocessing |
| **Dense but highly personal semantic content** | Unlike news footage (public, labeled events), sousveillance is egocentric daily life — closer to **Lifelog Search Challenge (LSC)** style data than to VBS-style broadcast video. This actually validates AIC HCMC's own stated inspiration from LSC |
| **Long continuous single-person recordings** (e.g., 5-hour session: market → cooking → conversation) | Long, low-cut-density video means **temporal search across large spans** matters more than in short news clips — the "advanced temporal search" from TycheVid becomes even more central, not less |
| **Goal**: "the assistant must understand what the user did across the entire session" | This pushes hard toward **egocentric activity recognition** and **long-horizon narrative summarization**, not just moment-level shot matching |

**Practical consequences for your pipeline:**
- **ASR becomes less reliable as a primary signal** — ambient noise ("có tiếng ồn môi trường") means Whisper transcripts may be noisy; still useful but should be weighted lower or paired with confidence filtering, unlike TycheVid where ASR was a clean complementary signal.
- **Object detection needs egocentric-hand-object interaction awareness** — the example explicitly mentions "hành động tay cầm nắm đồ vật" (hand grasping objects). Co-DETR-style general object detection may need to be paired with a hand-object interaction model (e.g., EgoHOS-style) rather than plain bounding boxes, since *what the hand is doing* carries more query-relevant meaning than *what objects exist in frame*.
- **Keyframe selection logic (BEiT-3 diff-based) needs retuning** — camera shake will trigger false "significant difference" detections between frames, inflating keyframe count with near-duplicates. Consider adding optical-flow-based motion filtering to distinguish "camera shook" from "scene actually changed."
- **This directly supports the KISC and VQA tasks**: the example query itself ("what did the user do for those 5 hours") is structurally a **VQA / conversational** question over a long egocentric session — suggesting 2026's VQA and KISC tasks may lean toward session-level reasoning rather than single-frame lookup like 2024's VQA.

---

### 1. Competition tasks this year

| Task | Query type | Answer format | Notes |
|---|---|---|---|
| **KIS-T** (Textual KIS) | Detailed natural-language description of a video segment | Exact video + timestamp/frame | Single ground-truth target; description may be revealed incrementally |
| **KIS-V** (Visual KIS) | A short (~20s) video clip shown to participant | Exact matching frame in the dataset | No text at all — pure visual matching against a randomly sampled frame from the clip |
| **VQA** (Video Question Answering) | A question tied to a specific frame/moment, potentially spanning a long session | Localize the moment + answer the question | Combines retrieval (find *when*) with reasoning (answer *what*); sousveillance data pushes this toward session-level reasoning |
| **AVS** (Ad-hoc Video Search) — *not yet confirmed with official 2026 data* | General event/concept description | **Ranked list** of multiple relevant segments (not just one) | Classic text-to-video retrieval; evaluated by something like mAP rather than single-hit accuracy |
| **KISC** (Conversational KIS) — *not yet confirmed with official 2026 data* | Multi-turn dialogue refining a query over several rounds | Exact video segment, found interactively | New/rare format — no prior AIC HCMC paper documents it directly; likely benefits from long-session summarization given the data style |

> **Status as of now**: official 2026 problem statement and full dataset for the qualification round have not been fully released to you yet. AVS and KISC are new relative to the 2024 paper below, so there's no local precedent to copy from — treat them as extensions of the KIS-T/AVS retrieval core rather than separate systems.

---

### 2. What the TycheVid paper (AIC HCMC 2024, 2nd place public test) actually used

**Scope of that system**: only KIS-T, KIS-V, and VQA existed in 2024 — no AVS, no KISC. **Data was news/daily-life broadcast footage, not sousveillance** — stable camera, controlled lighting, clean audio. This matters: several of its modules assume conditions the 2026 data won't have.

| Component | Model/tool used | Purpose | 2026 sousveillance risk |
|---|---|---|---|
| Shot segmentation | AutoShot | Split video into coherent shots | May over-segment due to camera shake |
| Keyframe selection | BEiT-3 feature diff every 8th frame | Drop redundant near-duplicate frames | May under-perform — motion blur triggers false "differences" |
| Storage | WebP | Smaller keyframe files, faster I/O | Still valid |
| Embedding search | InternVL + BEiT-3 + OpenCLIP H-14, combined via **rank fusion** | Each model has different query-type strengths; fusion compensates for weaknesses | Valid, but lighting variance may need augmentation |
| OCR | DeepSolo (detection) + PARSeq fine-tuned on Vietnamese (recognition) | Extract Vietnamese on-screen text | Less relevant — sousveillance has far less structured on-screen text than news |
| ASR | Whisper, 6-second window around each keyframe | Tightly aligned transcript | Degraded by ambient noise in 2026 data |
| Object filtering | Co-DETR → stored in Polars dataframe | Count/spatial-relationship queries | Needs hand-object interaction awareness added |
| Query expansion | GPT-4o paraphrasing | Mitigate vague queries | Same idea works, swap for local LLM |
| Score fusion | Min-max normalization, aggregate ranking | Cross-modality comparability | Still valid |
| **Advanced temporal search** (core contribution) | Multi-stage decomposition + ±10 shot window re-ranking | Resolves before/after ambiguity | **More valuable in 2026** — long single-session recordings need this even more than short news clips |
| UI | "U" mode (1 rep frame/cluster) / "G" mode (full cluster) | Balance overview vs detail | Still valid |
| Databases | Milvus + Elasticsearch + Polars | Per-modality storage | Still valid |

**Dataset used**: 1,476 videos, 328 hours, ~303,000 shots, ~626,906 keyframes — Vietnamese news/daily-life content.

---

### 3. Suggested adaptation for AVS and Conversational KIS (KISC) — updated for sousveillance context

**For AVS (Ad-hoc Video Search):**
- Reuse the same embedding + rank-fusion + object-filter pipeline built for KIS-T — output contract changes to top-N ranked list instead of single-best commit.
- Given the shaky/lighting-variable footage, weight **motion-robust embeddings** (models less sensitive to blur, e.g., video-native embeddings over frame-only CLIP where feasible) more heavily in the fusion for AVS recall.
- Add duplicate/near-duplicate suppression across the returned list — critical here since egocentric footage naturally produces long homogeneous stretches (e.g., minutes of "walking through market").
- Skip per-query temporal search for AVS by default; keep it as an optional refinement toggle.

**For KISC (Conversational KIS):**
- Treat it as KIS-T with a persistent query state accumulated across turns via local Qwen2.5-7B-Instruct (structured JSON merge, matching your ViPlag approach).
- Given the long-session nature of 2026 data, prioritize **session/activity-segment indexing** (e.g., pre-chunking each 5-hour recording into coarse activity segments — "at the market," "cooking," "talking") as a first-pass filter before fine-grained shot search — this mirrors how a person would actually narrow down "which part of the day" before "which exact moment."
- Reuse advanced temporal search per turn with shrinking window size as confidence increases.
- Add a confidence/convergence signal to decide when to stop asking for hints.
- Keep conversation-state management decoupled from the retrieval backend so it can be rewired once official KISC turn-format rules are published.

**Cross-cutting recommendation given the sousveillance reveal:** since the example use case ("Trợ lý ảo phải tự hiểu người dùng đã làm gì trong suốt 5 tiếng đó") frames the ideal system as one that narrates *what the user did*, consider building a **lightweight activity/event summarization layer** on top of the retrieval backend — even a lifelog-style "generate short captions per activity segment" pass using your local VLM — as shared infrastructure feeding both VQA (answer "what did I do") and KISC (narrow down "which part of my day").