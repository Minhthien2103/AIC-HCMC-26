# Adding VLM Reranking to TRAKE Task

## Overview
The recent commits from your teammate added a Qwen2-VL visual reranker to the KIS and VQA pipelines. However, the 	rake.py module still relies purely on CLIP + FAISS search scores.

To fulfill your request to add a VLM reranker in **all tasks**, I will implement a sequence reranker in 	rake.py.

## Proposed Changes

### src/tasks/trake.py
We will add a new _qwen_rerank_sequences method to TrakeTask.
Because VLM inference is slow, we cannot evaluate all generated sequences. Instead, we will:
1. Generate the initial CLIP-based sequence chains.
2. Filter the top N highest scoring sequences (e.g., top 15 chains).
3. For each top chain, use lm_pipeline.score_kis_match to score every keyframe in the sequence against its corresponding event description.
4. Add the total VLM score (or average) to the sequence's base score to produce a qwen_match_score for the sequence.
5. Promote sequences that have strong visual VLM verification, ensuring visually verified sequences bubble to the top.

### src/config.py
Add TRAKE_VLM_PROMOTE_MIN_SCORE and TRAKE_VLM_TOP_K to control how many sequences are sent to the VLM to avoid excessive GPU processing time.

## Verification
- Test 	rake.py with a sample query to ensure it accurately parses frames and reranks them.
- Once validated, I will execute un_queries.py to produce your final submission/ CSVs.
