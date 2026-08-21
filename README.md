# AIC-HCMC-26 — AIC2026 qualification pipeline

The repository contains the retrieval runtime and a separate Codabench result
submission tool.  The result ZIP must contain only `submission/*.csv`; it is not
the same as a Codabench code submission.

## CUDA/Colab setup

```bash
pip install -r requirements.txt
python scripts/preflight.py --repo-root . --kis-profile fast --offline
```

The VQA/TRAKE VLM path requires a Linux CUDA runtime and
`bitsandbytes>=0.46.1`.  KIS retrieval and the validators can be run without
loading Qwen2-VL.

## KIS final-run assets (GPU, video-first)

The default `fast` profile uses the existing ViT-B/32 index, official BTC
media metadata embedded with multilingual E5, candidate-only OCR, local mBART
translation and Qwen2-VL. It retrieves videos first, then localises frames in
the selected videos and creates bounded frame-neighbourhood proposals from
`map-keyframes`. It does not download/decode raw video or call a network
service while generating a submission. ViT-H/14 is an optional `full` ablation,
not a prerequisite for a final run.

Prepare the fast assets once on the persistent Drive/project copy. The first
two lines are the only network-permitted preparation step.

```bash
python scripts/prepare_kis_assets.py \
  --repo-root . --device cuda \
  --download-media-info --build-media-index

# Separate explicit network step. The result is read-only during the final run.
python scripts/prepare_kis_assets.py \
  --repo-root . --device cuda \
  --build-evidence-cache --manifest query/manifest_full.json

# Optional gate after all assets/models are cached.
python scripts/prepare_kis_assets.py \
  --repo-root . --device cuda --kis-profile fast --smoke-test --manifest query/manifest_full.json
```

Optional full-profile ViT-H build (resumable after a Colab disconnect):

```bash
python scripts/prepare_kis_assets.py \
  --repo-root . --device cuda --build-vith-index --resume \
  --batch-size 64 --checkpoint-every 16
```

First make a non-submitted review ZIP and inspect the generated contact sheets
in `outputs/kis_review/<query-id>/`. The runner creates one editable aggregate
template named `review_manifest.template.json`; use only listed candidate keys
and video IDs. `pin`/`pin_video` are explicit ordered lists, `keep` records
approval without a score boost, and `reject` moves a generated candidate after
all non-rejected candidates. Review never adds a video/frame/answer that the
pipeline did not generate.

```bash
python scripts/generate_submission.py \
  --manifest query/manifest_full.json --output outputs/review_only.zip \
  --device cuda --kis-profile fast --require-kis-assets --offline \
  --review-output-dir outputs/kis_review --max-rows 100
```

Copy/edit that template (for example to `outputs/kis_review/final_review.json`)
then make the final, separately named ZIP:

```bash
python scripts/generate_submission.py \
  --manifest query/manifest_full.json --output outputs/final_fast_video_first.zip \
  --device cuda --kis-profile fast --require-kis-assets --offline \
  --review-manifest outputs/kis_review/final_review.json --max-rows 100

python scripts/validate_submission.py \
  --zip outputs/final_fast_video_first.zip --manifest query/manifest_full.json
```

Each generation writes `<zip-stem>_provenance/run_provenance.json` next to the
ZIP with model IDs, effective budgets, asset/cache checksums and review input.
The ZIP itself remains BTC-compliant and contains only `submission/*.csv`.

## Generate and validate a result ZIP

```bash
python scripts/generate_submission.py \
  --queries-dir /path/to/query-pack \
  --output /tmp/aic2026-submission.zip \
  --device cuda

python scripts/validate_submission.py \
  --zip /tmp/aic2026-submission.zip \
  --queries-dir /path/to/query-pack
```

For organizers' query packs that need explicit descriptions/questions/events,
use `--manifest manifest.json`.  The runner supports `_kis.txt`, `_qa.txt` and
`_trake.txt` query names and writes at most 100 rows per query.

## Streamlit debug UI

```bash
python -m streamlit run app.py
```

The UI uses the same output formatter as the CLI and can download normalized
CSV files for the currently displayed query.  It is intended for inspection;
the batch runner is the submission path.
