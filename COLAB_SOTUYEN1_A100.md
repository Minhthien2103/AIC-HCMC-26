# SOTUYEN1 on Colab A100 — deadline runbook

This runbook deliberately leaves the old Drive checkout untouched. It clones
`aic2026-submission` to Colab's local SSD, stages only the required 177k
keyframes/index files locally, and writes small checkpoints plus the final ZIP
back to Drive. `metadata.parquet` already contains the frame mapping used by
the online pipeline, so a separate `data/map-keyframes` folder is not needed.

## 1. Select A100 and mount Drive

In Colab select **Runtime > Change runtime type > A100 GPU**, then run:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Stop if the result is not an A100.

## 2. Clone the new branch and stage assets

```bash
%%bash
set -Eeuo pipefail
mkdir -p /content/aic26
if [[ ! -d /content/aic26/AIC-HCMC-26/.git ]]; then
  git clone --depth 1 --branch aic2026-submission \
    https://github.com/Minhthien2103/AIC-HCMC-26.git \
    /content/aic26/AIC-HCMC-26
fi
bash /content/aic26/AIC-HCMC-26/scripts/colab_sotuyen1_setup.sh
```

Expected setup time is 25–45 minutes: 15–30 minutes for the 28.4 GB keyframe
archive, 5–15 minutes for models/cache, and 3–8 minutes for dependencies and
indexes. The script checks exactly 25 query files and at least 170k JPEGs.

## 3. Start the standard run

```bash
%%bash
set -Eeuo pipefail
bash /content/aic26/AIC-HCMC-26/scripts/run_sotuyen1_a100.sh standard
```

The final artifact is written to:

```text
MyDrive/AIC_HCMC_26/AIC-HCMC-26/outputs/SOTUYEN1_a100_submission.zip
```

The wrapper validates the ZIP before printing `READY TO UPLOAD`. Do not upload
the checkpoint directory or provenance directory.

## 4. Deadline fallback

The standard profile uses about 1,461 Qwen generations: 20 KIS queries at 64
images plus analysis, four QA queries at 24 images, and one 3-event TRAKE query
over five videos. On an A100 BF16, budget 30–55 minutes for the batch.

Count finished query checkpoints at any time:

```bash
!find /content/drive/MyDrive/AIC_HCMC_26/AIC-HCMC-26/outputs/SOTUYEN1_a100_checkpoint -maxdepth 1 -name '*.csv' | wc -l
```

If the total elapsed time reaches 85 minutes and fewer than 15 of 25 CSVs are
complete, interrupt the running cell once, then run:

```bash
%%bash
set -Eeuo pipefail
bash /content/aic26/AIC-HCMC-26/scripts/run_sotuyen1_a100.sh emergency
```

The emergency profile keeps every already validated standard CSV and generates
only the remaining queries with 32 KIS Qwen frames and smaller temporal
budgets. Budget 15–30 minutes for the remainder. Reserve the final 15 minutes
for downloading/uploading the validated ZIP.

## Important constraints

- Do not run from `MyDrive/data/keyframes`; 177k small Drive reads are much
  slower than local SSD access.
- Do not build ViT-H or PaddleOCR during this two-hour run.
- Do not add `--require-kis-assets`: the Drive screenshot does not show the new
  media-E5 index, so the intended fallback is ViT-B/32 + Qwen.
- Do not add `--offline` unless an evidence cache exists for every KIS query.
- A rerun is safe: valid per-query CSVs are skipped by `--resume`.
