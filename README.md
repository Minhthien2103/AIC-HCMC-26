# AIC-HCMC-26 — AIC2026 qualification pipeline

The repository contains the retrieval runtime and a separate Codabench result
submission tool.  The result ZIP must contain only `submission/*.csv`; it is not
the same as a Codabench code submission.

## CUDA/Colab setup

```bash
pip install -r requirements.txt
python scripts/preflight.py
```

The VQA/TRAKE VLM path requires a Linux CUDA runtime and
`bitsandbytes>=0.46.1`.  KIS retrieval and the validators can be run without
loading Qwen2-VL.

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
