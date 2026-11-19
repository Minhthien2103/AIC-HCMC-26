"""Exercise the production translator/expander on GPU and record outputs and latency."""
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from src.online_pipeline.query_expander import QueryExpander
from compare_4b_translation import QUERIES


def main():
    expander = QueryExpander()
    if expander.model is None:
        raise RuntimeError('Actual model not loaded')
    (ROOT / 'scratch').mkdir(exist_ok=True)
    if '--trace' in sys.argv:
        generate = expander._generate
        def traced(messages, **kwargs):
            value = generate(messages, **kwargs)
            print('RAW: ' + json.dumps(value, ensure_ascii=True), flush=True)
            return value
        expander._generate = traced
    rows = []
    for query in QUERIES:
        started = time.perf_counter()
        variants = expander.expand(query)
        row = dict(query=query, variants=variants, used_fallback=query not in expander.cache, seconds=round(time.perf_counter()-started, 3))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
        (ROOT / 'scratch' / ('translation_4b_trace.json' if '--trace' in sys.argv else 'translation_4b_pipeline.json')).write_text(
            json.dumps(dict(model=expander.model_id, results=rows,
                            peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20)),
                       ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
