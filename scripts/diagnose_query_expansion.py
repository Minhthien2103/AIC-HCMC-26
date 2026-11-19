"""Run the actual cached 3B expander without starting retrieval or VLM."""
import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.online_pipeline.query_expander import QueryExpander


def main():
    expander = QueryExpander(model_id='Qwen/Qwen2.5-3B-Instruct', device='cuda', load_in_4bit=True)
    if expander.model is None:
        raise RuntimeError('Could not load actual 3B model; refusing translation fallback for this diagnostic.')
    queries = ['Con lân màu trắng', 'A white quilin', 'Con lân màu trắng đang múa', 'A white lion dance costume']
    rows = []
    for query in queries:
        variants = expander.expand(query)
        rows.append({'query': query, 'variants': variants})
        print(json.dumps(rows[-1], ensure_ascii=True), flush=True)
    output = ROOT / 'scratch' / 'query_expansion_diagnostic.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', output)


if __name__ == '__main__':
    main()
