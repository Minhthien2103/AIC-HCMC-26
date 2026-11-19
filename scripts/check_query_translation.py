"""Run the production expander on a reported Vietnamese/English query pair."""
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.online_pipeline.query_expander import QueryExpander


def main():
    expander = QueryExpander()
    if expander.model is None:
        raise RuntimeError('SLM did not load; abort rather than test fallback translation')
    rows = []
    for query in ['Con lân màu trắng', 'A white quilin']:
        started = time.perf_counter()
        variants = expander.expand(query)
        if query not in expander.cache:
            raise RuntimeError('Expansion did not complete successfully')
        rows.append(dict(query=query, variants=variants,
                         seconds=round(time.perf_counter() - started, 3)))
        print(json.dumps(rows[-1], ensure_ascii=True), flush=True)
    output = ROOT / 'note' / 'query_translation_check.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(model=expander.model_id, device=expander.device,
                                     four_bit=expander.load_in_4bit, results=rows),
                                 ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
