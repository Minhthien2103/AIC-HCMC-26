"""Compare current 4B production prompting with plain translation-only prompting.

The initial report was captured before the two-stage translation change.
Rerunning compares the current implementation, not a frozen historical baseline.
"""
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

QUERIES = [
    'Con lân màu trắng', 'A white quilin', 'Con lân màu trắng đang múa',
    'Hai con lân đỏ và vàng múa trước cửa hàng',
    'Tay đua xe đạp mặc áo trắng', 'Người phụ nữ mặc áo vàng đạp xe',
    'Một người mặc áo dài trắng đứng cạnh xe đạp',
    'Một con ngựa trắng chạy trên cánh đồng',
    'Người đàn ông đứng bên trái chiếc xe cứu thương',
    'Một người cầm đầu lân, không đội lên đầu',
    'Một người cầm nón lá đứng trên thuyền',
    'Một con kỳ lân có một chiếc sừng trong phim hoạt hình',
]


def main():
    expander = QueryExpander(model_id='Qwen/Qwen3-4B-Instruct-2507', device='cuda', load_in_4bit=True)
    if expander.model is None:
        raise RuntimeError('Actual model unavailable; refusing fallback test')
    rows = []
    for query in QUERIES:
        started = time.perf_counter()
        baseline = expander.expand(query)
        baseline_s = time.perf_counter() - started
        messages = [
            {'role': 'system', 'content': 'Translate the user text from Vietnamese to natural English for image search. Preserve the exact meaning, subject, action, colors, numbers, negation and spatial relationships. Do not add information or describe an imagined scene. Interpret idioms and cultural terms in context, not word by word. If the text is already English, copy it unchanged. Output only the translation, no explanations or alternatives.'},
            {'role': 'user', 'content': query},
        ]
        text = expander.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = expander.tokenizer([text], return_tensors='pt').to(expander.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = expander.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        translated = expander.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        row = dict(query=query, baseline=baseline, translation_only=translated,
                   baseline_seconds=round(baseline_s, 3), translation_seconds=round(time.perf_counter()-started, 3))
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
        target = ROOT / 'scratch' / 'translation_4b_comparison.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(dict(model=expander.model_id, results=rows,
                                          peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20)),
                                    ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
