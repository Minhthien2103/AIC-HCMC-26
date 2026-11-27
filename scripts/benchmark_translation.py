"""Benchmark faithful translation with local models or the production Qwen prompt.

No expansion, retrieval, fallback translator, or production configuration changes.
Run only after the GPU is free. Models are loaded sequentially, from local cache.
Use --backend vinai --beams 5 --output-dir scratch/translation_vinai for VinAI.
Use --backend gemma --output-dir scratch/translation_gemma_nf4 for TranslateGemma NF4.
"""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NLLB_ID = 'facebook/nllb-200-distilled-600M'
# Safetensors conversion by SFconvertbot in the original model repository (PR 45).
NLLB_REVISION = 'a3e77be725cf30383f1faeb8d2f0b0ea98ac554e'
VINAI_ID = 'vinai/vinai-translate-vi2en-v2'
VINAI_REVISION = '7535503ab2ac27efd842005bf429f341f2e96143'
HY_ID = 'tencent/Hy-MT2-1.8B'
HY_REVISION = '9a341cd1b679d3efd23b46e847b01745a71ed792'
GEMMA_ID = 'google/translategemma-4b-it'
GEMMA_REVISION = '10042cb0e6e7fdce748996a71dc3dc432a4e0c89'

CASES = [
    ('culture', 'Con lân màu trắng'),
    ('culture', 'Con lân màu trắng đang múa'),
    ('culture', 'Hai con lân đỏ và vàng múa trước cửa hàng'),
    ('culture', 'Một người cầm đầu lân, không đội lên đầu'),
    ('culture', 'Một con kỳ lân có một chiếc sừng trong phim hoạt hình'),
    ('culture', 'Một người mặc áo dài trắng đứng cạnh xe đạp'),
    ('culture', 'Một người cầm nón lá đứng trên thuyền'),
    ('subject', 'Tay đua xe đạp mặc áo trắng'),
    ('subject', 'Người phụ nữ mặc áo vàng đạp xe'),
    ('subject', 'Một con ngựa trắng chạy trên cánh đồng'),
    ('spatial', 'Người đàn ông đứng bên trái chiếc xe cứu thương'),
    ('spatial', 'Người đàn ông đứng sau chiếc xe, không phải phía trước'),
    ('spatial', 'Người phụ nữ bên phải cầm túi đỏ, người bên trái cầm túi xanh'),
    ('spatial', 'Một chiếc xe máy đi phía sau xe tải màu xanh'),
    ('action', 'Một người đẩy xe đạp, không ngồi trên xe'),
    ('action', 'Người đàn ông đưa chiếc mũ cho một bé gái'),
    ('action', 'Một bé trai đưa chiếc mũ cho người đàn ông'),
    ('action', 'Người phụ nữ cầm chiếc áo vàng, không mặc nó'),
    ('negation', 'Không có người nào đứng trên cầu'),
    ('negation', 'Một người không đội mũ bảo hiểm đang đi xe máy'),
    ('count', 'Ba người mặc áo xanh đứng cạnh hai chiếc xe màu đỏ'),
    ('count', 'Hai người đàn ông và một phụ nữ cùng kéo một tấm lưới'),
    ('number', 'Xe cứu thương màu trắng có số 115'),
    ('number', 'Tay đua cuối cùng mặc áo số 71'),
    ('proper_name', 'Một người đứng trước biển hiệu ghi Bến Tre'),
    ('proper_name', 'Một cửa hàng có biển hiệu ghi CƠM TẤM'),
    ('ambiguous', 'Một người mặc áo xanh đứng cạnh xe'),
    ('ambiguous', 'Một người đang đá cầu ngoài sân'),
    ('long', 'Ba tay đua xe đạp chạy trên đường đất có nhiều bụi. Người đầu mặc áo đỏ, người thứ hai mặc áo xanh. Người cuối mặc áo trắng và mang số 71.'),
    ('long', 'Một người phụ nữ mặc áo vàng đứng bên trái xe cứu thương màu trắng. Cô ấy cầm một chiếc mũ đỏ và đưa nó cho bé trai đứng bên phải, không phải người đàn ông phía sau xe.'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['nllb', 'vinai', 'hy', 'gemma', 'qwen', 'both'], default='both')
    p.add_argument('--hy-4bit', action='store_true', help='Test Hy with NF4 on GPU')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--beams', type=int, default=4)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'scratch/translation_nllb')
    args = p.parse_args()
    if args.beams < 1 or args.limit < 0:
        p.error('Invalid beams/limit')
    # Keep adapter auto-detection from making network requests during timing.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from src.online_pipeline.query_translation import translation_messages, parse_translation
    from src.online_pipeline.query_expander import QueryExpander
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing to silently run CPU')
        free, _ = torch.cuda.mem_get_info()
        if free < 4 * 1024**3:
            raise RuntimeError('Less than 4 GiB VRAM free. Stop backend before this benchmark.')
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = CASES[:args.limit] if args.limit else CASES
    names = ['nllb', 'qwen'] if args.backend == 'both' else [args.backend]
    reports = {}
    for name in names:
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if name in ('nllb', 'vinai'):
            model_id = VINAI_ID if name == 'vinai' else NLLB_ID
            revision = VINAI_REVISION if name == 'vinai' else NLLB_REVISION
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                                      src_lang='vi_VN' if name == 'vinai' else 'vie_Latn',
                                                      local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, revision=revision, local_files_only=True, use_safetensors=True,
                torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32)
            model = model.to(args.device).eval()
            def translate(text):
                inputs = tokenizer(text, return_tensors='pt', truncation=False).to(args.device)
                if inputs.input_ids.shape[1] > 512:
                    raise ValueError('Input exceeds NLLB training length; not silently truncating')
                with torch.inference_mode():
                    lang_options = ({'decoder_start_token_id': tokenizer.convert_tokens_to_ids('en_XX')}
                                    if name == 'vinai' else
                                    {'forced_bos_token_id': tokenizer.convert_tokens_to_ids('eng_Latn')})
                    output = model.generate(**inputs, **lang_options,
                                            max_new_tokens=192, num_beams=args.beams, do_sample=False,
                                            early_stopping=True)
                if output.shape[1] >= 193 and output[0, -1].item() != tokenizer.eos_token_id:
                    raise ValueError('Translation exceeded token budget')
                return tokenizer.decode(output[0], skip_special_tokens=True).strip()
        elif name == 'gemma':
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
            if args.device != 'cuda':
                raise ValueError('TranslateGemma benchmark requires CUDA')
            model_id, revision = GEMMA_ID, GEMMA_REVISION
            processor = AutoProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, revision=revision, local_files_only=True,
                dtype=torch.bfloat16, device_map={'': 'cuda'},
                quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                    bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True)).eval()
            def translate(text):
                messages = [{'role': 'user', 'content': [{'type': 'text',
                    'source_lang_code': 'vi', 'target_lang_code': 'en', 'text': text}]}]
                inputs = processor.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors='pt').to(args.device)
                # Text-only inputs have no image blocks. Removing the all-zero
                # image type IDs selects Gemma's ordinary causal/sliding masks,
                # avoiding the vision overlay path that requires Torch >=2.6.
                type_ids = inputs.pop('token_type_ids', None)
                if type_ids is not None and torch.any(type_ids != 0).item():
                    raise ValueError('Expected text-only TranslateGemma input')
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
                generated = output[0, inputs.input_ids.shape[1]:]
                if len(generated) >= 256:
                    raise ValueError('TranslateGemma response exceeded token budget')
                result = processor.decode(generated, skip_special_tokens=True).strip()
                if not result:
                    raise ValueError('Empty TranslateGemma translation')
                return result
        elif name == 'hy':
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            model_id, revision = HY_ID, HY_REVISION
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
            options = dict(dtype=torch.bfloat16 if args.device == 'cuda' else torch.float32)
            if args.hy_4bit:
                if args.device != 'cuda':
                    raise ValueError('Hy NF4 requires CUDA')
                options.update(device_map={'': 'cuda'}, quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
                                                        local_files_only=True, **options)
            if not args.hy_4bit:
                model.to(args.device)
            model.eval()
            def translate(text):
                messages = [{'role': 'user', 'content':
                    'Translate the following text into English. Note that you should only output '
                    'the translated result without any additional explanation:\n' + text}]
                inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors='pt').to(args.device)
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                            repetition_penalty=1.05)
                generated = output[0, inputs.input_ids.shape[1]:]
                if len(generated) >= 256:
                    raise ValueError('Hy response exceeded token budget')
                return tokenizer.decode(generated, skip_special_tokens=True).strip()
        else:
            model = QueryExpander(device=args.device, load_in_4bit=args.device == 'cuda')
            if model.model is None:
                raise RuntimeError('Qwen failed to load; fallback comparison is not allowed')
            def translate(text):
                return parse_translation(model._generate(translation_messages(text)), text)
            model_id = model.model_id
        if args.device == 'cuda':
            torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        translate('Một người đang đi bộ.')  # Warm-up excluded from per-query latency.
        rows = []
        for category, query in cases:
            if args.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            result, error = None, None
            try:
                result = translate(query)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            if args.device == 'cuda':
                torch.cuda.synchronize()
            row = dict(category=category, query=query, translation=result, error=error,
                       seconds=round(time.perf_counter()-start, 4))
            rows.append(row)
            print(json.dumps(dict(backend=name, **row), ensure_ascii=True), flush=True)
            report = dict(model=model_id, revision=revision if name in ('nllb', 'vinai', 'hy', 'gemma') else 'local cached Qwen',
                          precision='nf4' if name == 'gemma' else (('nf4' if args.hy_4bit else 'bf16') if name == 'hy' else None),
                          text_only_masks=name == 'gemma',
                          device=args.device, transformers=transformers.__version__,
                          beams=args.beams if name in ('nllb', 'vinai') else 1,
                          load_seconds=load_seconds, results=rows,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device == 'cuda' else None,
                          median_seconds=statistics.median(r['seconds'] for r in rows))
            (args.output_dir / f'{name}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        reports[name] = report
        del model, translate
        if name in ('nllb', 'vinai', 'hy'):
            del tokenizer
        if name == 'gemma':
            del processor
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    if len(reports) == 2:
        with (args.output_dir / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['category', 'query_vi', 'nllb', 'qwen', 'nllb_seconds', 'qwen_seconds'])
            for n, q in zip(reports['nllb']['results'], reports['qwen']['results']):
                writer.writerow([n['category'], n['query'], n['translation'], q['translation'], n['seconds'], q['seconds']])


if __name__ == '__main__':
    main()
