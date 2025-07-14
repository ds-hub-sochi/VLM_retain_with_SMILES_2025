import json
import os
import torch
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image

import jiwer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

# --- Configuration keys ---
REQUIRED_CONFIGS = ('prompt', 'json_path', 'model_type')  # model_path is optional now
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

def setup_model(model_folder: str, model_type: str):
    if model_type.lower() == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_folder,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_folder, trust_remote_code=True, use_fast=True)
        processor = AutoProcessor.from_pretrained(model_folder, trust_remote_code=True, use_fast=True)
    elif model_type.lower() == "internvl":
        from transformers import InternVLForConditionalGeneration
        model = InternVLForConditionalGeneration.from_pretrained(
            model_folder,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_folder, trust_remote_code=True, use_fast=True)
        processor = AutoProcessor.from_pretrained(model_folder, trust_remote_code=True, use_fast=True)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model, tokenizer, processor

def find_assistant_content(image_path: str, data: list) -> str:
    for entry in data:
        if image_path in entry.get('images', []):
            for msg in entry.get('messages', []):
                if msg.get('role') == 'assistant':
                    return msg.get('content')
    return None

def normalize_text(text):
    if isinstance(text, list):
        normalized = ' '.join(text)
    else:
        normalized = text.replace('\n', ' ')
    return ' '.join(normalized.split())

def calculate_wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference, hypothesis)

def calculate_bleu(reference: str, hypothesis: str) -> float:
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference], hypothesis, smoothing_function=smoothie)

def compute_levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n]

def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    dist = compute_levenshtein_distance(ref, hyp)
    return dist / len(ref) if ref else 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference + metrics over a JSON dataset.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config, 'r', encoding='utf-8') as cf:
        cfg = yaml.safe_load(cf)

    # GPU cost settings
    gpu_cost_per_hour = cfg.get('gpu_cost_per_hour', 3.5)  # $ per hour
    price_per_second = gpu_cost_per_hour / 3600.0

    # Optional model_path, fallback to default HF weights
    model_folder = cfg.get('model_path')
    if not model_folder:
        print(f"No 'model_path' specified – loading default weights from Hugging Face ({DEFAULT_MODEL}).")
        model_folder = DEFAULT_MODEL

    # Required configs
    for name in REQUIRED_CONFIGS:
        if not cfg.get(name):
            raise ValueError(f"Config '{name}' must be specified.")

    prompt     = cfg['prompt']
    json_path  = cfg['json_path']
    model_type = cfg['model_type']
    output_csv = cfg.get('output_csv', 'results.csv')

    # --- Setup model ---
    model, tokenizer, processor = setup_model(model_folder, model_type)
    params = {'temperature': 0.9742102638497399, 
            'top_p': 0.8852959163547229,
            'typical_p': 0.910072663704335, 
            'top_k': 21, 
            'repetition_penalty': 1.1293594515176946, 
            'do_sample': True, 
            'assistant_confidence_threshold': 0.6921904635795412}

    # --- Load dataset ---
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --- Collect per-image metrics ---
    records = []
    total_gen_time = 0.0

    for entry in tqdm(data, desc="Entries"):
        for image_path in entry.get('images', []):
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image_path},
                    {'type': 'text',  'text': prompt},
                ],
            }]

            # prepare text+images
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            pil_images = [Image.open(m['image']).convert("RGB")
                          for m in messages[0]['content'] if m['type']=='image']

            if model_type.lower() == "qwen":
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors='pt'
                ).to(model.device)
            else:  # internvl
                inputs = processor(
                    text=text_input,
                    images=pil_images,
                    padding=True,
                    return_tensors='pt'
                )
                if 'pixel_values' in inputs:
                    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
                inputs = inputs.to(model.device)

            # --- measure inference time on GPU ---
            start_event = torch.cuda.Event(enable_timing=True)
            end_event   = torch.cuda.Event(enable_timing=True)
            start_event.record()

            generated_ids = model.generate(**inputs, max_new_tokens=1024, **params)

            end_event.record()
            torch.cuda.synchronize()
            gen_time = start_event.elapsed_time(end_event) / 1000.0  # seconds
            total_gen_time += gen_time

            # decode
            trimmed = [out_ids[len(in_ids):]
                       for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_list = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            hypothesis = ' '.join(output_list) if isinstance(output_list, list) else str(output_list)

            reference_raw = find_assistant_content(image_path, data)
            if reference_raw is None:
                continue

            ref_norm = normalize_text(reference_raw)
            hyp_norm = normalize_text(hypothesis)

            records.append({
                'image_path':      image_path,
                'reference':       ref_norm,
                'hypothesis':      hyp_norm,
                'WER':             calculate_wer(ref_norm, hyp_norm),
                'BLEU-4':          calculate_bleu(ref_norm, hyp_norm),
                'CER':             compute_cer(ref_norm, hyp_norm),
                'inference_time_s': gen_time
            })

    # --- Compute dataset averages ---
    wers  = [r['WER']   for r in records]
    bleus = [r['BLEU-4'] for r in records]
    cers  = [r['CER']   for r in records]

    avg_wer  = sum(wers)  / len(wers)  if wers  else 0.0
    avg_bleu = sum(bleus) / len(bleus) if bleus else 0.0
    avg_cer  = sum(cers)  / len(cers)  if cers  else 0.0

    for r in records:
        r['avg WER']  = avg_wer
        r['avg BLEU'] = avg_bleu
        r['avg CER']  = avg_cer

    # --- Estimate total GPU cost ---
    total_cost = total_gen_time * price_per_second
    print(f"\nTotal generation time: {total_gen_time:.2f} s")
    print(f"GPU cost per hour: ${gpu_cost_per_hour:.2f} → price per second = ${price_per_second:.6f}")
    print(f"Estimated total inference cost: ${total_cost:.2f}")

    # --- Save to CSV ---
    df = pd.DataFrame(records, columns=[
        'image_path', 'reference', 'hypothesis',
        'WER', 'BLEU-4', 'CER',
        'avg WER', 'avg BLEU', 'avg CER',
        'inference_time_s'
    ])
    df.to_csv(output_csv, index=False)
    print(f"All done — saved metrics for {len(records)} images to {output_csv}")