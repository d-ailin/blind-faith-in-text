"""
Simple HuggingFace dataset evaluator.

Usage examples (run from `raw_code/`):
python hf_evaluator.py --ds_name facebook/textvqa --model_type adapter --model_name llava-hf/llava-1.5-7b-hf --max_samples 200
python hf_evaluator.py --ds_name lmms-lab/VQAv2 --model_type hf --model_name gpt2 --max_samples 100

Supports model_type:
- adapter: uses the repo's `utils.QueryModel` (for multimodal adapters like Llava)
- hf: uses `transformers` text-generation pipeline (text-only)
- openai: uses OpenAI ChatCompletion (requires OPENAI_API_KEY)

Outputs:
- per-sample JSONL at `out_file` with fields: question_id, question, gt_answers, pred_answer, is_correct
- prints and saves final accuracy
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

# ensure raw_code is on path when running from raw_code/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset

# import local eval utilities
from eval_utils import best_subspan_em, eval_func, DocVQAEvaluator

# optional imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except Exception:
    pipeline = None

try:
    from openai import OpenAI
    import openai
    
    client = OpenAI()
except Exception:
    print('openai exception')
    openai = None

# adapter support (multimodal / repo adapters)
QueryModel = None
try:
    from utils.main import QueryModel
except Exception as e:
    print(f"Warning: Could not import QueryModel: {e}")


def parse_string_list(val):
    """
    Parse string representation of a list (e.g., "['answer1', 'answer2']") into an actual list.
    If already a list, return as-is. If a string that looks like a list, parse it. Otherwise wrap in list.
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        # Check if it looks like a string representation of a list
        if val.startswith('[') and val.endswith(']'):
            try:
                import ast
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    # ensure all items are strings
                    return [str(x) for x in parsed]
                else:
                    return [str(parsed)]
            except Exception:
                # if parsing fails, treat entire string as single answer
                return [val]
        else:
            # single string answer
            return [val]
    # single non-string answer
    return [str(val)]


def normalize_answer(answer):
    """
    Normalize answer for comparison:
    - Convert number words (one, two, three, etc.) to digits
    - Lowercase
    - Strip whitespace
    - Remove punctuation
    """
    import re

    if not isinstance(answer, str):
        answer = str(answer)

    # Mapping of number words to digits
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
        'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100',
        'thousand': '1000', 'million': '1000000'
    }

    answer = answer.lower().strip()

    # Replace number words with their digit equivalents
    for word, digit in number_words.items():
        # Use word boundaries to avoid partial matches
        answer = re.sub(r'\b' + word + r'\b', digit, answer)

    # Remove punctuation but keep spaces for now
    answer = re.sub(r'[^\w\s]', '', answer)

    # Remove extra whitespace
    answer = ' '.join(answer.split())

    return answer


def check_answer_match(pred, gt_answers):
    """
    Check if prediction matches any ground truth answer.
    Handles number word variations (e.g., 'two' and '2').
    """
    if not gt_answers:
        return False

    pred_norm = normalize_answer(str(pred))

    for gt in gt_answers:
        gt_norm = normalize_answer(str(gt))

        # Exact match after normalization
        if pred_norm == gt_norm:
            return True

        # Substring match (for longer answers)
        if len(gt_norm) > 0 and gt_norm in pred_norm:
            return True
        if len(pred_norm) > 0 and pred_norm in gt_norm:
            return True

    return False


def detect_fields(sample):
    # heuristics to find question, answers, image
    q = None
    qid = None
    answers = None
    image = None
    full_prompt = None
    text_type = None
    additional_info = None

    # common fields
    if 'question' in sample:
        q = sample['question']
    elif 'question_text' in sample:
        q = sample['question_text']
    elif 'query' in sample:
        q = sample['query']

    if 'full_prompt' in sample:
        full_prompt = sample['full_prompt']

    if 'question_id' in sample:
        qid = sample['question_id']
    elif 'questionId' in sample:
        qid = sample['questionId']
    elif 'id' in sample:
        qid = sample['id']

    if 'answers' in sample:
        answers = sample['answers']
    elif 'multiple_choice_answer' in sample:
        answers = [sample['multiple_choice_answer']]
    elif 'gt_answers' in sample:
        answers = sample['gt_answers']
    elif 'label' in sample:
        answers = [sample['label']]

    # Parse answers if they're in string representation format (e.g., "['ans1', 'ans2']")
    if answers is not None:
        answers = parse_string_list(answers)

    # additional metadata fields commonly present in this dataset
    for k in ['text_type', 'intext_type', 'text_kind']:
        if k in sample:
            text_type = sample[k]
            break

    for k in ['info', 'additional_info', 'context', 'description', 'extra']:
        if k in sample:
            additional_info = sample[k]
            break

    # image may be a PIL-compatible object or url or bytes
    for k in ['image', 'img', 'image_path', 'image_url']:
        if k in sample:
            image = sample[k]
            break

    # Fallback: if question_id is empty, use "{question}/{answers}" as ID
    if qid is None or qid == '':
        qid = f"{q}/{answers}"

    return qid, q, answers, image, full_prompt, text_type, additional_info


class HFTextGenModel:
    def __init__(self, model_name, device=-1, max_new_tokens=64, temperature=0.0, input_type='text-only'):
        if pipeline is None:
            raise RuntimeError('transformers not installed')
        self.model_name = model_name
        self.pipeline = pipeline('text-generation', model=model_name, tokenizer=model_name, device=device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.input_type = input_type

    def generate(self, prompt, image=None):
        # HF text-only models don't support images; log warning if image provided with text+image mode
        if image is not None and self.input_type == 'text+image':
            print(f"Warning: HF text-only model {self.model_name} does not support images; ignoring image input.")
        out = self.pipeline(prompt, max_new_tokens=self.max_new_tokens, do_sample=(self.temperature>0.0), temperature=self.temperature, truncation=True)
        # pipeline returns list of dicts with 'generated_text'
        return out[0]['generated_text'], 'None'


class OpenAIModel:
    def __init__(self, model_name, input_type='text-only', temperature=0.0):
        if openai is None:
            raise RuntimeError('openai package not installed')
        self.model_name = model_name
        self.input_type = input_type
        self.temperature = temperature
        # support both old and new openai python interfaces
        if hasattr(openai, 'OpenAI'):
            # new client
            try:
                self.client = openai.OpenAI()
            except Exception:
                self.client = None
        else:
            self.client = None

    def _encode_image_base64(self, image):
        """Encode PIL image or image path to base64."""
        import base64
        import io
        if isinstance(image, str):
            # image path
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        else:
            # PIL Image
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            return base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    def _extract_content(self, resp):
        # resp may be dict-like or object with attributes
        try:
            # dict-like
            return resp.choices[0].message.content
        except Exception:
            try:
                return resp.choices[0].message.content
            except Exception:
                return str(resp)

    def _extract_logprobs(self, resp):
        """Extract logprobs as a list of float values from response."""
        try:
            tokens = resp.choices[0].logprobs.content
            if tokens is not None:
                return [t.logprob for t in tokens]
        except Exception:
            pass
        try:
            tokens = resp.choices[0].logprobs.content
            if tokens is not None:
                return [t['logprob'] if isinstance(t, dict) else t.logprob for t in tokens]
        except Exception:
            pass
        return 'None'

    def generate(self, prompt, image=None):
        # Build message content
        content = []
        content.append({"type": "text", "text": prompt})

        # Add image if provided and input_type is text+image
        if image is not None and self.input_type == 'text+image':
            try:
                img_base64 = self._encode_image_base64(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            except Exception as e:
                print(f"Warning: Failed to encode image: {e}")

        # prefer new client if available
        if getattr(self, 'client', None) is not None:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                logprobs=True
            )
            return self._extract_content(resp), self._extract_logprobs(resp)
        else:
            # fallback to old module-level API if present
            if hasattr(openai, 'ChatCompletion'):
                resp = client.chat.completions.create(model=self.model_name,
                messages=[{"role":"user","content":content}],
                temperature=self.temperature,
                logprobs=True)
                return self._extract_content(resp), self._extract_logprobs(resp)
            raise RuntimeError('OpenAI client not available; please install openai>=1.0.0 and set up OpenAI(), or install older openai==0.28')


class AdapterModel:
    def __init__(self, model_name, input_type='text-only', temperature=0.0):
        if QueryModel is None:
            raise RuntimeError('QueryModel adapter not available')
        self.qm = QueryModel(model_name, seed=0, query_config={'temperature': temperature})
        self.input_type = input_type

    def generate(self, prompt, image=None):
        if image is not None and self.input_type == 'text+image':
            resp = self.qm.query(prompt, image=image)
        else:
            resp = self.qm.query(prompt)
        text = resp.get('response') or resp.get('output') or ''
        logprobs_raw = resp.get('logprobs', 'None')
        # Try to parse adapter logprobs into list of floats
        logprobs = 'None'
        if isinstance(logprobs_raw, list):
            try:
                logprobs = [float(lp) if not isinstance(lp, dict) else float(lp.get('logprob', lp.get('token_logprob', 0))) for lp in logprobs_raw]
            except Exception:
                logprobs = 'None'
        elif isinstance(logprobs_raw, str) and logprobs_raw != 'None':
            try:
                import ast
                parsed = ast.literal_eval(logprobs_raw)
                if isinstance(parsed, list):
                    logprobs = [float(lp) if not isinstance(lp, dict) else float(lp.get('logprob', lp.get('token_logprob', 0))) for lp in parsed]
            except Exception:
                logprobs = 'None'
        return text, logprobs


def run_eval(ds_name, model_type, model_name, max_samples=200, out_file='eval_results.jsonl', model_kwargs=None, seed=0, subset='', input_type='text-only', text_type_filter='', use_original_question=False):
    # load dataset
    if '/' in ds_name and ds_name.startswith('lmms-lab'):
        # DocVQA has subtask when using load_dataset('lmms-lab/DocVQA', 'DocVQA')
        parts = ds_name.split('/')
        if len(parts) >= 3:
            dataset_id = '/'.join(parts[:2])
            subset = parts[2]
            ds = load_dataset(dataset_id, subset)
            split = 'validation' if 'validation' in ds else 'test' if 'test' in ds else list(ds.keys())[0]
            ds = ds[split]
        else:
            ds = load_dataset(ds_name)
    else:
        try:
            ds = load_dataset(ds_name)
            # if the dataset returns a dict of splits, try to select appropriate split
            if isinstance(ds, dict):
                if subset and subset in ds:
                    ds = ds[subset]
                elif 'validation' in ds:
                    ds = ds['validation']
                elif 'test' in ds:
                    ds = ds['test']
                else:
                    # pick first split if only one present or none of the common names
                    first_key = list(ds.keys())[0]
                    ds = ds[first_key]
        except Exception:
            # try load by repo/subset format
            if subset:
                ds = load_dataset(ds_name, subset)
            else:
                ds = load_dataset(ds_name)

    # Filter by text_type if specified
    if text_type_filter:
        def filter_by_text_type(sample):
            for k in ['text_type', 'intext_type', 'text_kind']:
                if k in sample and sample[k] == text_type_filter:
                    return True
            return False
        ds = ds.filter(filter_by_text_type)

    # sample
    total = len(ds)
    max_samples = min(max_samples, total)
    ds = ds.shuffle(seed=seed).select(range(max_samples))

    # init model
    if model_type == 'hf':
        model = HFTextGenModel(model_name, **(model_kwargs or {}), input_type=input_type)
    elif model_type == 'openai':
        model = OpenAIModel(model_name, input_type=input_type, temperature=model_kwargs.get('temperature', 0.0))
    elif model_type == 'adapter':
        model = AdapterModel(model_name, input_type=input_type, temperature=model_kwargs.get('temperature', 0.0))
    else:
        raise ValueError('unknown model_type')

    # Generate output filename based on parameters if using default out_file
    if out_file == 'eval_outputs.jsonl':
        # Create results directory with model-based subdirectories
        results_dir = 'results'

        # Clean model name for directory (remove special chars)
        model_name_clean = model_name.replace('/', '_').replace(':', '_')
        model_dir = os.path.join(results_dir, model_name_clean)
        os.makedirs(model_dir, exist_ok=True)

        # Build filename: subset_input_type_samples_seed_text_type.jsonl
        subset_str = f"{subset}_" if subset else ""
        # Use 'original' for text_type when using original question
        if use_original_question:
            text_type_str = 'original'
        else:
            text_type_str = text_type_filter if text_type_filter else 'all'
        filename = f"{subset_str}{input_type.replace('+', 'plus')}_{max_samples}samples_seed{seed}_{text_type_str}.jsonl"
        out_file = os.path.join(model_dir, filename)

    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)

    results = []
    with open(out_file, 'w') as outf:
        preds_for_docvqa = []
        golds_for_docvqa = []
        qids_for_docvqa = []

        for item in tqdm(ds, total=max_samples):
            qid, question, answers, image, full_prompt, text_type, additional_info = detect_fields(item)
            if question is None:
                # try to stringify item
                question = str(item)
            # Choose prompt based on use_original_question flag
            if use_original_question:
                prompt_to_use = question
                # For DocVQA and VQAv2, add instruction to output single word/phrase
                if ('docvqa' in ds_name.lower() or 'vqav2' in ds_name.lower() or
                    (subset and ('docvqa' in subset.lower() or 'vqav2' in subset.lower()))):
                    prompt_to_use = prompt_to_use + " Please only output the answer with a single word or phrase."
            else:
                # Use full_prompt if available, otherwise use question
                prompt_to_use = full_prompt if full_prompt is not None else question
            # prompt directly question — user can customize externally
            logprobs = 'None'
            try:
                result = model.generate(prompt_to_use, image=image)
                if isinstance(result, tuple):
                    pred, logprobs = result
                else:
                    pred = result
            except Exception as e:
                pred = f'Error: {e}'

            # prepare eval item
            eval_item = {
                'question_id': qid if qid is not None else '',
                'question': question,
                'pred_answer': pred,
                'gt_answers': answers if answers is not None else [],
                'full_prompt': full_prompt,
                'text_type': text_type,
                'additional_info': additional_info
            }

            # compute per-sample acc using repo eval_func (works for known datasets)
            try:
                report = eval_func(ds_name.lower(), [eval_item])
                per_acc = report.get('acc', None)
            except Exception:
                per_acc = None

            # fallback using best_subspan_em
            # If the model returned an error string, mark as incorrect
            if isinstance(pred, str) and pred.strip().startswith('Error'):
                is_correct = False
                per_acc = 0.0
            else:
                if per_acc is None:
                    if eval_item['gt_answers']:
                        # First try normalized answer matching (handles number words like "two" vs "2")
                        is_correct = check_answer_match(pred, eval_item['gt_answers'])
                        # If that fails, fall back to substring matching
                        if not is_correct:
                            is_correct = bool(best_subspan_em(pred, eval_item['gt_answers']))
                        per_acc = 1.0 if is_correct else 0.0
                    else:
                        is_correct = False
                        per_acc = 0.0
                else:
                    is_correct = (per_acc > 0)

            out = {
                'question_id': eval_item['question_id'],
                'question': eval_item['question'],
                'gt_answers': eval_item['gt_answers'],
                'pred_answer': pred,
                'is_correct': bool(is_correct),
                'per_acc': float(per_acc if per_acc is not None else 0.0),
                'logprobs': logprobs
            }
            # include extra metadata in output when available
            if eval_item.get('full_prompt') is not None:
                out['full_prompt'] = eval_item.get('full_prompt')
            if eval_item.get('text_type') is not None:
                out['text_type'] = eval_item.get('text_type')
            if eval_item.get('additional_info') is not None:
                out['additional_info'] = eval_item.get('additional_info')

            # collect for DocVQA metric if applicable
            if (subset and 'docvqa' in subset.lower()) or ('docvqa' in str(ds_name).lower()):
                preds_for_docvqa.append(pred)
                # gt answers should be list of strings
                golds_for_docvqa.append(answers if answers is not None else [])
                qids_for_docvqa.append(qid)
            line = json.dumps(out) + '\n'
            for _attempt in range(5):
                try:
                    outf.write(line)
                    outf.flush()
                    break
                except OSError:
                    import time; time.sleep(2)
            results.append(out)

    # final accuracy
    accuracies = [r['per_acc'] for r in results]
    final_acc = sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0.0
    summary = {
        'dataset': ds_name,
        'model_type': model_type,
        'model_name': model_name,
        'num': len(accuracies),
        'accuracy': final_acc
    }
    # If this is DocVQA, compute DocVQA-specific metrics (ANLS/accuracy) using evaluator
    try:
        if (subset and 'docvqa' in subset.lower()) or ('docvqa' in str(ds_name).lower()):
            docvqa_eval = DocVQAEvaluator()
            docvqa_report = docvqa_eval.get_metrics(golds_for_docvqa, preds_for_docvqa)
            summary['docvqa'] = docvqa_report
    except Exception:
        pass
    # save summary
    summary_file = out_file + '.summary.json'
    with open(summary_file, 'w') as sf:
        json.dump(summary, sf, indent=2)

    print('Saved per-sample outputs to', out_file)
    print('Saved summary to', summary_file)
    print('Final accuracy:', final_acc)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--subset', type=str, default='', help='Optional dataset subset/split name (e.g., DocVQA)')
    parser.add_argument('--model_type', type=str, choices=['hf', 'openai', 'adapter'], default='hf')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--input_type', type=str, choices=['text-only', 'text+image'], default='text+image', help='Input modality: text-only or text+image')
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_file', type=str, default='eval_outputs.jsonl')
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--text_type', type=str, default='corrupted', choices=['', 'match', 'irrelevant', 'corrupted'], help='Filter by text type: match, irrelevance, or corruption (empty string for no filter)')
    parser.add_argument('--use_original_question', action='store_true', help='Use only the original question without full prompt or additional context')


    args = parser.parse_args()
    model_kwargs = dict(device=args.device, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    # pass subset through to run_eval so we can select named splits (e.g. 'DocVQA')
    run_eval(args.ds_name, args.model_type, args.model_name, max_samples=args.max_samples, out_file=args.out_file, model_kwargs=model_kwargs, seed=args.seed, subset=args.subset, input_type=args.input_type, text_type_filter=args.text_type, use_original_question=args.use_original_question)
