# Words or Vision

**Words or Vision: Do Vision-Language Models Have Blind Faith in Text?** [[Paper](https://arxiv.org/abs/2503.02199)]

[Ailin Deng](https://d-ailin.github.io), [Tri Cao](https://caothientri2001vn.github.io/), [Zhirui Chen](https://zchen42.github.io/), [Bryan Hooi](https://bhooi.github.io/)

CVPR 2025

--------

## Data

The evaluation data is available on [Hugging Face](https://huggingface.co/datasets/dal-289/word_or_vision).

The dataset contains VQA samples with three text-image conditions:
- **`corrupted`** — text in the image is corrupted/misleading relative to the visual content
- **`match`** — text in the image matches the correct answer
- **`irrelevant`** — text in the image is irrelevant to the question

--------

## Setup

```bash
cd raw_code
pip install -r requirements.txt
```

For multimodal adapter backends (LLaVA, InternVL, Qwen-VL, Molmo, Phi, etc.), install the model-specific packages as needed.

--------

## Evaluation

All evaluation commands are run from the `raw_code/` directory.

### Quick start with `hf_evaluator.py`

`hf_evaluator.py` is the main evaluation script. It loads the dataset from HuggingFace and evaluates a model, saving per-sample JSONL outputs and an accuracy summary.

**Key arguments:**

| Argument | Description |
|---|---|
| `--ds_name` | HuggingFace dataset name (e.g. `dal-289/word_or_vision`) |
| `--subset` | Dataset subset/split (e.g. `VQAv2`, `DocVQA`) |
| `--model_type` | `adapter` (repo multimodal adapters), `openai` (OpenAI API), or `hf` (HuggingFace text-gen) |
| `--model_name` | Model name or HuggingFace model ID |
| `--input_type` | `text+image` (default) or `text-only` |
| `--text_type` | Filter by condition: `corrupted` (default), `match`, `irrelevant`, or `""` for all |
| `--max_samples` | Number of samples to evaluate |
| `--use_original_question` | Use original question without dataset-specific prompt |
| `--out_file` | Output JSONL file path (auto-named by default) |

---

### Examples

```bash
# Open-source VLM via adapter backend (Qwen-VL, InternVL, LLaVA, Molmo, Phi, etc.)
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 \
    --model_type adapter --model_name Qwen/Qwen3-VL-2B-Instruct \
    --input_type text+image --text_type corrupted --max_samples 100 --seed 0

# OpenAI model (requires OPENAI_API_KEY)
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 \
    --model_type openai --model_name gpt-4o \
    --input_type text+image --text_type corrupted --max_samples 100 --seed 0
```

For the full evaluation matrix (all text-type conditions × input modalities × models), see [`eval.sh`](raw_code/eval.sh).

---

### Output format

Results are saved under `results/<model_name>/`:
- **JSONL file** — per-sample records with fields: `question_id`, `question`, `gt_answers`, `pred_answer`, `is_correct`, `per_acc`, `logprobs`, `text_type`, `full_prompt`
- **Summary JSON** — overall accuracy and dataset/model metadata

---

### Running multiple conditions at once

See [`eval.sh`](raw_code/eval.sh) for batch evaluation scripts covering all three text-type conditions and both input modalities across multiple models.

--------

## Model Checkpoints

We provide the finetuned model checkpoints via [Google Drive](https://drive.google.com/drive/folders/11gETrEJHnxYJWWVtLWBN60IkZ5jP9AoP).

--------

## Analysis

After running evaluation, the following notebook reproduces the paper's analysis — computing accuracy per condition, text ratios, and comparative bar plots across models:

[`raw_code/notebooks/analyze_text_ratios.ipynb`](raw_code/notebooks/analyze_text_ratios.ipynb)

--------

## Citation

```bibtex
@inproceedings{deng2025wordsorvision,
  title={Words or Vision: Do Vision-Language Models Have Blind Faith in Text?},
  author={Deng, Ailin and Cao, Tri and Chen, Zhirui and Hooi, Bryan},
  booktitle={CVPR},
  year={2025}
}
```
