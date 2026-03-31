# ===== Qwen/Qwen3-VL-2B-Instruct =====
# -- corrupted (default) --
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type adapter --model_name Qwen/Qwen3-VL-2B-Instruct --max_samples 100 --input_type text-only --use_original_question --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type adapter --model_name Qwen/Qwen3-VL-2B-Instruct --max_samples 100 --input_type text-only --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type adapter --model_name Qwen/Qwen3-VL-2B-Instruct --max_samples 100 --use_original_question --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type adapter --model_name Qwen/Qwen3-VL-2B-Instruct --max_samples 100 --seed 0
# -- match --
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type adapter --model_name Qwen/Qwen3-VL-2B-Instruct --max_samples 100 --input_type text-only --text_type match --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type adapter --model_name Qwen/Qwen3-VL-2B-Instruct --max_samples 100 --text_type match --seed 0
# -- irrelevant --
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type adapter --model_name Qwen/Qwen3-VL-2B-Instruct --max_samples 100 --input_type text-only --text_type irrelevant --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type adapter --model_name Qwen/Qwen3-VL-2B-Instruct --max_samples 100 --text_type irrelevant --seed 0


# # ===== gpt-4o =====
# -- corrupted (default) --
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type openai --model_name gpt-4o --max_samples 100 --input_type text-only --use_original_question --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type openai --model_name gpt-4o --max_samples 100 --input_type text-only --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type openai --model_name gpt-4o --max_samples 100 --use_original_question --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type openai --model_name gpt-4o --max_samples 100 --seed 0
# -- match --
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type openai --model_name gpt-4o --max_samples 100 --input_type text-only --text_type match --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type openai --model_name gpt-4o --max_samples 100 --text_type match --seed 0
# -- irrelevant --
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type openai --model_name gpt-4o --max_samples 100 --input_type text-only --text_type irrelevant --seed 0
python hf_evaluator.py --ds_name dal-289/word_or_vision --subset VQAv2 --model_type openai --model_name gpt-4o --max_samples 100 --text_type irrelevant --seed 0

