datasets=(
    'lmms-lab/VQAv2'
    # 'lmms-lab/DocVQA/DocVQA'
)

model_names=(
    # 'llava-hf/llava-1.5-7b-hf'
    # 'Qwen/Qwen2-VL-7B-Instruct'
    'llava-hf/llava-v1.6-vicuna-7b-hf'
    # 'microsoft/Phi-3.5-vision-instruct'
    # 'gpt-4o-mini-2024-07-18'
    # 'gpt-4o-2024-05-13'
    # 'meta-llama/Llama-3.2-11B-Vision-Instruct'

    # 'llava-hf/llava-v1.6-mistral-7b-hf'
    # 'microsoft/Phi-3-vision-128k-instruct'
 
    # 'claude-3-haiku-20240307'
    # 'claude-3-5-sonnet-20240620'

    # updated models
    'gpt-4.1-mini-2025-04-14'
    'claude-3-7-sonnet-20250219'


)


prompt='caption+custom-25'


if [ -z "$1" ]; then
    max_samples=1000
else
    max_samples=$1
fi

for model_name in "${model_names[@]}"; do
for ds_name in "${datasets[@]}"; do

        CUDA_VISIBLE_DEVICES=0 python vlm_test.py --ds_name $ds_name --model_name $model_name --max_samples ${max_samples} &
        CUDA_VISIBLE_DEVICES=1 python vlm_test.py --ds_name $ds_name --model_name $model_name --in_type wo-img --intext_type ${prompt} --max_samples ${max_samples} --cap_num 1  --text_source contradict_stat --text_src_model gpt-4o-2024-05-13
        CUDA_VISIBLE_DEVICES=2 python vlm_test.py --ds_name $ds_name --model_name $model_name --in_type w-img --intext_type ${prompt} --max_samples ${max_samples} --cap_num 1 --text_source contradict_stat --text_src_model gpt-4o-2024-05-13
        CUDA_VISIBLE_DEVICES=3 python vlm_test.py --ds_name $ds_name --model_name $model_name --in_type wo-img --intext_type ${prompt} --max_samples ${max_samples} --cap_num 1  --text_source supp_stat --text_src_model gpt-4o-2024-05-13 
        CUDA_VISIBLE_DEVICES=0 python vlm_test.py --ds_name $ds_name --model_name $model_name --in_type w-img --intext_type ${prompt} --max_samples ${max_samples} --cap_num 1 --text_source supp_stat --text_src_model gpt-4o-2024-05-13 &
        CUDA_VISIBLE_DEVICES=1 python vlm_test.py --ds_name $ds_name --model_name $model_name --in_type wo-img --intext_type ${prompt} --max_samples ${max_samples} --text_source irrel_wikitext &
        CUDA_VISIBLE_DEVICES=2 python vlm_test.py  --ds_name $ds_name --model_name $model_name --in_type w-img --intext_type ${prompt} --max_samples ${max_samples} --text_source irrel_wikitext 


done
done

