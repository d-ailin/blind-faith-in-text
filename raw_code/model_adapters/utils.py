from transformers import AutoProcessor, AutoModelForPreTraining, AutoModelForCausalLM, InstructBlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, MllamaForConditionalGeneration
import torch

# MODEL_NAME = {
#     "llava-1.5-7b-hf": "llava-hf/llava-1.5-7b-hf",
#     "llava-1.5-13b-hf": "llava-hf/llava-1.5-13b-hf",
#     "bakLlava": "llava-hf/bakLlava-v1-hf",
#     "fuyu-8b": "adept/fuyu-8b",
#     "InstructBLIP-7b": "Salesforce/instructblip-vicuna-7b",
# }

MODEL_LIST = [
    'llava-hf/llava-1.5-7b-hf',
    'llava-hf/llava-1.5-13b-hf',
    'llava-hf/bakLlava-v1-hf',
    'adept/fuyu-8b',
    'Salesforce/instructblip-vicuna-7b',
    'llava-hf/llava-v1.6-vicuna-7b-hf',
    'llava-hf/llava-v1.6-vicuna-13b-hf',
    'meta-llama/Meta-Llama-3-8B-Instruct',
    
    'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava-hf/llama3-llava-next-8b-hf',
    'llava-hf/llava-v1.6-34b-hf',
    
    'HuggingFaceM4/idefics2-8b',
    'Qwen/Qwen2-VL-7B-Instruct',
    
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    'meta-llama/Llama-3.2-11B-Vision',
    # fine-tuned models
    'model_ckpts/dpo_llava1.5-7b_rlaif-v',
    
    
]

def model_initialization(model, device='cuda:0'):    
    assert model in MODEL_LIST or 'model_ckpts' in model , f"Model {model} not found in the list of models"
    if 'model_ckpts' in model:
        processor = None
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model)
    
        
    if 'fuyu' in model:
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16).to(device)
    elif 'llava-1.5-7b-hf' in model or 'llava-v1.6-vicuna-7b-hf' in model or 'llava-v1.6-mistral-7b-hf' in model or 'llama3-llava-next-8b-hf' in model:
        # model = AutoModelForPreTraining.from_pretrained(model, torch_dtype=torch.bfloat16, device_map='auto').to(device)
        model = AutoModelForPreTraining.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
    elif 'llava-1.5-13b-hf' in model or 'bakLlava' in model or 'llava-v1.6-vicuna-13b-hf' in model or 'llava-v1.6-34b-hf' in model:
        model = AutoModelForPreTraining.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='auto',
            trust_remote_code=True)
    elif 'InstructBLIP' in model:
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model, 
            torch_dtype=torch.float16, 
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True            
            )
    elif 'Meta-Llama' in model and 'vision' not in model.lower():
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16).to(device)
        # processor = AutoTokenizer.from_pretrained(model)
    elif 'meta-llama' in model and 'vision' in model.lower():
        model = MllamaForConditionalGeneration.from_pretrained(model, torch_dtype=torch.float16).to(device)
        # model = MllamaForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
    elif 'model_ckpts' in model:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        output_dir = model
        model = AutoModelForVision2Seq.from_pretrained(output_dir, torch_dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(output_dir)
        print('Model loaded from:', output_dir)
               
    model.eval()
    return model, processor