from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor, GenerationConfig

from model_adapters import BaseAdapter

from typing import Any

import torch.nn.functional as F
import torch

class MolmoAdapter(BaseAdapter):
    def __init__(self, model: str):
        # self.model = model
        # # self.device = device
        # self.model = AutoModelForCausalLM.from_pretrained(model, device_map="cuda", trust_remote_code=True, torch_dtype="auto")
        
        # if '3.5' in model:
        #     self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, num_crops=16)
        # else:
        #     self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True) 
            
        # load the processor
        self.processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            # torch_dtype='auto',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map='auto'
        )

        # to reduce the memory usage
        # self.model.to(dtype=torch.bfloat16)

        
        self.generation_args = { 
            "max_new_tokens": 500, 
            # "temperature": 0.0, 
            "do_sample": False, 
            "return_dict_in_generate": True,
            # "output_scores": True,
            "output_logits": True,
            "stop_strings": "<|endoftext|>"
        } 

        
    def generate(self, query: str, image: Any, image_path: str, **kwargs) -> dict:
        
        processor = self.processor
        
        # if image is not None:
        #     content = "<|image_1|>\n" + query
        # else:
        #     content = query
        # messages = [ 
        #     {"role": "user", "content": content,} 
        # ]
        
        # prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # if image is not None:
        #     inputs = processor(prompt, [image], return_tensors="pt").to("cuda")
        # else:
        #     inputs = processor(prompt, return_tensors="pt").to("cuda")
        
        if image is not None:
            inputs = processor.process(
                images=[image],
                text=query,
            )
        else:
            inputs = processor.process(
                text=query,
            )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}


        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16): # for efficient inference
            outputs = self.model.generate_from_batch(inputs, GenerationConfig(**self.generation_args), tokenizer=processor.tokenizer, ) 

        generate_ids = outputs.sequences
        # scores = outputs.scores
        # probabilities = [F.softmax(score, dim=-1) for score in scores]
        
        # remove input tokens 
        generate_ids = generate_ids[0, inputs['input_ids'].shape[1]:]
        response = processor.tokenizer.decode(generate_ids, skip_special_tokens=True)

        
        
        logits = outputs.logits # (seq_len, [batch_size, vocab_size])
        # print(len(logits), logits[0].shape)
        # based on generated token idx
        token_probs = []
        token_ids = []
        for i in range(len(generate_ids)):
            token_probs.append(F.softmax(logits[i][0], dim=-1)[generate_ids[i]])
            token_ids.append({
                'token_id': generate_ids[i].item(),
                'decoded': processor.tokenizer.decode(generate_ids[i], skip_special_tokens=True)
            })
        
        # token_probs = F.softmax(logits, dim=-1)
        # based on generated token idx
        # token in generate_ids as idx
        # token_probs = token_probs[range(token_probs.shape[0]), generate_ids]
        token_probs = torch.tensor(token_probs)
        log_probs = token_probs.log().tolist()
        
        out_log_probs = []
        for i in range(len(generate_ids)):
            out_log_probs.append({
                'id': token_ids[i]['token_id'],
                'token': token_ids[i]['decoded'],
                'logprob': log_probs[i],
            })
        


        return dict({
            "response": response,
            "logprobs": str(out_log_probs)
        })
    def construct_messages(self, messages: list) -> dict:
        return super().construct_messages(messages)
    def raw_generate(self, messages: list, **kwargs) -> dict:
        return super().raw_generate(messages, **kwargs)


