from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

from model_adapters import BaseAdapter

from typing import Any

import torch.nn.functional as F
import torch

class PhiAdapter(BaseAdapter):
    def __init__(self, model: str):
        self.model = model
        # self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2')
        
        if '3.5' in model:
            self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, num_crops=16)
        else:
            self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True) 
        
        self.generation_args = { 
            "max_new_tokens": 500, 
            # "temperature": 0.0, 
            "do_sample": False, 
            "return_dict_in_generate": True,
            # "output_scores": True,
            "output_logits": True,
        } 

        
    def generate(self, query: str, image: Any, image_path: str, **kwargs) -> dict:
        
        processor = self.processor
        
        if image is not None:
            # content = "<|image_1|>\n" + query
            # content = query + "<|image_1|>"
            content = "<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n".format(prompt=query)
            # content = "<|user|>{prompt}\n<|image_1|>\n<|end|>\n<|assistant|>\n".format(prompt=query) # order test
        else:
            # content = query
            content = "<|user|>{prompt}\n<|end|>\n<|assistant|>\n".format(prompt=query)
            # content = "<|user|>{prompt}\n<|end|>\n<|assistant|>\n".format(prompt=query) # order test
        messages = [ 
            {"role": "user", "content": content,} 
        ]
        
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if image is not None:
            inputs = processor(prompt, [image], return_tensors="pt").to("cuda")
        else:
            inputs = processor(prompt, return_tensors="pt").to("cuda")
            
        outputs = self.model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **self.generation_args) 

        generate_ids = outputs.sequences
        # scores = outputs.scores
        # probabilities = [F.softmax(score, dim=-1) for score in scores]
        
        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        
        
        logits = outputs.logits # (seq_len, [batch_size, vocab_size])
        # print(len(logits), logits[0].shape)
        # based on generated token idx
        token_probs = []
        token_ids = []
        for i in range(len(generate_ids[0])):
            token_probs.append(F.softmax(logits[i][0], dim=-1)[generate_ids[0][i]])
            token_ids.append({
                'token_id': generate_ids[0][i].item(),
                'decoded': processor.decode(generate_ids[0][i], skip_special_tokens=True)
            })
        
        # token_probs = F.softmax(logits, dim=-1)
        # based on generated token idx
        # token in generate_ids as idx
        # token_probs = token_probs[range(token_probs.shape[0]), generate_ids]
        token_probs = torch.tensor(token_probs)
        log_probs = token_probs.log().tolist()
        
        out_log_probs = []
        for i in range(len(generate_ids[0])):
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


