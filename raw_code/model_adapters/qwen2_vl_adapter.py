from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from transformers import Qwen2VLForConditionalGeneration


from model_adapters import BaseAdapter

from typing import Any

import torch.nn.functional as F
import torch

class Qwen2VLAdapter(BaseAdapter):
    def __init__(self, model: str):
        self.model = model
        # self.device = device
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(model, trust_remote_code=True, torch_dtype="auto", device_map="cuda")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model, trust_remote_code=True, 
                                                                     torch_dtype="auto", 
                                                                    #  torch_dtype=torch.bfloat16,
                                                                    low_cpu_mem_usage=True,
                                                                     device_map="auto")
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, min_pixels=min_pixels, max_pixels=max_pixels) 
        
        self.generation_args = { 
            "max_new_tokens": 200, 
            # "temperature": 0.0, 
            "do_sample": False, 
            "return_dict_in_generate": True,
            # "output_scores": True,
            "output_logits": True,
        } 

        
    def generate(self, query: str, image: Any, image_path: str, **kwargs) -> dict:
        
        processor = self.processor
        
        if image is not None:
            # content = [
            #     {"type": "text", "text": query},
            #     {"type": "image"},
            # ]
            
            content = [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": query},
            ]
        else:
            content = [
                {"type": "text", "text": query},
            ]
        messages = [ 
            {"role": "user", "content": content,} 
        ]
        
        # https://huggingface.co/HuggingFaceM4/idefics2-8b/blob/main/processor_config.json
        # processor.chat_template = "{% for message in messages %}{{message['role'].capitalize()}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        # at inference time, one needs to pass `add_generation_prompt=True` in order to make sure the model completes the prompt
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        if image is not None:
            inputs = processor(images=[image], text=text, return_tensors="pt").to("cuda")
        else:
            inputs = processor(text=text, return_tensors="pt").to("cuda")

   
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


