import io
import re
import base64

from PIL import Image

from model_adapters import BaseAdapter

from transformers import AutoProcessor, AutoModelForPreTraining, AutoModelForCausalLM, InstructBlipForConditionalGeneration

from .utils import model_initialization
import torch
import torch.nn.functional as F

from typing import Optional

class LlamaVisionAdapter(BaseAdapter):
    def __init__(
        self, 
        model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        
        client, processor = model_initialization(model)
        
        
        self.client = client
        
        # self.client.generation_config.pad_token_id = tokenizer.pad_token_id
        # self.client.config.pad_token_id = client.config.eos_token_id


        self.processor = processor
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # self.image_holder = Image.open('images/image.png')

    def generate(
        self,
        query: str,
        # Optional[image] = None,
        image: Optional[Image.Image] = None,
        image_path: Optional[str] = None,
        **kwargs
    ) -> dict:
        
        processor = self.processor
        device = self.device
        model = self.client
        
        # if image is not None:
        #     prompt = '<image>USER: {}\nASSISTANT:'.format(query)
        # else:
        # prompt = 'USER: {}\nASSISTANT:'.format(query)
        
        # if image is not None:
        #     input_obj = processor(text=prompt, return_tensors="pt").to(device)
        # else:
            # print('prompt', prompt)
            # input_obj = processor(text=prompt, images=self.image_holder, return_tensors="pt").to(device)
        # input_obj = tokenizer(prompt, return_tensors="pt").to(device)
        # input_length = input_obj.input_ids.shape[1]
        
        # image = Image.open(requests.get(url, stream=True).raw)

        
        
        if 'instruct' in self.model.lower():
            
            if image is not None:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": query}
                    ]}
                ]
                
                input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(image, input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
            
            else:
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": query}
                    ]}
                ]
                
                input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
            
        # else:
        #     prompt = "<|image|><|begin_of_text|>{}".format(query)
        #     inputs = processor(image, prompt, return_tensors="pt").to(model.device)
    

                
        with torch.inference_mode():
            with torch.no_grad():
                output = model.generate(**inputs, 
                                        max_length=inputs['input_ids'].shape[1] + 300, 
                                        temperature=0.0,
                                        do_sample=False,
                                        return_dict_in_generate=True,
                                        output_logits=True,
                                        use_cache=True,
                                        # pad_token_id=self.tokenizer.eos_token_id
                                        )
                # generated_token = output[:, input_length:]
                # generated_seqs = output.sequences[:, :]
                generate_ids = output.sequences
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

                # generated_seqs = output.sequences
                generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
                

                logits = output.logits # (seq_len, [batch_size, vocab_size])
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
            "response": generated_text,
            "logprobs": str(out_log_probs)
        })
    def construct_messages(self, messages: list) -> dict:
        return super().construct_messages(messages)
    def raw_generate(self, messages: list, **kwargs) -> dict:
        return super().raw_generate(messages, **kwargs)
