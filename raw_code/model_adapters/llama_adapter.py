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

class LlamaAdapter(BaseAdapter):
    def __init__(
        self, 
        model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        
        client, tokenizer = model_initialization(model)
        
        
        self.client = client
        
        self.client.generation_config.pad_token_id = tokenizer.pad_token_id
        self.client.config.pad_token_id = client.config.eos_token_id


        self.tokenizer = tokenizer
        
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
        
        tokenizer = self.tokenizer
        device = self.device
        model = self.client
        
        # if image is not None:
        #     prompt = '<image>USER: {}\nASSISTANT:'.format(query)
        # else:
        prompt = 'USER: {}\nASSISTANT:'.format(query)
        
        # if image is not None:
        #     input_obj = processor(text=prompt, return_tensors="pt").to(device)
        # else:
            # print('prompt', prompt)
            # input_obj = processor(text=prompt, images=self.image_holder, return_tensors="pt").to(device)
        input_obj = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = input_obj.input_ids.shape[1]
        
        with torch.inference_mode():
            with torch.no_grad():
                output = model.generate(**input_obj, 
                                        max_length=input_length + 100, 
                                        # temperature=0.0,
                                        do_sample=False,
                                        return_dict_in_generate=True,
                                        output_scores=True,
                                        use_cache=True,
                                        pad_token_id=self.tokenizer.eos_token_id
                                        )
                # generated_token = output[:, input_length:]
                generated_seqs = output.sequences[:, input_length:]
                # generated_seqs = output.sequences
                generated_text = tokenizer.batch_decode(generated_seqs, skip_special_tokens=True)[0]
                
                scores = output.scores
                probabilities = [F.softmax(score, dim=-1) for score in scores]
        
            
        return dict({
            "response": generated_text,
            "logprobs": probabilities
        })

