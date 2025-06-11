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

class LlavaAdapter(BaseAdapter):
    def __init__(
        self, 
        model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        
        client, processor = model_initialization(model)
        
        self.client = client
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
        
        if image is not None:
            # if ('v1.6' in self.model or '1.5' in self.model) and ('model_ckpts' not in self.model):
            if ('1.6' in self.model or '1.5' in self.model):
                conversation = [
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image"},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            else:
                prompt = '<image>USER: {}\nASSISTANT:'.format(query)
        else:
            # if ('v1.6' in self.model or '1.5' in self.model) and ('model_ckpts' not in self.model):
            if ('1.6' in self.model or '1.5' in self.model):
                conversation = [
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            else:
                prompt = 'USER: {}\nASSISTANT:'.format(query)
        
        if image is not None:
            input_obj = processor(text=prompt, images=image, return_tensors="pt").to(device)
        else:
            # print('prompt', prompt)
            # input_obj = processor(text=prompt, images=self.image_holder, return_tensors="pt").to(device)
            input_obj = processor(text=prompt, return_tensors="pt").to(device)
        input_length = input_obj.input_ids.shape[1]
        
        with torch.inference_mode():
            with torch.no_grad():
                output = model.generate(**input_obj, 
                                        max_length=input_length + 100, 
                                        # temperature=0.0,
                                        do_sample=False,
                                        return_dict_in_generate=True,
                                        # output_scores=True,
                                        output_logits=True,
                                        use_cache=True,
                                        )
                # generated_token = output[:, input_length:]
                generated_seqs = output.sequences[:, input_length:]
                generated_text = processor.batch_decode(generated_seqs, skip_special_tokens=True)[0]
                
                # scores = output.scores
                # probabilities = [F.softmax(score, dim=-1) for score in scores]

                logits = output.logits # (seq_len, [batch_size, vocab_size])
                # print(len(logits), logits[0].shape)
                # based on generated token idx
                token_probs = []
                token_ids = []
                for i in range(len(generated_seqs[0])):
                    token_probs.append(F.softmax(logits[i][0], dim=-1)[generated_seqs[0][i]])
                    token_ids.append({
                        'token_id': generated_seqs[0][i].item(),
                        'decoded': processor.decode(generated_seqs[0][i], skip_special_tokens=True)
                    })
                
                # token_probs = F.softmax(logits, dim=-1)
                # based on generated token idx
                # token in generated_seqs as idx
                # token_probs = token_probs[range(token_probs.shape[0]), generated_seqs]
                token_probs = torch.tensor(token_probs)
                log_probs = token_probs.log().tolist()
                
                out_log_probs = []
                for i in range(len(generated_seqs[0])):
                    out_log_probs.append({
                        'id': token_ids[i]['token_id'],
                        'token': token_ids[i]['decoded'],
                        'logprob': log_probs[i],
                    })
                
                # probabilities = [F.softmax(score, dim=-1) for score in logits]
        
            
        return dict({
            "response": generated_text,
            "logprobs": str(out_log_probs),
        })
        
    def construct_messages(self, messages: list) -> dict:
        return super().construct_messages(messages)
    def raw_generate(self, messages: list, **kwargs) -> dict:
        return super().raw_generate(messages, **kwargs)

