from litellm import completion
from vllm import LLM, SamplingParams
import time
from vllm.multimodal.image import ImageFeatureData, ImagePixelData

def encode_image_path(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image(image):
    # image is a PIL Image
    import base64
    import io
    
    image_bytes = io.BytesIO()
    # when image is in png format
    # image.save(image_bytes, format="PNG")
    image.save(image_bytes, format="JPEG")
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    
def make_gpt4o_message(text, image_path=None, image=None):
    if image_path is not None:
        base64_image = encode_image_path(image_path)
    elif image is not None:
        base64_image = encode_image(image)
    
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64_image
                    },
                },
            ],
        }
    ]


from model_adapters import *
from openai import OpenAI
import os
class QueryModel:
    def __init__(self, model_name, query_config=None, seed=0) -> None:
        self.model_name = model_name
        self.query_config = query_config
        self.seed = seed
        
        if 'llava' in model_name:
            self.adapter = LlavaAdapter(model=model_name)
        elif 'gpt' in model_name:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.adapter = OpenAIAdapter(model=model_name, client=client)  
        elif 'phi' in model_name.lower():
            self.adapter = PhiAdapter(model=model_name)
        elif 'meta-llama' in model_name and 'vision' in model_name.lower():
            self.adapter = LlamaVisionAdapter(model=model_name)
        elif 'meta-llama' in model_name or 'vicuna' in model_name:
            self.adapter = LlamaAdapter(model=model_name)
        elif 'molmo' in model_name.lower():
            self.adapter = MolmoAdapter(model=model_name)
        elif 'claude' in model_name.lower():
            self.adapter = ClaudeAdapter(model=model_name)
        elif 'qwen' in model_name.lower():
            self.adapter = Qwen2VLAdapter(model=model_name)
    

    def query(self, text, query_config=None, image=None, image_path=None, **kwargs):
        model_name = self.model_name
        
        
        response = self.adapter.generate(text, image, image_path, **kwargs)
        
        return response
    
    def raw_generate(self, messages, **kwargs):
        return self.adapter.raw_generate(messages, **kwargs)
    def construct_messages(self, messages):
        return self.adapter.construct_messages(messages)
    

def get_save_path(args):
    
    ds_name = args['ds_name']
    model_name = args['model_name']
    
    # get current file's directory path's parent directory
    p  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    # p = os.path.dirname(os.path.abspath(__file__))
    
    save_dir = '{}/outputs/{}/{}/'.format(p, ds_name, model_name.replace('/', '-'))
    
    if args.get('sp_task', '') != '':
        save_dir = '{}/{}/'.format(save_dir, args['sp_task'])
    
    save_file = '{}/t={}_seed={}'.format(save_dir, args['t'], args['seed'])

    if args['in_type'] in ['wo-img', 'w-random-img', 'w-noisy-img']:
        save_file = save_file + '_intype={}'.format(args['in_type'])
    if args['intext_type'] != '':
        save_file = save_file + '_intext={}'.format(args['intext_type'])
        
    if 'ocr' in args['intext_type']:
        save_file = save_file + '_ocrparser={}'.format(args['ocr_parser'])
    
    if  args.get('context', '') != '':
        save_file = save_file + '_ctx={}'.format(args['context'])
        # debate source
        # if args['context'] == 'debate' and args['debate_src'] != '':
        #     save_file = save_file + '_debate={}'.format(args['debate_src'])
    
    if args.get('text_source', '') != '':
        save_file = save_file + '_textsrc={}'.format(args['text_source'])
    
    if args.get('text_src_model', '') != '':
        f_model = args['text_src_model'].replace('/', '-').replace('.', '_')
        save_file = save_file + '_textsrcmodel={}'.format(f_model)
    if args.get('desc_model_name', '') != '':
        d_model = args['desc_model_name'].replace('/', '-').replace('.', '_')
        save_file = save_file + '_descmodel={}'.format(d_model)

    if args.get('cap_num', -1) != -1:
        save_file = save_file + '_capnum={}'.format(args['cap_num'])
    
    if args.get('data_seed', 0) != 0:
        save_file = save_file + '_dseed={}'.format(args['data_seed'])

    
    if args.get('note', '') != '':
        save_file = save_file + '_note={}'.format(args['note'])

        
    save_file = save_file + '.jsonl'
    
    return save_file

