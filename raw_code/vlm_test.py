from datasets import load_dataset
import os
from utils import QueryModel, get_save_path
import pathlib
import json
from tqdm import tqdm
from PIL import Image
import argparse
import numpy as np
import random
from data_utils import suffix_irrevelant_data, get_question_prompt, add_noise_to_img, load_coco_captions, suffix_caption_data, get_logprobs_from_resp, get_confidence, load_text_source, suffix_data
import io
import time



parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str, default='facebook/textvqa')
parser.add_argument('--model_name', type=str, default='llava-hf/llava-1.5-7b-hf')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--t', type=int, default=0)
parser.add_argument('--in_type', type=str, default='w-img', choices=['w-img', 'wo-img', 'w-random-img', 'w-noisy-img'])
parser.add_argument('--intext_type', type=str, default='', choices=['', 'irrelevant', 'ocr', 'search', 'wrong-ocr', 'ocr+search', 'oracle', 'noisy-ocr', 
                                                                    'ocr+custom-1', 'ocr+custom-2', 'ocr+custom-3', 'ocr+custom-4', 'ocr+custom-5', 'ocr+custom-6', 'ocr+custom-10', 'ocr+custom-11', 'ocr+custom-12', 'ocr+custom-13', 'ocr+custom-14', 'ocr+custom-15', 'ocr+custom-16', 'ocr+custom-17', 'ocr+custom-18', 'ocr+custom-19', 'ocr+custom-20', 'ocr+custom-21', 'ocr+custom-22',
                                                                    'ocr+custom-24',
                                                                    'caption', 'caption+custom-1', 'caption+custom-3', 'caption+custom-4', 'caption+custom-5', 'caption+custom-6', 'caption+custom-7', 'caption+custom-8', 'caption+custom-9', 'caption+custom-10', 'caption+custom-11', 'caption+custom-12', 'caption+custom-13', 'caption+custom-14', 'caption+custom-15', 'caption+custom-16', 'caption+custom-17', 'caption+custom-18', 'caption+custom-19', 'caption+custom-20', 'caption+custom-21', 'caption+custom-22', 'caption+custom-23',
                                                                    'caption+custom-24', 'caption+custom-25', 'caption+custom-26', 'caption+custom-27', 'caption+custom-28', 'caption+custom-29', 'caption+custom-30',
                                                                    'caption+custom-irrel-1', 'caption+custom-irrel-2'])
parser.add_argument('--max_samples', type=int, default=2000) 
parser.add_argument('--note', type=str, default='')
parser.add_argument('--cap_num', type=int, default=-1)
parser.add_argument('--text_source', type=str, default='')
parser.add_argument('--desc_model_name', type=str, default='')
parser.add_argument('--text_src_model', type=str, default='')
parser.add_argument('--data_seed', type=int, default=0)
parser.add_argument('--sp_task', type=str, default='')

args = parser.parse_args()

args = vars(args)

ds_name = args['ds_name']
model_name = args['model_name']

# set seed
random.seed(args['seed'])



save_file = get_save_path(args)
save_dir = os.path.dirname(save_file)


if 'lmms-lab/DocVQA' in ds_name:
    sub_task = ds_name.split('/')[-1]
    _ds_name = 'lmms-lab/DocVQA'
    ds = load_dataset(_ds_name, sub_task)
    ds = ds['validation']
    
    if args['text_source'] != '':
        text_source_data = load_text_source(args)

elif 'lmms-lab/VQAv2' in ds_name:
    ds = load_dataset(ds_name)
    ds = ds['validation']

    if args['text_source'] != '':
        text_source_data = load_text_source(args)
    else:
        captions_map = load_coco_captions()

    
# only take max_samples with random seed
max_samples = min(args['max_samples'], len(ds))
ds = ds.shuffle(seed=args['seed']).select(range(max_samples))



tested_question_ids = []
if os.path.exists(save_file):
    print(save_file)
    print('File already exists')
    
    # load the file and check how many data
    with open(save_file, 'r') as f:
        lines = f.readlines()
        print('Number of data: ', len(lines))
        
        if len(lines) >= args['max_samples']:
            print('Already have enough data')
            exit()
        else:
            print('Continue to test more data')
            tested_data = [json.loads(line) for line in lines]
            tested_question_ids = []
            
            for data in tested_data:
                tested_question_ids.append(data['question_id'])


# if all data is tested, then exit
if len(tested_question_ids) == max_samples:
    print('All data is tested')
    exit()



text_conf_data = None

            
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)


query_config = {
    "temperature": 0.0
}

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "lmsys/vicuna-7b-v1.5"
# model_name = 'llava-hf/llava-1.5-7b-hf'
# model_name = 'llava-hf/llava-v1.6-vicuna-7b-hf'


llm = QueryModel(model_name, seed=0, query_config=query_config)



for i, data in enumerate(tqdm(ds)):

    _question = None
    
    if 'lmms-lab/DocVQA' in ds_name:
        question_id = '{}/{}'.format(data['questionId'], str(data['question_types']))
        answers = data['answers']
        image = data['image']
        ocr_image = np.array(image)
        
        question = data['question']
        
        question = get_question_prompt(ds_name, question)
        
        if args['text_source'] != '':
            captions = text_source_data.get(question_id, [])
            
        
    elif 'lmms-lab/VQAv2' in ds_name:
        question_id = data['question_id']
        question = data['question']
        answers = [data['multiple_choice_answer']]
        image = data['image']
        image_id = data['image_id']
        ocr_image = np.array(image)
        
        question = get_question_prompt(ds_name, question)
        
        if args['text_source'] != '':
            captions = text_source_data.get(question_id, [])
        else:
            captions = captions_map.get(image_id, [])
    
    elif 'infoseek' in ds_name:
        question_id = data['question_id']
        question = data['question']
        answers = data['answers']
        image = data['image']
        ocr_image = np.array(image)
        
        _question = question
        
        question = get_question_prompt(ds_name, question)
        
        if args['text_source'] != '':
            captions = text_source_data.get(question_id, [])
        else:
            captions = []
        
        
    if question_id in tested_question_ids:
        # print('Already tested this question_id')
        continue
    
    # question = get_question_prompt(ds_name, question)
    tpl_params = {
    }
    if text_conf_data is not None:
        tpl_params['text_conf'] = text_conf_data.get(question_id, None).get('confidence', 0.0)
        tpl_params['scaled_text_conf'] = text_conf_data.get(question_id, None).get('scaled_confidence', 0.0)
    
    

    if args['intext_type'] == 'irrelevant':
        question = suffix_irrevelant_data(question)

    elif 'caption' in args['intext_type']:
        # caption = random.choice(captions)
        
        assert len(captions) > 0, 'No caption available'
        
        if args['cap_num'] != -1:
            captions = captions[:args['cap_num']]
        
        if 'custom' in args['intext_type']:
            template = args['intext_type'].split('+')[-1]
            
            question = suffix_caption_data(question, captions, template=template, tpl_params=tpl_params)
        else:
            question = suffix_caption_data(question, captions)
    

    
    if args['in_type'] == 'wo-img':
        ans = llm.query(question)        
    else:        
        image_path = None
        ans = llm.query(question, image=image, image_path=image_path)
        
    response = ans['response']
    logprobs = ans['logprobs']
    
    with open(save_file, 'a') as f:
        if logprobs is None:
            json.dump({'question': data.get('question', _question) , 'question_id': question_id, 'input': question, 'output': response, 'answers': answers}, f)
        else:
            json.dump({'question': data.get('question', _question), 'question_id': question_id, 'input': question, 'output': response, 'answers': answers, 'logprobs': logprobs}, f)
        
        f.write('\n')
