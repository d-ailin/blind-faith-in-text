

def suffix_irrevelant_data(question):
    
    # suffix_text = '''Here is some information but could be irrelevant to the question.:
    # The image describe a fox in the forest. The fox is looking for food. The fox is hungry. The fox is looking for a rabbit to eat.
    # The image shows a kitchen with a small stove, a sink with two faucets extending from the wall and yellow painted walls and cabinets.
    # '''
    
    suffix_text = '''Here is some information but could be irrelevant to the question.:
    The image describes a fox in the forest. The fox is looking for food. The fox is hungry. The fox is looking for a rabbit to eat.
    '''

    
    final_question = suffix_text + question
    
    return final_question


import re
def clean_ocr_text(text):
    # Replace multiple consecutive newlines with a single newline
    cleaned_text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    # cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Optionally, strip leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    return cleaned_text

from utils.constant import CAPTION_TEMPS
import json
import os

def suffix_data(question, template='', tpl_params={}, template_task='ocr'):
    
    task_temps = {
        'ocr': OCR_TEMPS,
        'caption': CAPTION_TEMPS
    }
    
    if 'custom' in template:
        suffix_text = task_temps[template_task][template].format(**tpl_params)
    else:
        suffix_text = ''
    
    final_question = suffix_text + question
    
    return final_question

def suffix_ans_data(answer, answer_format, ans_type='match'):
    
    if answer_format == 'process_sup':
        mapping = {
            'match': 'The provided text is accurate and provide correct information for answering the question. Given both image and text information, the answer is {}',
            'corrupt': 'The provided text is inaccurate and provide incorrect information for answering the question. We should ignore the incorrect text information and focus more on the image information. Given this, the answer is {}',
            'irrelevant': 'The provided text is irrelevant to the question. We should ignore the text information and only focus on the image information. Given this, the answer is {}'
        }
    
    suffix_text = mapping[ans_type].format(answer)
    
    return suffix_text
    

def suffix_caption_data(question, captions, template='', tpl_params={}):

    # str_captions = '\n'.join(captions)

    if len(captions) > 1:
        str_captions = '\n'
        for i, caption in enumerate(captions):
            str_captions += f'Information {i+1}: {caption}\n'
    else:
        str_captions = '\n'.join(captions)
    
    tpl_params['captions'] = str_captions
    
    if 'custom' in template:
        # suffix_text = CAPTION_TEMPS[template].format(str_captions)
        suffix_text = CAPTION_TEMPS[template].format(**tpl_params)
    else:
        suffix_text = 'Here are image captions:' + str_captions + '\n'
    
    final_question = suffix_text + question
    
    return final_question

def suffix_debate_data(question, question_id, debate_data, template=''):
    if question_id in debate_data:
        debate_answer = debate_data[question_id]['answer']
    
        suffix_text =  '\nThis is the answer from other agent: {}\n'.format(debate_answer)
    else:
        suffix_text = ''
    
    final_question = suffix_text + question
    
    return final_question

# import wikipediaapi

# Initialize the Wikipedia API
# wiki_wiki = wikipediaapi.Wikipedia('en')

# Get a page
# page = wiki_wiki.page("Python (programming language)")

# Check if the page exists
# if page.exists():
#     print("Page - Title: %s" % page.title)
#     print("Page - Summary: %s" % page.summary[:60])
# else:
#     print("The page does not exist.")

import requests
from bs4 import BeautifulSoup


def get_question_prompt(dataset, question):
    if 'textvqa' in dataset:
        # return 'Answer the question using a single word or phrase. ' + question
        return question + ' Answer the question using a single word or phrase.'
        # return 'Could be OCR related. Answer the question using a single word or phrase. ' + question
    elif 'visualwebbench' in dataset:
        # return question # originally
        if 'heading_ocr' in dataset:
            return question + ' Please answer the heading text drectly.'
        elif 'element_ocr' in dataset:
            return question + ' Please answer the text content directly.'
        else:
            return question
    elif 'OCRBench' in dataset:
        # return question # originally
        # return 'Answer the question using a single word or phrase. ' + question
        return question + ' Please only output the answer directly.'
    
    elif 'lmms-lab/DocVQA' in dataset:
        # return 'Answer the question using a single word or phrase. ' + question
        return  question + ' Please only output the answer directly.'
    
    elif 'HuggingFaceM4/ChartQA' in dataset:
        return question + ' Please only output the answer directly.'
    
    elif 'lmms-lab/VQAv2' in dataset:
        # return 'Answer the question using a single word or phrase. ' + question
        # return question + ' Answer the question using a single word or phrase.'
        return question + ' Please only output the answer with a single word or phrase.'
    elif 'infoseek' in dataset:
        return question + ' Please only output the answer directly.'

def get_question_prompt_v2(dataset, question):
    if 'textvqa' in dataset:
        return question
    elif 'visualwebbench' in dataset:
        return question # originally
    elif 'OCRBench' in dataset:
        return question # originally
    elif 'lmms-lab/DocVQA' in dataset:
        return question
    elif 'lmms-lab/VQAv2' in dataset:
        return question
    elif 'infoseek' in dataset:
        return question


def load_coco_captions():
    import json
    with open('/data/ailin/captions_val2014.json', 'r') as f:
        a = json.load(f)

    annots = {}
    for annot in a['annotations']:
        image_id = annot['image_id']
        caption = annot['caption']
        if image_id not in annots:
            annots[image_id] = []
        annots[image_id].append(caption)
    
    return annots

from utils.cor_utils import get_save_path as get_cor_save_path
def load_text_source(args):
    text_source = args.get('text_source', None)
    _args = args.copy()
    
    all_text_source_data = {}
    
    if '+' in text_source:
        text_sources = text_source.split('+')
    else:
        text_sources = [text_source]
        
    for _text_source in text_sources:
    
        _args['instruct'] = _text_source
        if _args.get('text_src_model', '') == '':
            _args['model_name'] = 'gpt-4o-mini-2024-07-18' # the text is modified based on gpt4o
        else:
            _args['model_name'] = _args['text_src_model']
        
        text_source_file = get_cor_save_path(_args)
        text_source_data = {}
        
        for line in open(text_source_file, 'r'):
            one_data = json.loads(line)
            
            if one_data['question_id'] not in text_source_data:
                #
                texts = one_data['response']
                text_source_data[one_data['question_id']] = texts

                if one_data['question_id'] not in all_text_source_data:
                    all_text_source_data[one_data['question_id']] = texts
                else:
                    all_text_source_data[one_data['question_id']].extend(texts)
    
    return all_text_source_data

def load_change_ans_source(args):
    text_source = args.get('text_source', None)
    _args = args.copy()
    
    _args['instruct'] = text_source
    if _args.get('text_src_model', '') == '':
        _args['model_name'] = 'gpt-4o-mini-2024-07-18' # the text is modified based on gpt4o
    else:
        _args['model_name'] = _args['text_src_model']
    
    text_source_file = get_cor_save_path(_args)
    
    text_source_data = {}
    
    for line in open(text_source_file, 'r'):
        one_data = json.loads(line)
        
        if one_data['question_id'] not in text_source_data:
            #
            texts = one_data['changed_ans']
            text_source_data[one_data['question_id']] = texts
    
    return text_source_data
    

import numpy as np
from PIL import Image
def add_noise_to_img(numpy_image, stddev=50):
    # numpy_image = np.array(image)
        
    # Create Gaussian noise
    mean = 0
    # stddev = 50
    noise = np.random.normal(mean, stddev, numpy_image.shape)

    # Add the noise to the image
    noisy_image = numpy_image + noise
    
    # Ensure the values are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)

    # Convert back to PIL Image
    noisy_image_pil = Image.fromarray(np.uint8(noisy_image))
    
    noisy_image = np.array(noisy_image_pil)
    
    return noisy_image


def construct_debate_messages(question, question_id, img=None, **kwargs):
    messages = []
    
    if img is not None:
        messages.append({
            "role": "user",
            "content": question,
            "has_img": True,
            "img": img
        })
    
    # follow up with the original answer
    messages.append({
        "role": "assistant",
        "content": kwargs['debate_org_data'][question_id]['answer']
    })
    
    # follow up with the answers from other agents
    other_agents_str = 'These are the responses from other agents: \n'
    
    all_img_logprobs = []
    all_text_logprobs = []
    
    if kwargs.get('debate_img_src_data', None) is not None:
        for key in kwargs['debate_img_src_data']:
            all_img_logprobs.append(get_logprobs_from_resp(kwargs['debate_img_src_data'][key]))
    if kwargs.get('debate_text_src_data', None) is not None:
        for key in kwargs['debate_text_src_data']:
            all_text_logprobs.append(get_logprobs_from_resp(kwargs['debate_text_src_data'][key]))

    if kwargs.get('debate_img_src_data', None) is not None:
        other_agents_str += 'One response from an agent based on the image:' + kwargs['debate_img_src_data'][question_id]['answer'] + '\n'
        other_agents_str += 'The confidence of the response is ' + str(round(get_confidence(get_logprobs_from_resp(kwargs['debate_img_src_data'][question_id])) * 100)) + '%\n\n'
        # other_agents_str += 'The confidence of the response is ' + str(round(get_scaled_confidence_minmax(get_logprobs_from_resp(kwargs['debate_img_src_data'][question_id]), all_img_logprobs) * 100)) + '%\n\n'

    if kwargs.get('debate_text_src_data', None) is not None:
        other_agents_str += 'One response from an agent based on the caption:' + kwargs['debate_text_src_data'][question_id]['answer'] + '\n'
        other_agents_str += 'The confidence of the response is ' + str(round(get_confidence(get_logprobs_from_resp(kwargs['debate_text_src_data'][question_id]))*100)) + '%\n\n'
        # other_agents_str += 'The confidence of the response is ' + str(round(get_scaled_confidence_minmax(get_logprobs_from_resp(kwargs['debate_text_src_data'][question_id]), all_text_logprobs)*100)) + '%\n\n'

    
    other_agents_str += '\n\n Use these responses carefully as additional advice, can you provide an updated answer? Please answer the question using a single word or phrase.'
    
    messages.append({
        'role': 'user',
        'content': other_agents_str
    })
    
    return messages

from eval_utils import normalize_answer
def get_discriminate_prompt(question, question_id, img=None, **kwargs):
    messages = []
    
    # follow up with the answers from other agents
    # other_agents_str = 'These are the responses from other agents: \n'
    
    # all_img_logprobs = []
    # all_text_logprobs = []
    
    # if kwargs.get('debate_img_src_data', None) is not None:
    #     for key in kwargs['debate_img_src_data']:
    #         all_img_logprobs.append(get_logprobs_from_resp(kwargs['debate_img_src_data'][key]))
    # if kwargs.get('debate_text_src_data', None) is not None:
    #     for key in kwargs['debate_text_src_data']:
    #         all_text_logprobs.append(get_logprobs_from_resp(kwargs['debate_text_src_data'][key]))

    if kwargs.get('debate_img_src_data', None) is not None:
        img_ans = kwargs['debate_img_src_data'][question_id]['answer']
        
        # other_agents_str += 'One response from an agent based on the image:' + kwargs['debate_img_src_data'][question_id]['answer'] + '\n'
        # other_agents_str += 'The confidence of the response is ' + str(round(get_confidence(get_logprobs_from_resp(kwargs['debate_img_src_data'][question_id])) * 100)) + '%\n\n'
        # other_agents_str += 'The confidence of the response is ' + str(round(get_scaled_confidence_minmax(get_logprobs_from_resp(kwargs['debate_img_src_data'][question_id]), all_img_logprobs) * 100)) + '%\n\n'

    if kwargs.get('debate_text_src_data', None) is not None:
        text_ans = kwargs['debate_text_src_data'][question_id]['answer']
        
        # other_agents_str += 'One response from an agent based on the caption:' + kwargs['debate_text_src_data'][question_id]['answer'] + '\n'
        # other_agents_str += 'The confidence of the response is ' + str(round(get_confidence(get_logprobs_from_resp(kwargs['debate_text_src_data'][question_id]))*100)) + '%\n\n'
        # other_agents_str += 'The confidence of the

    if kwargs.get('debate_org_data', None) is not None:
        org_ans = kwargs['debate_org_data'][question_id]['answer']
        
    # cut off the question with the original question
    if kwargs.get('org_question', None) is not None:
        org_question = kwargs['org_question']
        question = question.split(org_question)[0] + org_question
    
    # turn into MCQ format
    options = [img_ans, text_ans, org_ans]
    indexs = []
    
    processed_options = []
    if normalize_answer(org_ans) == normalize_answer(img_ans) == normalize_answer(text_ans):
        processed_options = [img_ans]
        indexs = [[0,1,2]]
    elif normalize_answer(org_ans) == normalize_answer(img_ans):
        processed_options = [img_ans, text_ans]
        indexs = [[0,2], [1]]
    elif normalize_answer(org_ans) == normalize_answer(text_ans):
        processed_options = [img_ans, text_ans]
        indexs = [[0], [1,2]]
    elif normalize_answer(img_ans) == normalize_answer(text_ans):
        processed_options = [img_ans, org_ans]
        indexs = [[0,1], [2]]
    else:
        processed_options = options
        indexs = [[0], [1], [2]]    
    
    
    # get shuffle index
    shuffle_idx = np.random.permutation(len(processed_options))
    real_options = [processed_options[i] for i in shuffle_idx]
    real_indexs = [indexs[i] for i in shuffle_idx]
    
    # construct the question
    max_opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    question = question + ' Which of the following answers is the most relevant to the question? \n'
    for i, (option, index) in enumerate(zip(real_options, real_indexs)):
        question += f'({max_opts[i]}) {option}\n'
            
    
    return question, real_indexs, real_options
    

from openai.types.chat import ChatCompletionTokenLogprob
def get_logprobs_from_resp(one_data):
    logprobs = one_data.get('logprobs', None)
    
    ret_logprobs = []
    if logprobs is not None:
        logprobs = eval(logprobs)

        if logprobs is not None:
            for i in range(len(logprobs)):
                if isinstance(logprobs[i], ChatCompletionTokenLogprob):
                    ret_logprobs.append(logprobs[i].logprob)
                else:
                    ret_logprobs.append(logprobs[i]['logprob'])
                    
    return ret_logprobs

import numpy as np
def get_confidence(logprobs):
    # return np.mean(logprobs)
    # return logprobs[0]
    # return logprobs[-1]
    return np.exp(np.mean(logprobs))
    # return np.power(np.exp(np.sum(logprobs)), 1/len(logprobs))

def get_scaled_confidence_minmax(logprobs, all_logprobs):
    all_confs = []
    for logprob in all_logprobs:
        all_confs.append(get_confidence(logprob))
    
    conf = get_confidence(logprobs)
    min_conf = min(all_confs)
    max_conf = max(all_confs)
    
    return (conf - min_conf) / (max_conf - min_conf)
    


def get_sp_question(args):
    if args.get('sp_task', '') == 'disc':
        question = 'Is the text information matched, corrupted or irrelevant to the image? Please only output the answer directly.'
    
    return question

def get_sp_answers(args):
    if args.get('sp_task', '') == 'disc':
        if 'contradict' in args.get('text_source', ''):
            return ['corrupted']
        elif 'supp' in args.get('text_source', ''):
            return ['matched']
        elif 'irrel' in args.get('text_source', ''):
            return ['irrelevant']

import json
import pandas as pd
import datasets
from datasets import Dataset, Features, Value, Sequence
