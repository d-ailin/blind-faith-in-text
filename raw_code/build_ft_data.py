'''
    construct the data as huggingface dataset locally
'''
import datasets
from PIL import Image
import os
import pathlib
import json
from tqdm import tqdm
import argparse
import numpy as np
from data_utils import load_coco_captions, suffix_data, get_question_prompt, get_question_prompt_v2, suffix_ans_data
from utils import QueryModel
from utils.cor_utils import get_cor_prompt, clean_response, get_save_path, load_custom_dataset, build_ds_index, DataProcessor, DataCollector
import random
import yaml


def convert_img(image_data):
    # Step 1: Convert the image to a NumPy array
    image_array = np.array(image)

    # Now you can access the pixel values using indexing, for example:
    # Access the pixel at row 0, column 0 (top-left corner)
    # pixel = image_array[0, 0]
    # print(f"Pixel at (0, 0): {pixel}")

    # Step 2: If you want to convert it back to a PIL Image after manipulation
    image_back = Image.fromarray(image_array)
    
    return image_back


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')
# parser.add_argument('--seed', type=int, default=0)
# parser.add_argument('--t', type=int, default=0)
# parser.add_argument('--instruct', type=str, default='replace_noun')
# parser.add_argument('--max_samples', type=int, default=2000) # change from 800 to 2000
# parser.add_argument('--exclude_list', type=str, default='') # exclude the question ids from the list file
# parser.add_argument('--save_dir', type=str, default='')

args = parser.parse_args()

args = vars(args)

# load config
config = yaml.load(open(args['config']), Loader=yaml.FullLoader)

# output_dir is the same as the config file, adding '/data' to the end
output_dir = os.path.join(os.path.dirname(args['config']), 'data')

data_type = config['data_type']


# all_data_dict = {
#     'chosen': [],
#     'rejected': [],
#     'images': [],
#     'prompt': []
# }

data_processor = DataProcessor(data_type)

data_collector = DataCollector()

# print(data_processor.init_feature_format())
# print(data_processor.all_data_dict)

train_size = config['train']
test_size = config['test']
total_size = config['total']

# load the data
for source in config['sources']:
    ds_name = source['dataset']
    input_files = source['inputs']
    question_type = source.get('question_type', None)
    answer_format = source.get('answer_format', None)
    additional_output_types = source.get('additional_output_types', None)
    
    ds = load_custom_dataset(ds_name)
    
    question_ids = []
    items = {}
    for input_file in input_files:
        with open(input_file, 'r') as f:
            for line in f:
                question_ids.append(json.loads(line)['question_id'])
                items[json.loads(line)['question_id']] = json.loads(line)
                
    
    ds_index_map = build_ds_index(ds, question_ids)
    
    for question_id in tqdm(question_ids):
        data = items[question_id]
        raw_data = ds_index_map.get(question_id, None)
        row = {}
        
        # if raw_data is None:
        #     continue        
        
        # image = raw_data['image']
        # jpg_image = convert_img(image)
                
        row['images'] = [raw_data['image']]
        question = raw_data['question']
        
        if source['name'] == 'conflict':
            
            prompts = source['prompts']
            task = source['task']

            
            # randomly choose a negative answer
            neg_index = np.random.choice(len(data['changed_ans']))
            prompt_tpl = np.random.choice(prompts)
            
            if task == 'caption':
                tpl_params = {
                    'captions': data['response'][neg_index]
                }
                
                # if there exists a 'supp_stat' key in the data, use it
                # this is for the 'supp_contradict_stat' type of data
                # we think the response is the conflicting statement
                if 'supp_stat' in data:
                    pos_tpl_params = {
                        'captions': data['supp_stat'][neg_index]
                    }
                else:
                    pos_tpl_params = {
                        'captions': data['original'][neg_index]
                    }
                                
            elif task == 'ocr':
                tpl_params = {
                    'ocr_results': data['response'][neg_index]
                }
                
                pos_tpl_params = {
                    'ocr_results': data['original_ans'][neg_index]
                }
            
            if question_type == 'v2':
                question = get_question_prompt_v2(ds_name, question)
            else:
                question = get_question_prompt(ds_name, question)
            
            neg_final_query = suffix_data(question, prompt_tpl, tpl_params, task)
            pos_final_query = suffix_data(question, prompt_tpl, pos_tpl_params, task)
            
            answer = raw_data['answers'][0]['answer']
            
            if answer_format == 'process_sup':
                pos_final_answer = suffix_ans_data(answer, answer_format, ans_type='match')
                neg_final_answer = suffix_ans_data(answer, answer_format, ans_type='corrupt')
            else:
                pos_final_answer = answer
                neg_final_answer = answer
                neg_real_answer = data['changed_ans'][neg_index]
            
            
            if data_type == 'dpo':
                
                neg_answer = data['changed_ans'][neg_index]
                
                
                row['prompt'] = [{'content': [{'text': None, 'type': 'image'}, {'text': neg_final_query, 'type': 'text'}], 'role': 'user'}]
                row['chosen'] = [{'content': [{'text': pos_final_answer, 'type': 'text'}], 'role': 'assistant'}]
                row['rejected'] = [{'content': [{'text': neg_answer, 'type': 'text'}], 'role': 'assistant'}]
                
                # data_processor.add_data(row)
                
                data_collector.add_data(row, 'dpo')

            elif data_type == 'sft':
                
                pos_messages = [
                    {'content': [{'index': 0, 'text': None, 'type': 'image'}, {'index': None, 'text': pos_final_query , 'type': 'text'}], 'role': 'user'},
                    {'content': [{'index': None, 'text': pos_final_answer, 'type': 'text'}], 'role': 'assistant'}
                ]
                
                neg_messages = [
                    {'content': [{'index': 0, 'text': None, 'type': 'image'}, {'index': None, 'text': neg_final_query, 'type': 'text'}], 'role': 'user'},
                    {'content': [{'index': None, 'text': neg_final_answer, 'type': 'text'}], 'role': 'assistant'}
                ]
                
                
                pos_row = row.copy()
                neg_row = row.copy()
                pos_row['messages'] = pos_messages
                neg_row['messages'] = neg_messages
                
                # data_processor.add_data(pos_row)
                # data_processor.add_data(neg_row)
                
                data_collector.add_data(pos_row, 'sft_pos_imgtext')
                data_collector.add_data(neg_row, 'sft_neg_imgtext')
                
                if additional_output_types is not None:
                    if additional_output_types[0] == 'text':
                        text_row = row.copy()
                        # delete the 'images' key
                        text_row['images'] = []
                        pos_text_messages = [
                            {'content': [{'index': None, 'text': pos_final_query, 'type': 'text'}], 'role': 'user'},
                            {'content': [{'index': None, 'text': pos_final_answer, 'type': 'text'}], 'role': 'assistant'}
                        ]
                        
                        neg_text_messages = [
                            {'content': [{'index': None, 'text': neg_final_query, 'type': 'text'}], 'role': 'user'},
                            {'content': [{'index': None, 'text': neg_real_answer, 'type': 'text'}], 'role': 'assistant'}
                        ]
                        
                        pos_text_row = text_row.copy()
                        neg_text_row = text_row.copy()
                        
                        pos_text_row['messages'] = pos_text_messages
                        neg_text_row['messages'] = neg_text_messages
                        
                        # data_processor.add_data(pos_text_row)
                        # data_processor.add_data(neg_text_row)
                        
                        data_collector.add_data(pos_text_row, 'sft_pos_text')
                        data_collector.add_data(neg_text_row, 'sft_neg_text')

                        
                
            
        elif source['name'] == 'original':
            row['messages'] = [
                {'content': [{'index': 0, 'text': None, 'type': 'image'}, {'index': None, 'text': question, 'type': 'text'}], 'role': 'user'},
                {'content': [{'index': None, 'text': raw_data['answers'][0]['answer'], 'type': 'text'}], 'role': 'assistant'}
            ]
        
            # data_processor.add_data(row)
            data_collector.add_data(row, 'sft_qa_imgtext')
        
        elif source['name'] == 'irrelevant':
            prompts = source['prompts']
            task = source['task']            
            prompt_tpl = np.random.choice(prompts)
            
            if task == 'caption':
                tpl_params = {
                    'captions': data['response'][neg_index]
                }
                
            elif task == 'ocr':
                tpl_params = {
                    'ocr_results': data['response'][neg_index]
                }

            if question_type == 'v2':
                question = get_question_prompt_v2(ds_name, question)
            else:
                question = get_question_prompt(ds_name, question)
            
            neg_final_query = suffix_data(question, prompt_tpl, tpl_params, task)
            # pos_final_query = suffix_data(question, prompt_tpl, pos_tpl_params, task)
            
            answer = raw_data['answers'][0]['answer']
            
            if answer_format == 'process_sup':
                neg_final_answer = suffix_ans_data(answer, answer_format, ans_type='irrelevant')
            else:
                neg_final_answer = answer
            
            
            row['messages'] = [
                {'content': [{'index': 0, 'text': None, 'type': 'image'}, {'index': None, 'text': neg_final_query, 'type': 'text'}], 'role': 'user'},
                {'content': [{'index': None, 'text': neg_final_answer, 'type': 'text'}], 'role': 'assistant'}
            ]
        
            # data_processor.add_data(row)
            
            data_collector.add_data(row, 'sft_irrel_imgtext')
        
        # Add each feature's data to its respective list in the dictionary
        # all_data_dict['chosen'].append(row['chosen'])
        # all_data_dict['rejected'].append(row['rejected'])
        # all_data_dict['images'].append(row['images'])
        # all_data_dict['prompt'].append(row['prompt'])
        
# shuffle the data and split
# random.shuffle(all_data)
# train_data = all_data[:train_size]
# test_data = all_data[train_size:train_size+test_size]

# from datasets import Dataset, Features, Value, Sequence, Image as HFImage



# # Define dataset features
# features = Features({
#     'chosen': [{
#         'content': [{
#             'text': Value('string'),
#             'type': Value('string'),
#         }],
#         'role': Value('string'),
#     }],
#     'rejected': [{
#         'content': [{
#             'text': Value('string'),
#             'type': Value('string'),
#         }],
#         'role': Value('string'),
#     }],
#     'images': Sequence(HFImage()),  # Use Sequence(HFImage()) if multiple images, otherwise just HFImage()
#     'prompt': [{
#         'content': [{
#             'text': Value('string'),
#             'type': Value('string'),
#         }],
#         'role': Value('string'),
#     }]
# })

# all_data_dict = data_processor.get_all_data_dict()
# features = data_processor

# Create the dataset from the list of items
# dataset = Dataset.from_dict(all_data_dict, features=features)

# print('data_processor.all_data_dict', data_processor.all_data_dict[:5])
# print('all_data_dict[image][:5]', data_processor.all_data_dict['images'][:5])
# print('features', data_processor.features)

if 'num_map' in config:
    print('using num_map')
    data = data_collector.subset(config['num_map'])
else:
    data = data_collector.subset()
    
for d in data:
    data_processor.add_data(d)

dataset = data_processor.mk_dataset()

# only sample the data with total_size, new added at Sep 26
dataset = dataset.shuffle(seed=0)
dataset = dataset.select(range(total_size))

# Split the dataset into train and test sets
train_test_split = dataset.train_test_split(test_size=test_size/total_size)

# Access the train and test datasets
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Combine the train and test splits into a DatasetDict
dataset_dict = datasets.DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# # Create the dataset with training and test splits
# data_dict = {
#     'train': train_data,
#     'test': test_data
# }

# # Create a dataset in Hugging Face format
# my_dataset = datasets.DatasetDict({
#     'train': datasets.Dataset.from_list(data_dict['train']),
#     'test': datasets.Dataset.from_list(data_dict['test'])
# })

# my_dataset = my_dataset.cast_column('images', datasets.Image())

# Save the dataset locally
dataset_dict.save_to_disk(output_dir)

# To load the dataset locally in the future
loaded_dataset = datasets.load_from_disk(output_dir)
print(loaded_dataset)