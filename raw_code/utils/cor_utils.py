from .cor_constant import INSTRUCT_TEMP
import os
from datasets import load_dataset


def get_cor_prompt(ds_name, question, instruct='', tpl_params={}):
    tpl_params['question'] = question
    if 'lmms-lab/VQAv2' in ds_name:
        return INSTRUCT_TEMP[instruct].format(**tpl_params)
    elif 'lmms-lab/DocVQA' in ds_name:
        return INSTRUCT_TEMP[instruct].format(**tpl_params)
    elif 'echo840/OCRBench' in ds_name:
        return INSTRUCT_TEMP[instruct].format(**tpl_params)
    elif 'infoseek' in ds_name:
        return INSTRUCT_TEMP[instruct].format(**tpl_params)
    
def clean_response(response, instruct=''):
    if instruct in ['replace_noun', 'conflict_answer', 'conflict_answer_wo_ref', 'noisy_but_same'] or 'noisy_but_same' in instruct:
        return response.split('New sentence: ')[-1].strip()
    
    elif instruct == 'describe':
        return response
    
    elif 'supp_contradict_stat' in instruct:
        sent1 = response.split('Description 1: ')[-1].split('\n')[0].strip()
        sent2 = response.split('Description 2: ')[-1].split('\n')[0].strip()
        ans2 = response.split('Answer 2: ')[-1].strip()
        
        return sent1, sent2, ans2
    
    return response

def get_save_path(args):
    ds_name = args['ds_name']
    model_name = args['model_name']
    desc_model_name = args.get('desc_model_name', '')
    
    # get current file's directory path's parent directory
    if args.get('save_dir', '') != '':
        p = args['save_dir']
    else:
        p  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args['instruct'] in ['replace_noun', 'describe', 'conflict_answer', 'desc_one_sent', 'what_is_written', 'conflict_answer_wo_ref', 'noisy_but_same'] or 'noisy_but_same' in args['instruct']:
        save_dir = '{}/syth_response/{}/{}/'.format(p, ds_name, model_name.replace('/', '-'))
    
    elif 'supp_contradict_stat' in args['instruct'] or 'supp_stat' in args['instruct'] or 'contradict_stat' in args['instruct']:
        save_dir = '{}/syth_response/{}/{}/'.format(p, ds_name, model_name.replace('/', '-'))
    
    else:
        save_dir = '{}/syth_response/{}/'.format(p, ds_name)
        
    if args.get('data_seed', 0) != 0:
        save_dir = save_dir + 'seed={}/'.format(args['data_seed'])
    
    
    if desc_model_name != '':
        save_file = '{}/descmodel={}/{}'.format(save_dir, desc_model_name.replace('/', '-'), args['instruct'])
    else:
        save_file = '{}/{}'.format(save_dir, args['instruct'])
        
    if 'supp_contradict_stat' in args['instruct']:
        save_file = save_file.replace('supp_contradict_stat', '{supp_contradict_stat}')

    
    save_file = save_file + '.jsonl'
    
    return save_file


def load_custom_dataset(ds_name):
    if 'lmms-lab/VQAv2' in ds_name:
        ds = load_dataset(ds_name)
        ds = ds['validation']
        # ds = ds['test'] # should use test cases, but they don't provide

    elif 'lmms-lab/DocVQA' in ds_name:
        sub_task = ds_name.split('/')[-1]
        _ds_name = 'lmms-lab/DocVQA'
        ds = load_dataset(_ds_name, sub_task)
        ds = ds['validation']    
    elif 'echo840/OCRBench' in ds_name:
        ds = load_dataset('echo840/OCRBench')
        ds = ds['test']
        
        _sub_task = ds_name.split('/')[-1]
        
        if _sub_task == 'text_recognition':
            # filter the question type with 'text recognition'
            ds = ds.filter(lambda x: 'text recognition' in x['question_type'].lower())
    
    return ds

from tqdm import tqdm
def build_ds_index(ds, question_ids):
    ds_index = {}
    
    progress_bar = tqdm(total=len(question_ids))
    
    for d in ds:
        if d['question_id'] in question_ids and d['question_id'] not in ds_index:
            ds_index[d['question_id']] = d
            progress_bar.update(1)
            
            # if len(ds_index) > 100: break
            
            if len(ds_index) == len(question_ids):
                break
    progress_bar.close()
    
    return ds_index
            

from datasets import Dataset, Features, Value, Sequence, Image as HFImage

class DataProcessor:
    def __init__(self, data_type):
        self.data_type = data_type
        self.all_data_dict = {}
        self.features = None
        
        self.init_data_dict(self.data_type)
        self.init_feature_format()
    
    def init_data_dict(self, data_type):
        if data_type == 'dpo':
            self.all_data_dict = {
                'chosen': [],
                'rejected': [],
                'images': [],
                'prompt': []
            }
        elif data_type == 'sft':
            self.all_data_dict = {
                'messages': [],
                'images': []
            }
        
    
    def init_feature_format(self):
        
        if self.data_type == 'dpo':
            self.features = Features({
                'chosen': [{
                    'content': [{
                        'text': Value('string'),
                        'type': Value('string'),
                    }],
                    'role': Value('string'),
                }],
                'rejected': [{
                    'content': [{
                        'text': Value('string'),
                        'type': Value('string'),
                    }],
                    'role': Value('string'),
                }],
                'images': Sequence(HFImage()),  # Use Sequence(HFImage()) if multiple images, otherwise just HFImage()
                'prompt': [{
                    'content': [{
                        'text': Value('string'),
                        'type': Value('string'),
                    }],
                    'role': Value('string'),
                }]
            })
        elif self.data_type == 'sft':
            self.features = Features({
                'images': Sequence(HFImage()),
                'messages': [{
                    'content': [{
                        'index': Value('int32'),
                        'text': Value('string'),
                        'type': Value('string'),
                    }],
                    'role': Value('string'),
                }],
            })
        
        return self.features
    
    def add_data(self, data):
        
        for k in data.keys():
            assert k in self.all_data_dict
            # print('key', k)
            
            self.all_data_dict[k].append(data[k])
            # print(self.all_data_dict.keys())
            # print(self.all_data_dict['images'])
        
    def get_all_data_dict(self):
        return self.all_data_dict
    
    def mk_dataset(self):
        return Dataset.from_dict(self.all_data_dict, features=self.features)


# def add_to_collector(collector, data, data_type):
#     if data_type not in collector:
#         collector[data_type] = []
    
#     collector[data_type].append(data)

#     return collector
import random
class DataCollector:
    def __init__(self):
        self.collector = {}
    
    def add_data(self, data, data_type):
        if data_type not in self.collector:
            self.collector[data_type] = []
        
        self.collector[data_type].append(data)
        
    def subset(self, num_map=None):
        all_data = []
        if num_map is None:
            for k, v in self.collector.items():
                all_data.extend(v)
        else:
            for k, v in num_map.items():
                shuffle_data = self.collector[k].copy()
                random.shuffle(shuffle_data)
                all_data.extend(shuffle_data[:v])
        
        return all_data