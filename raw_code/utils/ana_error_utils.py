import os
import json

PROMPTS = {
    'reason_w_text': '''
        Given a question, we provide an image and text information to answer,
        what is the reason for the predicted answer? Is it due to incorrect, irrelevant, or incomplete information or other reasons?

        Here are the examples:
        Question: What is the year printed on this calendar?
        Groundtruth answer: soul
        Final query: Here is some information from OCR:'Ml'\nAnswer the question using a single word or phrase. what is written in the image?
        Predicted answer: foul
        Reason: Incorrect OCR information
    
        Question: {question}
        Groundtruth answer: {answer}
        Final query: {query}
        Predicted answer: {response}
    '''
}

def get_ana_error_prompt(ds_name, question, instruct='', tpl_params={}):
    tpl_params['question'] = question
    if 'lmms-lab/VQAv2' in ds_name:
        return PROMPTS[instruct].format(**tpl_params)
    elif 'lmms-lab/DocVQA' in ds_name:
        return PROMPTS[instruct].format(**tpl_params)
    
def load_response(source_file):
    res = {}
    with open(source_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            res[data['question_id']] = data
    return res

def get_save_path(args):
    ds_name = args['ds_name']
    model_name = args['model_name']
    
    # get current file's directory path's parent directory
    p  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args['instruct'] in ['reason_w_text']:
        save_dir = '{}/ana_error_response/{}/{}/'.format(p, ds_name, model_name.replace('/', '-'))
    else:
        save_dir = '{}/ana_error_response/{}/'.format(p, ds_name)
    save_file = '{}/{}'.format(save_dir, args['instruct'])
    
    save_file = save_file + '.jsonl'
    
    return save_file