import glob
import json
import argparse
from eval_utils import *

from openai.types.chat import ChatCompletionTokenLogprob
import numpy as np


def main(args):
    output_file = glob.glob(args['file_name'])[0]
    is_verbose = args.get('verbose', False)
    max_num_samples = args.get('max_num_samples', None)
    
    res = []
    with open(output_file, 'r') as file:
        for line in file:
            if max_num_samples is not None and len(res) >= max_num_samples:
                break
            one_data = json.loads(line)
            
            # fix bug
            # if element still list and current list has only one element
            if isinstance(one_data['answers'][0], list) and len(one_data['answers']) == 1:
                answers = one_data['answers'][0]
            else:
                answers = one_data['answers']
                
            logprobs = one_data.get('logprobs', None)
            
            if answers == 'Error' or logprobs == 'Error':
                continue
            
            ret_logprobs = []
            if logprobs is not None:
                logprobs = eval(logprobs)

                if logprobs is not None:
                    for i in range(len(logprobs)):
                        if isinstance(logprobs[i], ChatCompletionTokenLogprob):
                            ret_logprobs.append(logprobs[i].logprob)
                        else:
                            ret_logprobs.append(logprobs[i]['logprob'])
                            
            
            item = {
                'question_id': one_data['question_id'],
                'question': one_data['input'],
                'pred_answer': one_data['output'],
                'gt_answers': answers,
            }
            
            ds_name = get_dataset_name(output_file)
            
            # try:
            one_data_acc = eval_func(ds_name, [item])['acc']
            # except Exception as e:
            #     print(output_file)
            #     print(e)
            #     one_data_acc = 0.0 
            
            res.append({
                'question_id': one_data['question_id'],
                'question': one_data['input'],
                'pred_answer': one_data['output'],
                'gt_answers': answers,
                'logprobs': ret_logprobs,
                
                # my eval
                'best_subspan_em': best_subspan_em(one_data['output'], answers),
                'acc': one_data_acc
            })
            
            
    report = {
        'res': {},
        'file_name': output_file,
        'num_samples': len(res)
    }
    
    if is_verbose:
        report['verbose'] = res
    
    
    if 'lmms-lab/DocVQA' in output_file:
        preds, golds, question_ids = [], [], []
        answer_types = []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'])
            question_ids.append(r['question_id']) # 123/["xxx"]
            answer_types.append(eval(''.join(r['question_id'].split('/')[1:])))
        
        docvqa_evaluator = DocVQAEvaluator()
        _report =  docvqa_evaluator.get_metrics(golds, preds, None)
        
        report['res'] = _report
        # report['res']['acc'] = _report['accuracy']
        report['res']['acc'] = _report['anls']
    
    elif 'lmms-lab/VQAv2' in output_file:
        preds, golds, question_ids = [], [], []
        answer_types = []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'])
            question_ids.append(r['question_id'])
        
        vqav2_evaluator = VQAv2Evaluator(question_ids=question_ids)
        _report = vqav2_evaluator.evaluate(question_ids, preds)
        # _report = vqav2_evaluator.get_metrics(golds, preds, None)
        
        report['res'] = _report
        report['res']['acc'] = _report['accuracy']
    
    elif 'openphish' in output_file:
        preds, golds = [], []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'][0])
            
        report['res'] = eval_openphish(preds, golds)

    # report['num_samples'] = len(res)
    # report['file_name'] = output_file
    
    # my eval
    report['res']['best_subspan_em'] = np.mean([r['best_subspan_em'] for r in res])
    
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file_name', type=str, default='outputs/facebook/textvqa/llava-hf-llava-1.5-7b-hf/t=0_seed=0.jsonl')
    args = parser.parse_args()
    
    args = vars(args)
    
    report = main(args)
    
    print(report)
