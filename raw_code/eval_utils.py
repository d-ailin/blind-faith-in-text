'''
    https://github.com/TRI-ML/vlm-evaluation/blob/main/vlm_eval/util/evaluation/textvqa/m4c_evaluators.py
'''

# Copyright (c) Facebook, Inc. and its affiliates.
## THE OFFICIAL EVAL CODE FOR TEXT VQA
import re


# ruff: noqa: RUF012
class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI.

    See: github.com/facebookresearch/mmf/blob/c46b3b3391275b4181567db80943473a89ab98ab/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (re.search(self.COMMA_STRIP, in_text) is not None):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            word = self.NUMBER_MAP.setdefault(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


class TextVQAAccuracyEvaluator:
    def __init__(self):
        self.answer_processor = EvalAIAnswerProcessor()

    def _compute_answer_scores(self, raw_answers):
        """
        compute the accuracy (soft score) of human answers
        """
        answers = [self.answer_processor(a) for a in raw_answers]
        assert len(answers) == 10
        gt_answers = list(enumerate(answers))
        unique_answers = set(answers)
        unique_answer_scores = {}

        for unique_answer in unique_answers:
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [item for item in other_answers if item[1] == unique_answer]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[unique_answer] = sum(accs) / len(accs)

        return unique_answer_scores

    def eval_pred_list(self, pred_list, verbose=False):
        pred_scores = []
        for entry in pred_list:
            pred_answer = self.answer_processor(entry["pred_answer"])
            unique_answer_scores = self._compute_answer_scores(entry["gt_answers"])
            score = unique_answer_scores.get(pred_answer, 0.0)
            pred_scores.append(score)
            
            if verbose:
                if score == 0:
                    print(f"Question: {entry['question']}")
                    print(f"Predicted Answer: {pred_answer}")
                    print(f"GT Answers: {entry['gt_answers']}")
                    print(f"Answer Scores: {unique_answer_scores}")
                    print("\n")

        accuracy = sum(pred_scores) / len(pred_scores)
        return accuracy
    

'''
https://github.com/VisualWebBench/VisualWebBench/blob/main/utils/eval_utils.py
'''

import re

import numpy as np
from rouge import Rouge 

import torch
from torchvision.ops import box_iou


def eval_web_caption(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )


def eval_heading_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    res = dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )
    
    res['acc'] = res['rouge_l']
    
    return res


def eval_element_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i] or len(preds[i]) == 1:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    res = dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )
    
    res['acc'] = res['rouge_l']
    
    return res


def eval_action_prediction(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_element_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_action_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_element_bbox_ground(preds, golds, **kwargs):
    # print('preds[0]', preds[0])
    # print('golds[0]', golds[0])
    correct = total_cnt = 0
    for i, predict_bbox in enumerate(preds):
        if not predict_bbox:
            predict_bbox = (0., 0., 0., 0.)
        try:
            target_bbox = torch.tensor(golds[i], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            iou = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= 0.5:
                correct += 1
        except:
            pass

        total_cnt += 1

    return dict(
        precision=correct / total_cnt * 100
    )


def eval_action_bbox_ground(preds, golds, **kwargs):
    correct = total_cnt = 0
    for i, predict_bbox in enumerate(preds):
        if not predict_bbox:
            predict_bbox = (0., 0., 0., 0.)
        try:
            target_bbox = torch.tensor(golds[i], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            iou = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= 0.5:
                correct += 1
        except:
            pass

        total_cnt += 1

    return dict(
        precision=correct / total_cnt * 100
    )

# print error stack
import traceback
def eval_webqa(preds, golds, **kwargs):
    f1_scores = []
    rouge = Rouge(metrics=['rouge-1'])
    for pred, gold_list in zip(preds, golds):
        try:
            if not pred:
                pred = " "
            cur_f1 = max([rouge.get_scores([pred], [gold], avg=True)['rouge-1']['f'] for gold in gold_list])
            f1_scores.append(cur_f1)
        except Exception as e:
            # print(e)
            # print('preds', preds)
            # print('golds', golds)

            traceback.print_exc()
            pass

    res = dict(
        f1=sum(f1_scores) / len(f1_scores) * 100
    )
    
    res['acc'] = res['f1']
    
    return res

def eval_element_point_ground(preds, golds):
    acc_lst = []
    for pred, gold in zip(preds, golds):
        x, y = pred
        left, top, right, bottom = gold
        acc_lst.append(left<=x<=right and top<=y<=bottom)
    return dict(
        accuracy=sum(acc_lst) / len(acc_lst) * 100
    )

def eval_action_point_ground(preds, golds):
    acc_lst = []
    for pred, gold in zip(preds, golds):
        x, y = pred
        left, top, right, bottom = gold
        acc_lst.append(left<=x<=right and top<=y<=bottom)
    return dict(
        accuracy=sum(acc_lst) / len(acc_lst) * 100
    )

# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response: str, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if len(response) == 1:
        return response.upper()
    elif not response:
        return 'a'
    elif re.match(r"[A-Z]\.", response):
        return response[0]

    for char in [',', '.', '!', '?', ';', ':', "'", '"']:
        response = response.replace(char, "")
    response = " " + response + " " # add space to avoid partial match

    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    if len(candidates) == 0:  # still not get answer
        # pred_index = random.choice(all_choices)
        pred_index = "z"
    elif len(candidates) > 1:
        start_indexes = []
        if ans_with_brack: 
            for can in candidates:
                index = response.rfind(f'({can})')
                start_indexes.append(index) # -1 will be ignored anyway
            # start_indexes = [generated_response.index(f'({can})') for can in candidates]
        else:
            for can in candidates:
                index = response.rfind(f" {can} ")
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


'''
    https://github.com/Yuliang-Liu/MultimodalOCR/blob/main/example.py 
'''
def eval_ocrbench(preds, golds, question_ids):
    score_map = {}
    
    for i, (predict, golds, que) in enumerate(zip(preds, golds, question_ids)):
        data_split, question_type = que.split('/')[:2]
        answers = golds
        
        if question_type not in score_map:
            score_map[question_type] = []
        
        if data_split == "HME100k":
            if type(answers)==list:
                _res = 0
                for j in range(len(answers)):
                    answer = answers[j].strip().replace("\n"," ").replace(" ","")
                    predict = predict.strip().replace("\n"," ").replace(" ","")
                    
                    if answers[j] in predict:
                        _res = 1
                        break
                score_map[question_type].append(_res)
            else:
                answers = answers.strip().replace("\n"," ").replace(" ","")
                predict = predict.strip().replace("\n"," ").replace(" ","")
                
                score_map[question_type].append(1 if answers in predict else 0)

        else:
            if type(answers)==list:
                _res = 0
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace("\n"," ")
                    predict = predict.lower().strip().replace("\n"," ")
                    
                    if answers[j] in predict:
                        _res = 1
                        break
                    
                score_map[question_type].append(_res)
            else:
                answers = answers.lower().strip().replace("\n"," ")
                predict = predict.lower().strip().replace("\n"," ")
                
                score_map[question_type].append(1 if answers in predict else 0)

    acc_score_map = {}
    for key in score_map:
        # acc_score_map[key] = sum(score_map[key]) / len(score_map[key]) * 100
        acc_score_map[key] = sum(score_map[key])
    
    OCRBench_score = acc_score_map
    
    # print('OCRBench_score', OCRBench_score)
    
    # recognition_score = OCRBench_score['Regular Text Recognition']+OCRBench_score['Irregular Text Recognition']+OCRBench_score['Artistic Text Recognition']+OCRBench_score['Handwriting Recognition']+OCRBench_score['Digit String Recognition']+OCRBench_score['Non-Semantic Text Recognition']
    # Final_score = recognition_score+OCRBench_score['Scene Text-centric VQA']+OCRBench_score['Doc-oriented VQA']+OCRBench_score['Key Information Extraction']+OCRBench_score['Handwritten Mathematical Expression Recognition']
    
    # acc_score_map['Recognition'] = recognition_score
    
    
    acc_score_map['acc'] = 0
    for key in acc_score_map:
        if key != 'acc':
            acc_score_map['acc'] += acc_score_map[key]
    
    return acc_score_map

def eval_func(f, res):
    report = {}
    if 'textvqa' in f:
        evaluator = TextVQAAccuracyEvaluator()
        acc = evaluator.eval_pred_list(res)
        
        report['acc'] = acc
        
    elif 'visualwebbench' in f:
        preds, golds = [], []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'][0])
        if 'heading_ocr' in f:
            report = eval_heading_ocr(preds, golds)
        elif 'element_ocr' in f:
            report = eval_element_ocr(preds, golds)
        elif 'webqa' in f:
            report = eval_webqa(preds, golds)
    
    elif 'OCRBench' in f:
        preds, golds, question_ids = [], [], []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'][0])
            question_ids.append(r['question_id'])
        report = eval_ocrbench(preds, golds, question_ids)

    return report


import numpy as np
from Levenshtein import distance

def levenshtein_distance(s1, s2):
    return distance(s1, s2)

def normalized_levenshtein_distance(s1, s2):
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0  # Both strings are empty
    return distance / max_len

def similarity_score(s1, s2):
    if s1 in s2 or s2 in s1:
        return 1.0
    return 1 - normalized_levenshtein_distance(s1, s2)

def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                longest = max(longest, dp[i][j])
            else:
                dp[i][j] = 0

    return longest

def similarity_based_on_substring(A, B):
    if B in A:
        return 1.0
    
    max_len = longest_common_substring(A, B)
    return max_len / len(B)

    # max_len = longest_common_substring(A, B)
    # return max_len
    
import nltk

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def cer(ocr_result, ground_truth):
    distance = levenshtein_distance(ocr_result, ground_truth)
    return distance / len(ground_truth)

def wer(ocr_result, ground_truth):
    ocr_words = ocr_result.split()
    ground_truth_words = ground_truth.split()
    distance = levenshtein_distance(ocr_words, ground_truth_words)
    return distance / len(ground_truth_words)

def ocr_acc(ocr_result, ground_truth):
    
    # if ground_truth in ocr_result:
    #     return 1.0
    
    return 1 - cer(ocr_result, ground_truth)
    # return 1 - wer(ocr_result, ground_truth)
    
import regex
import string
from typing import List

def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

'''
    DocVQA evaluation: https://github.com/rubenpt91/PFL-DocVQA-Competition/blob/master/metrics.py
'''
import editdistance

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class DocVQAEvaluator(metaclass=Singleton):
    def __init__(self, case_sensitive=False):

        self.case_sensitive = case_sensitive
        self.get_edit_distance = editdistance.eval
        self.anls_threshold = 0.5

        self.total_accuracies = []
        self.total_anls = []

        self.best_accuracy = 0
        # self.best_anls = 0
        self.best_epoch = 0
        

    def get_metrics(self, gt_answers, preds, answer_types=None, update_global_metrics=True):
        answer_types = answer_types if answer_types is not None else ['string' for batch_idx in range(len(gt_answers))]
        batch_accuracy = []
        batch_anls = []
        for batch_idx in range(len(preds)):
            gt = [self._preprocess_str(gt_elm) for gt_elm in gt_answers[batch_idx]]
            pred = self._preprocess_str(preds[batch_idx])

            batch_accuracy.append(self._calculate_accuracy(gt, pred, answer_types[batch_idx]))
            batch_anls.append(self._calculate_anls(gt, pred, answer_types[batch_idx]))

        # return {'accuracy': batch_accuracy, 'anls': batch_anls}
        return {'accuracy': np.mean(batch_accuracy), 'anls': np.mean(batch_anls)}
        

    def update_global_metrics(self, accuracy, anls, current_epoch):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = current_epoch
            return True

        else:
            return False

    def _preprocess_str(self, string):
        if not self.case_sensitive:
            string = string.lower()
        
        # my adding: normalize the answer
        string = normalize_answer(string)

        return string.strip()

    def _calculate_accuracy(self, gt, pred, answer_type):

        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0

        if pred == 'none' and answer_type != 'not-answerable':
            return 0

        for gt_elm in gt:
            if gt_elm == pred:
                return 1

        return 0

    def _calculate_anls(self, gt, pred, answer_type):
        if len(pred) == 0:
            return 0

        if answer_type == 'not-answerable':
            return 1 if pred in ['', 'none', 'NA', None, []] else 0

        if pred == 'none' and answer_type != 'not-answerable':
            return 0

        answers_similarity = [1 - self.get_edit_distance(gt_elm, pred) / max(len(gt_elm), len(pred)) for gt_elm in gt]
        max_similarity = max(answers_similarity)

        anls = max_similarity if max_similarity >= self.anls_threshold else 0
        return anls

from datasets import load_dataset
import os
import json
from tqdm import tqdm

class VQAv2Evaluator:
    vqav2_val_data = None
    def __init__(self, question_ids=[]):
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                                "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap    = { 'none': '0',
                                'zero': '0',
                                'one': '1',
                                'two': '2',
                                'three': '3',
                                'four': '4',
                                'five': '5',
                                'six': '6',
                                'seven': '7',
                                'eight': '8',
                                'nine': '9',
                                'ten': '10'
                            }
        self.articles     = ['a',
                                'an',
                                'the'
                            ]


        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
                                '(', ')', '=', '+', '\\', '_', '-',
                                '>', '<', '@', '`', ',', '?', '!']
        
        # self.question_ids = question_ids
        # self.gts = None
        self.gts = self.load_gt_ans()
    
    def load_gt_ans(self):
        # file
        # ds = load_dataset('lmms-lab/VQAv2')
        # ds = ds['validation']
        # if self.gts is not None:
        #     return self.gts
        if VQAv2Evaluator.vqav2_val_data is not  None:
            return VQAv2Evaluator.vqav2_val_data
        
        
        ds_json = 'lib/vqav2_val.jsonl'
        # get current file path
        current_path = os.path.dirname(os.path.realpath(__file__))
        ds_json = os.path.join(current_path, ds_json)
        ds = {}
        with open(ds_json, 'r') as f:
            for line in f:
                item = json.loads(line)
                ds[item['question_id']] = item
        
        item_map = ds
        
        VQAv2Evaluator.vqav2_val_data = item_map
    
        # for item in ds:
        #     if item['question_id'] not in item_map and item['question_id'] in question_ids:
        #         item_map[item['question_id']] = item
        # print('loaded dataset')
                
        # order_items = []
        # for qid in self.question_ids:
        #     order_items.append(item_map[qid])
        # self.ans_item_list = order_items
        return item_map
        

    # temporary computing, it is not how vqav2 evaluates
    def get_metrics(self, golds, preds, answer_types):
        best_subspan_ems = []
        for gold, pred in zip(golds, preds):
            best_subspan_ems.append(best_subspan_em(pred, gold))
        
        return {
            'accuracy': np.mean(best_subspan_ems) * 100
        }
    
    # from https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py
    def evaluate(self, quesIds, preds):
        # we use golds from self.ans_item_list
        gts = self.gts
        res = {}
        for quesId, pred in zip(quesIds, preds):
            # if 'the answer is xxx' in pred: 
            # extract the answer; the hack trick for evaluating the answer with reasoning path
            if 'the answer is ' in pred: # my change
                pred = pred.split('the answer is ')[1].strip()
            # if '.' is the last character, remove it
            if len(pred) > 0 and pred[-1] == '.': # my change
                pred = pred[:-1]
            
            res[quesId] = {'answer': pred.lower()}
            
        
        # if quesIds == None:
        #     quesIds = [quesId for quesId in self.params['question_id']]
        # gts = {}
        # res = {}
        # for quesId in quesIds:
        #     gts[quesId] = self.vqa.qa[quesId]
        #     res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA       = []
        accQuesType = {}
        accAnsType  = {}
        step = 0
        for quesId in quesIds:
            for ansDic in gts[quesId]['answers']:
                ansDic['answer'] = ansDic['answer'].replace('\n', ' ')
                ansDic['answer'] = ansDic['answer'].replace('\t', ' ')
                ansDic['answer'] = ansDic['answer'].strip()
                ansDic['answer'] = ansDic['answer'].lower() # my adding
            resAns = res[quesId]['answer']
            resAns = resAns.replace('\n', ' ')
            resAns = resAns.replace('\t', ' ')
            resAns = resAns.strip()
            gtAcc = []
            gtAnswers = [ans['answer'] for ans in gts[quesId]['answers']]

            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]['answers']:
                    ansDic['answer'] = self.processPunctuation(ansDic['answer'])
                    ansDic['answer'] = self.processDigitArticle(ansDic['answer'])
            resAns = self.processPunctuation(resAns) # my change
            resAns = self.processDigitArticle(resAns) # my change

            for gtAnsDatum in gts[quesId]['answers']:
                otherGTAns = [item for item in gts[quesId]['answers'] if item!=gtAnsDatum]
                # matchingAns = [item for item in otherGTAns if item['answer']==resAns]
                matchingAns = [item for item in otherGTAns if item['answer'] in resAns] # my change
                acc = min(1, float(len(matchingAns))/3)
                gtAcc.append(acc)
            # quesType    = gts[quesId]['question_type']
            # ansType     = gts[quesId]['answer_type']
            avgGTAcc = float(sum(gtAcc))/len(gtAcc)
            accQA.append(avgGTAcc)
            # if quesType not in accQuesType:
            #     accQuesType[quesType] = []
            # accQuesType[quesType].append(avgGTAcc)
            # if ansType not in accAnsType:
            #     accAnsType[ansType] = []
            # accAnsType[ansType].append(avgGTAcc)
            # self.setEvalQA(quesId, avgGTAcc)
            # self.setEvalQuesType(quesId, quesType, avgGTAcc)
            # self.setEvalAnsType(quesId, ansType, avgGTAcc)
            # if step%100 == 0:
            #     self.updateProgress(step/float(len(quesIds)))
            # step = step + 1
            # print('step', step)

        # self.setAccuracy(accQA, accQuesType, accAnsType)
        return {
            'accuracy': round(100*float(sum(accQA))/len(accQA), 2)
        }
    
    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')
        outText = self.periodStrip.sub("",
                                    outText,
                                    re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy['overall']         = round(100*float(sum(accQA))/len(accQA), self.n)
        self.accuracy['perQuestionType'] = {quesType: round(100*float(sum(accQuesType[quesType]))/len(accQuesType[quesType]), self.n) for quesType in accQuesType}
        self.accuracy['perAnswerType']   = {ansType:  round(100*float(sum(accAnsType[ansType]))/len(accAnsType[ansType]), self.n) for ansType in accAnsType}

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100*acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100*acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100*acc, self.n)

    
def eval_func(ds_name, res):
    report = {}
    if 'textvqa' in ds_name:
        evaluator = TextVQAAccuracyEvaluator()
        acc = evaluator.eval_pred_list(res)
        
        report['acc'] = acc
        
    elif 'visualwebbench' in ds_name:
        preds, golds = [], []
        for r in res:
            if 'webqa' in ds_name:
                preds.append(r['pred_answer'])
                golds.append(r['gt_answers'])
            else:
                preds.append(r['pred_answer'])
                golds.append(r['gt_answers'][0])
        if 'heading_ocr' in ds_name:
            report = eval_heading_ocr(preds, golds)
        elif 'element_ocr' in ds_name:
            report = eval_element_ocr(preds, golds)
        elif 'webqa' in ds_name:
            report = eval_webqa(preds, golds)
    
    elif 'OCRBench' in ds_name:
        preds, golds, question_ids = [], [], []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'][0])
            question_ids.append(r['question_id'])
        report = eval_ocrbench(preds, golds, question_ids)
    
    elif 'lmms-lab/DocVQA' in ds_name:
        preds, golds, question_ids = [], [], []
        # answer_types = []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'])
            question_ids.append(r['question_id']) # 123/["xxx"]
            # answer_types .append(eval(''.join(r['question_id'].split('/')[1:]))
        
        docvqa_evaluator = DocVQAEvaluator()
        _report =  docvqa_evaluator.get_metrics(golds, preds, None)
        
        report = _report
        report['acc'] = _report['accuracy']
        
    elif 'lmms-lab/VQAv2' in ds_name:
        preds, golds, question_ids = [], [], []
        answer_types = []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'])
            question_ids.append(r['question_id'])
        
        vqav2_evaluator = VQAv2Evaluator()
        _report = vqav2_evaluator.evaluate(question_ids, preds)
        # _report = vqav2_evaluator.get_metrics(golds, preds, None)
        
        report = _report
        report['acc'] = _report['accuracy']
    
    elif 'openphish' in ds_name:
        preds, golds = [], []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'][0])
        report = eval_openphish(preds, golds)
        
    elif 'ChartQA' in ds_name:
        preds, golds = [], []
        for r in res:
            preds.append(r['pred_answer'])
            golds.append(r['gt_answers'][0])
        report = eval_chartqa(preds, golds)
        
    elif 'infoseek' in ds_name:
        preds, golds = [], []
        acc_list = []
        for r in res:
            _acc = best_subspan_em(r['pred_answer'], r['gt_answers'])
        
            acc_list.append(_acc)
        acc = np.mean(acc_list)
        report['acc'] = acc

    return report

def eval_chartqa(preds, golds):
    # return best_subspan_em(preds, golds)
    
    best_subspan_ems = []
    for gold, pred in zip(golds, preds):
        best_subspan_ems.append(best_subspan_em(pred, gold))
    
    return {
        'accuracy': np.mean(best_subspan_ems) * 100,
        'acc': np.mean(best_subspan_ems) * 100
    }


def eval_openphish(preds, golds):
    acc_lst = []
    for pred, gold in zip(preds, golds):
        pred = normalize_answer(pred)
        gold = normalize_answer(gold)
        
        acc_lst.append(pred == gold)
    return dict(
        accuracy=sum(acc_lst) / len(acc_lst) * 100,
        acc=sum(acc_lst) / len(acc_lst) * 100
    )

def get_dataset_name(f):
    if 'textvqa' in f:
        return 'textvqa'
    elif 'visualwebbench' in f:
        if 'heading_ocr' in f:
            return 'visualwebbench/heading_ocr'
        elif 'element_ocr' in f:
            return 'visualwebbench/element_ocr'
        elif 'webqa' in f:
            return 'visualwebbench/webqa'
        
        # return 'visualwebbench'
    elif 'OCRBench' in f:
        return 'OCRBench'
    elif 'lmms-lab/DocVQA' in f:
        return 'lmms-lab/DocVQA'
    elif 'lmms-lab/VQAv2' in f:
        return 'lmms-lab/VQAv2'
    elif 'openphish' in f:
        return 'openphish'
    elif 'ChartQA' in f:
        return 'ChartQA'
    elif 'infoseek' in f:
        return 'infoseek'
    
    raise ValueError(f'Unknown dataset: {f}')
    
    return None