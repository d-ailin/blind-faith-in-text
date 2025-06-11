import re

def extract_number(sentence):
    # \" wit the first end quote \"
    match = re.search(r'\"([^"]+)\"', sentence)
    if match:
        return match.group(1)
    
    match = re.search(r"\'([^']+)\'", sentence)
    if match:
        return match.group(1)
    
    return sentence

def process_ans(sentence):
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
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
    # one, two, three, four, five, six, seven, eight, nine, ten
    manualMap = { 'none': '0',
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
    
    articles     = ['a',
                                'an',
                                'the'
                            ]
    
    def processDigitArticle(inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = manualMap.setdefault(word, word)
            if word not in articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in contractions:
                outText[wordId] = contractions[word]
        outText = ' '.join(outText)
        return outText
    
    return processDigitArticle(sentence)

def clean_ans(ans):
    ans = ans.strip()
    # if more strict exact match
    if len(ans) > 0 and ans[-1] == '.':
        ans = ans[:-1]
    
    return ans
    
    
    ans = ans.lower().strip()
    
    if '\"' in ans or '\'' in ans:
        return extract_number(ans)   
    
    replaces = [
        'the number in the image is',
        'The number in the image is',
        'The text content in the red bounding box is:'.lower(),
    ]
    
    for r in replaces:
        ans = ans.replace(r, '')
    
    if len(ans) > 0 and ans[-1] == '.':
        ans = ans[:-1]
        
    ans = process_ans(ans)
    
    # if "no" or "yes" as a word in the sentence, then we just use "no" or "yes" as the answer
    # split with space or comma
    words = ans.split()
    if 'no' in words or 'no,' in words:
        ans = 'no'
    if 'yes' in words or 'yes,' in words:
        ans = 'yes'
    # ans = ans
    
    return ans.strip()

# STATUS = {
#     'use_img': 0,
#     'use_text': 1,
#     'neither': 2,
#     'same': 3
# }

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

def get_modality_prefer_score(output):
    img_text = output['img_text']
    only_text = output['only_text']
    only_img = output['only_img']
    
    data = {}
    
    # only consider the question_id existing in all three
    img_text_que_ids = []
    for item in img_text:
        question_id = item['question_id']
        img_text_que_ids.append(question_id)
    img_que_ids = []
    for item in only_img:
        question_id = item['question_id']
        img_que_ids.append(question_id)
    text_que_ids = []
    for item in only_text:
        question_id = item['question_id']
        text_que_ids.append(question_id)
    
    common_que_ids = set(img_text_que_ids) & set(img_que_ids) & set(text_que_ids)
    
    for item in img_text:
        question_id = item['question_id']
        if question_id not in common_que_ids:
            continue
        
        data[question_id] = {
            'ans:img_text': clean_ans(item['pred_answer']),
            'acc:img_text': item['acc'],
            'input': item['question']
        }
    
    for item in only_text:
        question_id = item['question_id']
        
        if question_id not in data:
            continue
        
        data[question_id]['ans:only_text'] = clean_ans(item['pred_answer'])
        data[question_id]['acc:only_text'] = item['acc']
    
    for item in only_img:
        question_id = item['question_id']
        
        if question_id not in data:
            continue
        
        data[question_id]['ans:only_img'] = clean_ans(item['pred_answer'])
        data[question_id]['acc:only_img'] = item['acc']
    
    
    
    for k, v in data.items():
        data[k]['ans:img_text==text'] = v['ans:only_text'] == v['ans:img_text']
        data[k]['ans:img_text==img'] = v['ans:only_img'] == v['ans:img_text']
        data[k]['ans:img==text'] = v['ans:only_text'] == v['ans:only_img']
        
    use_img_count = 0
    use_text_count = 0
    neither_count = 0
    same_count = 0
    both_count = 0
    undefined_count = 0
    
    details = {}
    corse_img_scores = []
        
    for k, v in data.items():
        details[k] = {
            'img_acc': v['acc:only_img'],
            'text_acc': v['acc:only_text'],
            'img_text_acc': v['acc:img_text'],
            'ans:only_text': v['ans:only_text'],
            'ans:only_img': v['ans:only_img'],
            'ans:img_text': v['ans:img_text'],
            'input': v['input']
        }
        
        details[k].update({
            'img==text': v['ans:img==text'],
        })
        
        if v['ans:img_text==text'] and v['ans:img_text==img']:
            same_count += 1
            details[k].update({
                'status': 'same',
                'img_score': np.NaN,
                'text_score': np.NaN,
            })
        elif v['ans:img_text==text']:
            use_text_count += 1
            details[k].update({
                'status': 'text',
                'img_score': 0,
                'text_score': 1
            })
            corse_img_scores.append(0)
        elif v['ans:img_text==img']:
            use_img_count += 1
            details[k].update({
                'status': 'img',
                'img_score': 1,
                'text_score': 0
            })
            corse_img_scores.append(1)
        else:
            
            # text_score = similarity_based_on_substring(v['ans:only_text'], v['ans:img_text'])
            # img_score = similarity_based_on_substring(v['ans:only_img'], v['ans:img_text'])
            
            text_score = similarity_score(v['ans:only_text'], v['ans:img_text'])
            img_score = similarity_score(v['ans:only_img'], v['ans:img_text'])
            
            if text_score > 0 and img_score > 0:
                score = img_score / (text_score + img_score)
                
                
                both_count += 1
                
                details[k].update({
                    'status': 'both',
                    'img_score': img_score,
                    'text_score': text_score
                })
                corse_img_scores.append(img_score)
            elif text_score == 0 and img_score == 0:
                score = -1
            
                neither_count += 1
                details[k].update({
                    'status': 'neither',
                    'img_score': img_score,
                    'text_score': text_score
                })
            else:
                undefined_count += 1
                details[k].update({
                    'status': 'undefined',
                    'img_score': img_score,
                    'text_score': text_score
                })
    
    report = {
        'use_img_count': use_img_count,
        'use_text_count': use_text_count,
        'neither_count': neither_count,
        'both_count': both_count,
        'same_count': same_count,
        'undefined_count': undefined_count,
        'img_text_ratio': use_img_count / (use_img_count + use_text_count),
        'corse_img_text_ratio': sum(corse_img_scores) / len(corse_img_scores),
        'details': details
    }
    
    return report