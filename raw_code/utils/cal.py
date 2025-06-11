import numpy as np
from xgboost import XGBClassifier

def prepare_data(details, include_tags='all'):
    
    if include_tags == 'all':
        include_tags = ['img', 'text', 'both', 'neither', 'same', 'undefined']
    elif type(include_tags) == list:
        include_tags = include_tags
        
    img_conf = []
    text_conf = []
    img_text_conf = []
    use_imgs = []
    use_texts = []
    img_text_conf_diff = []
    text_img_conf_diff = []
    img_accs = []
    text_accs = []
    img_text_accs = []
    
    for que_id, v in details.items():
        
        # if v['status'] not in ['img', 'text', 'both']:
        # if v['status'] not in ['img', 'text', 'both', 'neither']:
        if v['status'] not in include_tags:
        # if v['status'] not in ['img', 'text']:
        # if v['status'] not in ['both']:
            continue
        
        img_conf.append(v['img_conf'])
        text_conf.append(v['text_conf'])
        img_text_conf.append(v['img_text_conf'])

        
        use_imgs.append(v['img_score'])
        use_texts.append(v['text_score'])
        
        img_accs.append(v['img_acc'])
        text_accs.append(v['text_acc'])
        img_text_accs.append(v['img_text_acc'])
        
        # for k in status_group_masks.keys():
        #     if k != v['status']:
        #         status_group_masks[k].append(False)
        #     else:
        #         status_group_masks[k].append(True)
        
        img_text_conf_diff.append((v['img_conf'] - v['text_conf'])) # best
        
    
    img_conf = np.array(img_conf)
    text_conf = np.array(text_conf)
    img_text_conf = np.array(img_text_conf)
    use_imgs = np.array(use_imgs)
    use_texts = np.array(use_texts)
    img_text_conf_diff = np.array(img_text_conf_diff)
    text_img_conf_diff = np.array(text_img_conf_diff)
    img_accs = np.array(img_accs)
    text_accs = np.array(text_accs)
    img_text_accs = np.array(img_text_accs)
    
    x = np.stack([img_conf, text_conf, img_text_conf], axis=1)
    accs = np.stack([img_accs, text_accs, img_text_accs], axis=1)
    y = []
    best_accs = []
    real_accs = []
    for img_acc, text_acc, img_text_acc in zip(img_accs, text_accs, img_text_accs):
        if img_acc > text_acc and img_acc > img_text_acc:
            y.append(0)
            best_accs.append(img_acc)
        elif text_acc > img_acc and text_acc > img_text_acc:
            y.append(1)
            best_accs.append(text_acc)
        elif img_acc == text_acc and img_acc > img_text_acc:
            _y = np.random.choice([0, 1])
            y.append(_y)
            best_accs.append(img_acc)
        else:
            y.append(2)
            best_accs.append(img_text_acc)

                
        real_accs.append(img_text_acc)
        
    real_accs = np.array(real_accs)
    best_accs = np.array(best_accs)
    y = np.array(y)
    
    return {
        'x': x,
        'accs': accs,
        'y': y,
        'best_accs': best_accs,
        'real_accs': real_accs,
    }
    
def recal_acc(x, y, accs, seed=0, train_ratio=0.5):
    np.random.seed(seed)
    all_indexs = np.random.permutation(len(x))
    train_indexs = all_indexs[:int(len(x)*train_ratio)]
    test_indexs = all_indexs[int(len(x)*train_ratio):]
    
    x_train = x[train_indexs]
    x_test = x[test_indexs]
    y_train = np.array(y)[train_indexs]
    y_test = np.array(y)[test_indexs]
    
    accs_train = accs[train_indexs]
    accs_test = accs[test_indexs]
    
    # model = XGBClassifier(eval_metric='mlogloss', random_state=seed, n_jobs=1, colsample_bytree=1.0, subsample=1.0, max_depth=20, n_estimators=10)
    model = XGBClassifier(eval_metric='mlogloss', random_state=seed)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    new_accs = [ accs_test[i][y_pred[i]] for i in range(len(y_pred)) ]
    
    return new_accs, train_indexs, test_indexs
    
    