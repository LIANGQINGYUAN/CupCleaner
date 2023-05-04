import pandas as pd
import semantic_similarity
import pickle
import diff_utils
import random
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer, BertModel, BertTokenizer, BertForSequenceClassification
from datetime import datetime 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

def remove_blank_line(code):
    code = ''.join([i for i in code.splitlines(keepends=True) if i.strip()!=''])
    return code

# main
split = ['train','valid','test']
length = []
items = [] 
data = pd.DataFrame()
for s in split:
    temp = pd.read_json(f'../ACL20/comment_update/{s}.json')
    data = pd.concat([data, temp])
    length.append(len(temp))
data.reset_index(drop=True, inplace=True)
data['split'] = ['train']*length[0]+['valid']*length[1]+['test']*length[2]


# change semantic
from diff_utils import REPLACE, REPLACE_NEW, REPLACE_END, REPLACE_NEW_DELETE_KEEP_BEFORE, REPLACE_NEW_DELETE_KEEP_AFTER, REPLACE_NEW_KEEP_AFTER, REPLACE_NEW_KEEP_BEFORE, REPLACE_OLD, REPLACE_OLD_DELETE_KEEP_AFTER, REPLACE_OLD_DELETE_KEEP_BEFORE, REPLACE_OLD_KEEP_AFTER, REPLACE_OLD_KEEP_BEFORE, INSERT, INSERT_END, INSERT_NEW, INSERT_NEW_KEEP_AFTER, INSERT_NEW_KEEP_BEFORE, INSERT_OLD, INSERT_OLD_KEEP_AFTER, INSERT_OLD_KEEP_BEFORE, DELETE, DELETE_END, KEEP, KEEP_END
keylist = [REPLACE, REPLACE_NEW, REPLACE_END, REPLACE_NEW_DELETE_KEEP_BEFORE, REPLACE_NEW_DELETE_KEEP_AFTER, REPLACE_NEW_KEEP_AFTER, REPLACE_NEW_KEEP_BEFORE, REPLACE_OLD, REPLACE_OLD_DELETE_KEEP_AFTER, REPLACE_OLD_DELETE_KEEP_BEFORE, REPLACE_OLD_KEEP_AFTER, REPLACE_OLD_KEEP_BEFORE, INSERT, INSERT_END, INSERT_NEW, INSERT_NEW_KEEP_AFTER, INSERT_NEW_KEEP_BEFORE, INSERT_OLD, INSERT_OLD_KEEP_AFTER, INSERT_OLD_KEEP_BEFORE, DELETE, DELETE_END, KEEP, KEEP_END]
def clean_tag(text):
    cleaned = []
    for i in text.split():
        if i.strip() not in keylist:
            cleaned.append(i.strip())
    return ' '.join(cleaned)

code_change_list = [' '.join(diff_utils.compute_minimal_comment_diffs(data['old_code'][i].split(),data['new_code'][i].split())[0]) for i in range(len(data))]
comm_change_list = [' '.join(diff_utils.compute_minimal_comment_diffs(data['old_comment'][i].split(),data['new_comment'][i].split())[0]) for i in range(len(data))]
code_change_list = [clean_tag(i) for i in code_change_list]
comm_change_list = [clean_tag(i) for i in comm_change_list]

a=datetime.now() 

# device = 'cuda:3'
device = 'cuda:0'
gcb = RobertaModel.from_pretrained('microsoft/graphcodebert-base').to(device)
tk_gcb = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')

# rb = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext').to(device)
# tk_comm = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

bert = BertModel.from_pretrained('bert-base-uncased').to(device)
tk_bert = BertTokenizer.from_pretrained('bert-base-uncased')
# rb = RobertaModel.from_pretrained('roberta-base').to(device)
# tk_comm = RobertaTokenizer.from_pretrained('roberta-base')
# rb = RobertaModel.from_pretrained('microsoft/graphcodebert-base').to(device)
# tk_comm = RobertaTokenizer.from_pretrained('microsoft/graphcodebert-base')

def get_list(itype):
    if itype=='code':
        return [gcb, tk_gcb, device]
    if itype=='comm':
        # return [bert, tk_bert, device]
        return [gcb, tk_gcb, device]

# token level semantic
print("get code and comment semantic")
semantics_comm_1 = semantic_similarity.get_semantic(data['old_comment'].tolist(), get_list('comm'))
semantics_comm_2 = semantic_similarity.get_semantic(data['new_comment'].tolist(), get_list('comm'))
semantics_code_1 = semantic_similarity.get_semantic(data['old_code'].tolist(),get_list('code'))
semantics_code_2 = semantic_similarity.get_semantic(data['new_code'].tolist(),get_list('code'))
# sent level
sent_semantics_comm_1 = semantic_similarity.get_sent_semantic(data['old_comment'].tolist())
sent_semantics_comm_2 = semantic_similarity.get_sent_semantic(data['new_comment'].tolist())




print("get diff semantic")
semantics_diff_code = semantic_similarity.get_semantic(code_change_list,get_list('code'))
semantics_diff_comm = semantic_similarity.get_semantic(comm_change_list,get_list('code'))

with open("acl20_semantics.pkl",'wb') as f:
    pickle.dump({"semantics_code_1":semantics_code_1,
                "semantics_code_2":semantics_code_2,
                "semantics_comm_1":semantics_comm_1,
                "semantics_comm_2":semantics_comm_2,
                "semantics_diff_code":semantics_diff_code,
                "semantics_diff_comm":semantics_diff_comm,
                "sent_semantics_comm_1":sent_semantics_comm_1,
                "sent_semantics_comm_2":sent_semantics_comm_2,
                "split":data['split'].tolist()},f)
    

b=datetime.now() 
print('time: ',(b-a).seconds)
# time:  190