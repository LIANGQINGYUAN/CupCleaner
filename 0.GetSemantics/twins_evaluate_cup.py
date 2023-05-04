import pandas as pd
import semantic_similarity
import pickle
import diff_utils
import random
import numpy as np
import json
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
for s in split:
    num = 0
    with open(f"../CUPData/dataset/{s}.jsonl", "r+", encoding="utf8") as f:
        for item in f.readlines():
            items.append(json.loads(item))
            num+=1
        length.append(num)
data = pd.DataFrame(items)
data['split'] = ['train']*length[0]+['valid']*length[1]+['test']*length[2]

data['src_method'] = data['src_method'].apply(remove_blank_line)
data['dst_method'] = data['dst_method'].apply(remove_blank_line)

print("original length: ",len(data))

invalid = [0]*len(data)
for i in range(len(data)):
    if data['src_method'][i] == data['dst_method'][i]:
        invalid[i]=1
    if data['src_desc'][i] == data['dst_desc'][i]:
        invalid[i]=1
data['invalid'] = invalid
data = data.loc[data.invalid==0]
data.reset_index(drop=True, inplace=True)
print("delete invalid: ",len(data))

from diff_utils import REPLACE, REPLACE_NEW, REPLACE_END, REPLACE_NEW_DELETE_KEEP_BEFORE, REPLACE_NEW_DELETE_KEEP_AFTER, REPLACE_NEW_KEEP_AFTER, REPLACE_NEW_KEEP_BEFORE, REPLACE_OLD, REPLACE_OLD_DELETE_KEEP_AFTER, REPLACE_OLD_DELETE_KEEP_BEFORE, REPLACE_OLD_KEEP_AFTER, REPLACE_OLD_KEEP_BEFORE, INSERT, INSERT_END, INSERT_NEW, INSERT_NEW_KEEP_AFTER, INSERT_NEW_KEEP_BEFORE, INSERT_OLD, INSERT_OLD_KEEP_AFTER, INSERT_OLD_KEEP_BEFORE, DELETE, DELETE_END, KEEP, KEEP_END
keylist = [REPLACE, REPLACE_NEW, REPLACE_END, REPLACE_NEW_DELETE_KEEP_BEFORE, REPLACE_NEW_DELETE_KEEP_AFTER, REPLACE_NEW_KEEP_AFTER, REPLACE_NEW_KEEP_BEFORE, REPLACE_OLD, REPLACE_OLD_DELETE_KEEP_AFTER, REPLACE_OLD_DELETE_KEEP_BEFORE, REPLACE_OLD_KEEP_AFTER, REPLACE_OLD_KEEP_BEFORE, INSERT, INSERT_END, INSERT_NEW, INSERT_NEW_KEEP_AFTER, INSERT_NEW_KEEP_BEFORE, INSERT_OLD, INSERT_OLD_KEEP_AFTER, INSERT_OLD_KEEP_BEFORE, DELETE, DELETE_END, KEEP, KEEP_END]
def clean_tag(text):
    cleaned = []
    for i in text.split():
        if i.strip() not in keylist:
            cleaned.append(i.strip())
    return ' '.join(cleaned)

code_change_list = [' '.join(diff_utils.compute_minimal_comment_diffs(data['src_method'][i].split(),data['dst_method'][i].split())[0]) for i in range(len(data))]
comm_change_list = [' '.join(diff_utils.compute_minimal_comment_diffs(data['src_desc'][i].split(),data['dst_desc'][i].split())[0]) for i in range(len(data))]
code_change_list = [clean_tag(i) for i in code_change_list]
comm_change_list = [clean_tag(i) for i in comm_change_list]

a=datetime.now() 

# device = 'cuda:3'
device = 'cuda:2'
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
semantics_code_1 = semantic_similarity.get_semantic(data['src_method'].tolist(),get_list('code'))
semantics_code_2 = semantic_similarity.get_semantic(data['dst_method'].tolist(),get_list('code'))
semantics_comm_1 = semantic_similarity.get_semantic(data['src_desc'].tolist(),get_list('comm'))
semantics_comm_2 = semantic_similarity.get_semantic(data['dst_desc'].tolist(),get_list('comm'))
# sent level
sent_semantics_comm_1 = semantic_similarity.get_sent_semantic(data['src_desc'].tolist())
sent_semantics_comm_2 = semantic_similarity.get_sent_semantic(data['dst_desc'].tolist())

# change semantic
semantics_diff_code = semantic_similarity.get_semantic(code_change_list,get_list('code'))
semantics_diff_comm = semantic_similarity.get_semantic(comm_change_list,get_list('code'))

with open("cup_semantics.pkl",'wb') as f:
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
# time:  4782