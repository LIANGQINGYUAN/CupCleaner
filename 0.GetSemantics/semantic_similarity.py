from transformers import RobertaModel, RobertaTokenizer, BertModel, BertTokenizer, BertForSequenceClassification
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
import numpy as np
from sentence_transformers import SentenceTransformer
sent_model = SentenceTransformer('all-MiniLM-L6-v2')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

b_size = 20
def get_semantic(text_list, m_list):
    # test_example = ['hello world']*b_size
    model = m_list[0]
    tk = m_list[1]
    device = m_list[2]
    batch_num = len(text_list)/b_size
    if batch_num<=1:
        tokenized_list = [tk.encode(j, max_length=512, truncation=True, pad_to_max_length = True) for j in text_list] 
        return model(torch.as_tensor(tokenized_list).to(device)).pooler_output.detach().tolist()
    else:
        output = []
        if len(text_list) % b_size == 0:
            batchs = int(batch_num)
        else:
            batchs = int(batch_num)+1
        for i in tqdm(range(batchs)):
            # tokenized_list = [tk.encode(j, max_length=512, truncation=True, pad_to_max_length = True) for j in text_list[b_size*i:b_size*(i+1)]] 
            tokenized_list = tk(text_list[b_size*i:b_size*(i+1)], max_length=512, truncation=True, padding=True, return_tensors="pt").to(device)
            output.append(model(**tokenized_list).pooler_output.detach())
        return torch.cat(output).tolist()
    
    
def get_sent_semantic(text_list):
    return sent_model.encode(text_list).tolist()