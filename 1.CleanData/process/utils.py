
import numpy as np
from tqdm import tqdm
import json


from diff_utils import REPLACE, REPLACE_NEW, REPLACE_END, REPLACE_NEW_DELETE_KEEP_BEFORE, REPLACE_NEW_DELETE_KEEP_AFTER, REPLACE_NEW_KEEP_AFTER, REPLACE_NEW_KEEP_BEFORE, REPLACE_OLD, REPLACE_OLD_DELETE_KEEP_AFTER, REPLACE_OLD_DELETE_KEEP_BEFORE, REPLACE_OLD_KEEP_AFTER, REPLACE_OLD_KEEP_BEFORE, INSERT, INSERT_END, INSERT_NEW, INSERT_NEW_KEEP_AFTER, INSERT_NEW_KEEP_BEFORE, INSERT_OLD, INSERT_OLD_KEEP_AFTER, INSERT_OLD_KEEP_BEFORE, DELETE, DELETE_END, KEEP, KEEP_END
keylist = [REPLACE, REPLACE_NEW, REPLACE_END, REPLACE_NEW_DELETE_KEEP_BEFORE, REPLACE_NEW_DELETE_KEEP_AFTER, REPLACE_NEW_KEEP_AFTER, REPLACE_NEW_KEEP_BEFORE, REPLACE_OLD, REPLACE_OLD_DELETE_KEEP_AFTER, REPLACE_OLD_DELETE_KEEP_BEFORE, REPLACE_OLD_KEEP_AFTER, REPLACE_OLD_KEEP_BEFORE, INSERT, INSERT_END, INSERT_NEW, INSERT_NEW_KEEP_AFTER, INSERT_NEW_KEEP_BEFORE, INSERT_OLD, INSERT_OLD_KEEP_AFTER, INSERT_OLD_KEEP_BEFORE, DELETE, DELETE_END, KEEP, KEEP_END]
from nltk.corpus import stopwords
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','``',"''","'",'`']
stops = list(set(stopwords.words("english")))

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos

def lcs_dp(input_x, input_y):
        # input_y as column, input_x as row
        dp = [([0] * (len(input_y)+1)) for i in range(len(input_x)+1)]
        for i in range(1, len(input_x)+1):
            for j in range(1, len(input_y)+1):
                if i == 0 or j == 0:  
                        dp[i][j] = 1
                elif input_x[i-1] == input_y[j-1]:  
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:  
                    dp[i][j] = max(dp[i - 1][j], dp[i][j -1])
        # for dp_line in dp:
        #     print(dp_line)
        return dp[-1][-1]

def overlap_metrixv2(diff_comm, diff_code, long_comm):
    diff_comm = [i for i in diff_comm.split() if i not in keylist+english_punctuations+stops]
    diff_code = [i for i in diff_code.split() if i not in keylist]
    if long_comm:
        # print(diff_comm, diff_code)
        if len(diff_comm) == 0:
            return 0
        max_overlap = []
        for i in diff_comm:
            match2code = []
            for j in diff_code:
                lcs_comm_i = lcs_dp(i.lower(),j.lower())
                match2code.append(lcs_comm_i/len(i))
            max_overlap.append(max(match2code))
        return sum(max_overlap)/len(max_overlap)
    else:
        return 0
    
    
def diffsim(diffcode, diffcomm, long_comm):
    if long_comm:
        return cos_sim(diffcode, diffcomm)
    else:
        return 0
    

def get_refine_data(data, metrics_list, m_values, tgt_dir='./', r='', merge=''):
    print("Length of train: ", len(data.loc[data['split']=='train']))
    print("Length of valid: ", len(data.loc[data['split']=='valid']))
    rm = []
    assert len(metrics_list) == len(m_values)
    for i in range(len(data)):
        if merge=='':
            if all([metrics[i]>m_v for metrics, m_v in zip(metrics_list, m_values)]):
                rm.append(1)
            else:
                rm.append(0)
                
    data['rm'] = rm
    data_cleaned = data.loc[data['rm']==1]
    
    df = data_cleaned
    df = df.rename(columns={'src_desc':'old_comment',"dst_desc":"new_comment",\
                            "src_method":"old_code", "dst_method":"new_code" })
    df_train = df.loc[df['split']=='train']
    df_valid = df.loc[df['split']=='valid']
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)
    df_train = df_train.astype(object)
    df_valid = df_valid.astype(object)

    with open(f"{tgt_dir}/train_refine{r}.json",'w+') as t, open(f"{tgt_dir}/valid_refine{r}.json",'w+') as v:
        for i in tqdm(range(len(df_train))):
            item = df_train.iloc[i,:].to_dict()
            t.write(json.dumps(item)+'\n')
        for i in tqdm(range(len(df_valid))):
            item = df_valid.iloc[i,:].to_dict()
            v.write(json.dumps(item)+'\n')
    return df