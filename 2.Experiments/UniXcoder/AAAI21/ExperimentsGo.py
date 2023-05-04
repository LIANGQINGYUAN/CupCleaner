import os
import threading

def run(gpuid, task_tag, data_dir, train_data, valid_data, test_data, model_name):
    os.system(f'bash run.sh "{gpuid}" "{task_tag}" "{data_dir}" "{train_data}" "{valid_data}" "{test_data}" "{model_name}"')

# AAAI21-unixcoder
gpuid = 0
tag = 'AAAI21'
data_dir = '../CodeT5/data/comment_update/{}'.format(tag)
task_tag = tag + ''
traindata = 'train.json'
validdata = 'valid.json'
testdata = 'test.json'
modelname = 'microsoft/unixcoder-base'
t = threading.Thread(target=run, args=(gpuid, task_tag, data_dir, traindata, validdata, testdata, modelname))
t.start()

gpuid = 1
task_tag = tag + '_random'
traindata = 'train_randomdiffc11.json'
validdata = 'valid_randomdiffc11.json'
t = threading.Thread(target=run, args=(gpuid, task_tag, data_dir, traindata, validdata, testdata, modelname))
t.start()

gpuid = 2
task_tag = tag + '_cleaned'
traindata = 'train_refinediffc11gcbgcb03.json'
validdata = 'valid_refinediffc11gcbgcb03.json'
t = threading.Thread(target=run, args=(gpuid, task_tag, data_dir, traindata, validdata, testdata, modelname))
t.start()
