import os
import threading

def run(gpuid, subtask):
    os.system(f'bash run.sh "{gpuid}" "{subtask}"')

# training in 622 mode
validdata = 'valid_samplecleaned.json'
testdata = 'test_samplecleaned.json'

# subtask = 'ACL20_622noising'
# traindata = 'train_samplenoising.json'
# t = threading.Thread(target=run, args=(5,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ACL20_622'
# traindata = 'train_samplecleaned.json'
# t = threading.Thread(target=run, args=(6,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_712noising7'
# traindata = 'train_samplenoising07.json'
# t = threading.Thread(target=run, args=(0,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_712noising5'
# traindata = 'train_samplenoising05.json'
# t = threading.Thread(target=run, args=(2,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_622noisingL065'
# traindata = 'train_samplenoisingL065.json'
# t = threading.Thread(target=run, args=(3,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_622'
# traindata = 'train_samplecleaned.json'
# t = threading.Thread(target=run, args=(4,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ACL20_622noisingL065'
# traindata = 'train_samplenoisingL065.json'
# t = threading.Thread(target=run, args=(0,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ACL20_622'
# traindata = 'train_samplecleaned.json'
# t = threading.Thread(target=run, args=(1,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()



# subtask = 'ACL20_622noisingLfix065'
# traindata = 'train_samplenoisingLfix065.json'
# t = threading.Thread(target=run, args=(2,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_622noisingLfix065'
# traindata = 'train_samplenoisingLfix065.json'
# t = threading.Thread(target=run, args=(5,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()


# subtask = 'ACL20_622noising'
# traindata = 'train_samplenoising.json'
# t = threading.Thread(target=run, args=(0,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()


# subtask = 'ACL20_523noisingLfix065v2'
# traindata = 'train_samplenoisingLfix065v2.json'
# t = threading.Thread(target=run, args=(0,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ACL20_nonoising'
# traindata = 'train_samplecleaned.json'
# t = threading.Thread(target=run, args=(1,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ACL20_noising'
# traindata = 'data_noising.json'
# t = threading.Thread(target=run, args=(2,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ACL20_noisingcleaned'
# traindata = 'train_samplenoising.json'
# t = threading.Thread(target=run, args=(3,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_nonoising'
# traindata = 'train_samplecleaned.json'
# t = threading.Thread(target=run, args=(4,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_noising'
# traindata = 'data_noising.json'
# t = threading.Thread(target=run, args=(5,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_noisingcleaned'
# traindata = 'train_samplenoising.json'
# t = threading.Thread(target=run, args=(6,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

subtask = 'ASE20_nonoising'
traindata = 'train_samplecleaned.json'
t = threading.Thread(target=run, args=(4,f'{subtask},{traindata},{validdata},{testdata}',))
t.start()

subtask = 'ASE20_noising'
traindata = 'data_noising.json'
t = threading.Thread(target=run, args=(5,f'{subtask},{traindata},{validdata},{testdata}',))
t.start()

subtask = 'ASE20_noisingcleaned'
traindata = 'train_samplenoising.json'
t = threading.Thread(target=run, args=(6,f'{subtask},{traindata},{validdata},{testdata}',))
t.start()


# subtask = 'AAAI21_523noisingLfix065'
# traindata = 'train_samplenoisingLfix065.json'
# t = threading.Thread(target=run, args=(3,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()
# subtask = 'AAAI21_523noising'
# traindata = 'train_samplenoising.json'
# t = threading.Thread(target=run, args=(4,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_523'
# traindata = 'train_samplecleaned.json'
# t = threading.Thread(target=run, args=(5,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ASE20_712noisingL07'
# traindata = 'train_samplenoisingL07.json'
# t = threading.Thread(target=run, args=(2,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ASE20_712'
# traindata = 'train_samplecleaned.json'
# t = threading.Thread(target=run, args=(6,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_622noising'
# traindata = 'train_samplenoising.json'
# t = threading.Thread(target=run, args=(1,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_622'
# traindata = 'train_samplecleaned.json'
# t = threading.Thread(target=run, args=(1,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ASE20_622noising'
# traindata = 'train_samplenoising.json'
# t = threading.Thread(target=run, args=(5,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ASE20_622'
# traindata = 'train_samplecleaned.json'
# t = threading.Thread(target=run, args=(6,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()


# subtask = 'AAAI21_sampled'
# traindata = 'train_sampled.json'
# validdata = 'valid_sampled.json'
# testdata = 'valid_refinediffc11gcbgcb03.json'
# t = threading.Thread(target=run, args=(0,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'AAAI21_sampledclean'
# traindata = 'train_sampledclean.json'
# validdata = 'valid_sampledclean.json'
# testdata = 'valid_refinediffc11gcbgcb03.json'
# t = threading.Thread(target=run, args=(1,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ACL20_sampled'
# traindata = 'train_sampled.json'
# validdata = 'valid_sampled.json'
# testdata = 'valid_refinediffc11gcbgcb04.json'
# t = threading.Thread(target=run, args=(3,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ACL20_sampledclean'
# traindata = 'train_sampledclean.json'
# validdata = 'valid_sampledclean.json'
# testdata = 'valid_refinediffc11gcbgcb04.json'
# t = threading.Thread(target=run, args=(4,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ASE20_sampled'
# traindata = 'train_sampled.json'
# validdata = 'valid_sampled.json'
# testdata = 'valid_refinediffc11t02gcbgcb02.json'
# t = threading.Thread(target=run, args=(5,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = 'ASE20_sampledclean'
# traindata = 'train_sampledclean.json'
# validdata = 'valid_sampledclean.json'
# testdata = 'valid_refinediffc11t02gcbgcb02.json'
# t = threading.Thread(target=run, args=(6,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# tag = 'test'

# subtask = f'AAAI21_sampledclean{tag}'
# traindata = 'train_refinediffc11gcbgcb03.json'
# validdata = 'valid_refinediffc11gcbgcb03.json'
# testdata = 'test_cleaned.json'
# t = threading.Thread(target=run, args=(1,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()


# subtask = f'ACL20_sampledclean{tag}'
# traindata = 'train_refinediffc11gcbgcb04.json'
# validdata = 'valid_refinediffc11gcbgcb04.json'
# testdata = 'test_cleaned.json'
# t = threading.Thread(target=run, args=(2,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()

# subtask = F'ASE20_sampledclean{tag}'
# traindata = 'train_refinediffc11t02gcbgcb02.json'
# validdata = 'valid_refinediffc11t02gcbgcb02.json'
# testdata = 'test_cleaned.json'
# t = threading.Thread(target=run, args=(3,f'{subtask},{traindata},{validdata},{testdata}',))
# t.start()
