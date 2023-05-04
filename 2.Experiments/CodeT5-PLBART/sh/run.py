import os
import threading

def run(gpuid, subtask):
    os.system(f'bash run.sh "{gpuid}" "{subtask}"')



ids = [0]
thresholds = [0.02]

prefix = 'diffc12'
comm = 'gcb'
code = 'gcb'
sub_tasks = []
for threshold in thresholds:
    threshold_tag = str(threshold)[str(threshold).index('.')+1:]
    suffix = comm+code+threshold_tag
    sub_tasks.append(prefix+suffix)
for i, s in zip(ids, sub_tasks):
    s = 'AAAI21_refine'+s
    t = threading.Thread(target=run, args=(i,s,))
    t.start()

ids = [1]
thresholds = [0.03]
prefix = 'diffc12'
comm = 'gcb'
code = 'gcb'
sub_tasks = []
for threshold in thresholds:
    threshold_tag = str(threshold)[str(threshold).index('.')+1:]
    suffix = comm+code+threshold_tag
    sub_tasks.append(prefix+suffix)
for i, s in zip(ids, sub_tasks):
    s = 'ACL20_refine'+s
    t = threading.Thread(target=run, args=(i,s,))
    t.start()

# t = threading.Thread(target=run, args=(6,'AAAI21',))
# t.start()

# t = threading.Thread(target=run, args=(2,'ASE20',))
# t.start()

# t = threading.Thread(target=run, args=(4,'AAAI21_randomdiffc2',))
# t.start()

# t = threading.Thread(target=run, args=(0, 'ASE20_randomdiffc2',))
# t.start()



