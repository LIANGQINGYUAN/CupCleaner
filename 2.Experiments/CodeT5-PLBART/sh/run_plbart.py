import os
import threading

def run(gpuid, subtask):
    os.system(f'bash run_plbart.sh "{gpuid}" "{subtask}"')



# t = threading.Thread(target=run, args=(1,'ACL20',))
# t.start()
# t = threading.Thread(target=run, args=(1,'ACL20_randomdiffc11',))
# t.start()
# t = threading.Thread(target=run, args=(2,'ACL20_refinediffc11gcbgcb04',))
# t.start()


t = threading.Thread(target=run, args=(0,'AAAI21',))
t.start()
t = threading.Thread(target=run, args=(1,'AAAI21_randomdiffc11',))
t.start()
# t = threading.Thread(target=run, args=(6,'AAAI21_refinediffc11gcbgcb03',))
# t.start()


# t = threading.Thread(target=run, args=(3,'ASE20',))
# t.start()
# t = threading.Thread(target=run, args=(4, 'ASE20_randomdiffc11',))
# t.start()
# t = threading.Thread(target=run, args=(5,'ASE20_refinediffc11t02gcbgcb02',))
# t.start()

