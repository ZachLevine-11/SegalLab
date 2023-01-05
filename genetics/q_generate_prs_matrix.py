import pandas as pd
from LabQueue.qp import qp
from LabUtils.addloglevels import sethandlers
from GeneticsPipeline.config import gencove_logs_path
import os
from q_loop import q_loop
from modified_tom_functions import getnotrawPrses

##double queuing, use fakeqp for the inner one and real qp for the outer one
##warning: correcting for age and gender without a savename will triple queue. probably best to use a savefile and do that first
def q_generate_prs_matrix(test = "m", duplicate_rows = "mean", saveName = None, tailsTest = "rightLeft", random_shuffle_prsLoader = False, use_prsLoader = True):
    os.chdir("/net/mraid08/export/mb/logs/")
    #sethandlers()
    with qp(jobname=saveName, delay_batch = 30, _suppress_handlers_warning =True) as q:
        q.startpermanentrun()
        ## create the qp before doing anything with big variables, and delete everything that isn't required before calling qp
        if use_prsLoader:
            prses = getnotrawPrses() # subset prses here, i.e, [0:120],to reduce matrix size
        else:
            prses = pd.read_csv("/net/mraid08/export/jasmine/zach/scores/score_results/SOMAscan/scores_all_raw.csv").set_index("RegistrationCode").columns
        fundict = {}
        ###We also care about the column names
        varNames = {}
        ##each batch is one prs
        for prs_id in range(len(prses)):
            ##empty string matches positional argument for PRSpath
            fundict[prs_id] = q.method(q_loop, (test, duplicate_rows, prses[prs_id], saveName, tailsTest, random_shuffle_prsLoader, use_prsLoader, prs_id))  ##test can be "t" for t test or "r" for regression))
            print("now onto prs: ", prs_id)
        for k, v in fundict.items():
            try:
                ##For each PRS id, a set of the correponding P value for each column
                fundict[k] = q.waitforresult(v)
            except Exception:
                fundict[k] = None ##broken PRSes
    for k,v in fundict.copy().items(): ##catch broken PRSes, don't iterate over original dictionary
        if v is None:
            del fundict[k]
    final_res = pd.concat(fundict.values(), axis = 1, join = "outer")
    #final_res = final_res.loc[:,~final_res.columns.duplicated()] ##drop the duplicate indices we've accumulated atthis point
    return final_res

