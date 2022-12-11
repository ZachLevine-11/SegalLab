import pandas as pd
from LabQueue.qp import qp
from LabUtils.addloglevels import sethandlers
from GeneticsPipeline.config import gencove_logs_path
import os
from q_loop import q_loop
from modified_tom_functions import getsigunique

##double queuing, use fakeqp for the inner one and real qp for the outer one
##warning: correcting for age and gender without a savename will triple queue. probably best to use a savefile and do that first
def q_generate_prs_matrix(loader, index_is_10k, test = "m", duplicate_rows = "mean", use_clustering = True, use_imputed = True, correct_for_age_gender = False, saveName = None, get_data_args = None, tailsTest = "rightLeft", usePath = False, prs_path = "/home/zacharyl/Desktop/intermed_prs.csv", random_shuffle_prsLoader = False, use_prsLoader = True):
    os.chdir(gencove_logs_path)
    #sethandlers()
    with qp(jobname="q", delay_batch = 30, _suppress_handlers_warning =True) as q:
        q.startpermanentrun()
        ## create the qp before doing anything with big variables, and delete everything that isn't required before calling qp
        if use_prsLoader:
            prses = getsigunique()  # subset prses here, i.e, [0:120],to reduce matrix size
        else:
            prses = pd.read_csv("/net/mraid08/export/jasmine/zach/scores/score_results/SOMAscan/scores_all_raw.csv").set_index("RegistrationCode").columns
        fundict = {}
        ###We also care about the column names
        varNames = {}
        ##each batch is one prs
        for prs_id in range(len(prses)):
            ##empty string matches positional argument for PRSpath
            fundict[prs_id] = q.method(q_loop, (loader, index_is_10k, test, duplicate_rows, usePath, prs_path, prses[prs_id], use_clustering, use_imputed, correct_for_age_gender, saveName, get_data_args, tailsTest, random_shuffle_prsLoader, use_prsLoader))  ##test can be "t" for t test or "r" for regression))
            print("now onto prs: ", prs_id)
        fundict = {k: q.waitforresult(v) for k, v in fundict.items()}
    for k,v in fundict.copy().items(): ##catch broken PRSes, don't iterate over original dictionary
        if v is None:
            del fundict[k]
    final_res = pd.concat(fundict.values(), axis = 1)
    #final_res = final_res.loc[:,~final_res.columns.duplicated()] ##drop the duplicate indices we've accumulated atthis point
    return final_res

