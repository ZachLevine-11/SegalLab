import pandas as pd
from q_loop import q_loop
from modified_tom_functions import getsigunique

##it turns out that the real queue is faster than this loop, but for whatever reason it often doesn't work
##use fakeqp in the q loop
def loop_generate_prs_matrix(loader, index_is_10k, test = "m", duplicate_rows = "mean", use_clustering = True, use_imputed = True, correct_for_age_gender = False, saveName = None, get_data_args = None, tailsTest = "rightLeft", usePath = False, prs_path = "/home/zacharyl/Desktop/intermed_prs.csv", random_shuffle_prsLoader = False, use_prsLoader = True):
    fundict = {}
    ###We also care about the column names
    if use_prsLoader:
        prses = getsigunique()#[0:120]
    else:
        prses = pd.read_csv("/net/mraid08/export/jasmine/zach/scores/score_results/SOMAscan/scores_all_raw.csv").set_index("RegistrationCode").columns
    ##each batch is one prs
    for prs_id in range(len(prses)):
        print("now starting prs number: ", prs_id, "/" , str(len(prses)))
        #empty string matches positional argument for PRSpath, and the 0 fixes the prs id being an array
        fundict[prs_id] = q_loop(loader, index_is_10k, test, duplicate_rows, usePath, prs_path, prses[prs_id], use_clustering, use_imputed,correct_for_age_gender, saveName, get_data_args, tailsTest, random_shuffle_prsLoader, use_prsLoader)  ##test can be "t" for t test or "r" for regression))
    for k,v in fundict.copy().items(): ##catch broken PRSes
        if v is None:
            del fundict[k]
    final_res = pd.concat(fundict.values(), axis=1)
    # final_res = final_res.loc[:,~final_res.columns.duplicated()] ##drop the duplicate indices we've accumulated atthis point
    print("Loader matrix finished!")
    return(final_res)
