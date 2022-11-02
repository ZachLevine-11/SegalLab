import numpy as np
from LabQueue.qp import fakeqp as qp
from LabUtils.addloglevels import sethandlers
from GeneticsPipeline.config import gencove_logs_path
import pandas as pd
import os
from preprocess_loader_data import preprocess_loader_data
from statsmodels.stats.multitest import multipletests
from prs_exclude import get_exclusion_map
from manytests import manyTestsbatched

##index_is_10k indicates whether the index of the data matches the 10k version or if specific conversion is required
## see if matches prs
##use_clustering is only defined for Metabolomics Data
def q_loop(loader, index_is_10k = False, test = "t", duplicate_rows = "last", usePath = False, prs_path = "/home/zacharyl/Desktop/intermed_prs.csv",  prs_from_loader = None, use_clustering = False, use_imputed = False, correct_for_age_gender = False, saveName = None, get_data_args = None, tailsTest = "rightLeft", random_shuffle_prsLoader = False, use_prsLoader = True): ##test can be "t" for t test or "r" for regression
    os.chdir("/net/mraid08/export/mb/logs/")
    #os.chdir(gencove_logs_path)
    #sethandlers() ##should only set once, need a switch to not do if in a loop, but queing might fix this problem
    ## create the qp before doing anything with big variables, and delete everything that isn't required before calling qp
    with qp(jobname= "z_in", max_r = 100, max_u = 100, _suppress_handlers_warning= True) as q:
        q.startpermanentrun()
        batch_width = 400
        ##automatically grab tails or continuous data depending on what we want
        if test == "t" or test == "m":
            dataprs = preprocess_loader_data(loader=loader, use_clustering = use_clustering, duplicate_rows = duplicate_rows, keep_prs=False, index_is_10k=index_is_10k, usePath = usePath, prs_path = prs_path,  prs_from_loader = prs_from_loader, use_imputed = use_imputed, use_corrected = correct_for_age_gender, saveName = saveName, get_data_args = get_data_args, random_shuffle_prsLoader = random_shuffle_prsLoader, use_prsLoader = use_prsLoader)
            dataprs_prs_index = dataprs.set_index("PRS_class", drop=True)
        else:
            dataprs = preprocess_loader_data(loader=loader, use_clustering = use_clustering, duplicate_rows = duplicate_rows, keep_prs=True, index_is_10k=index_is_10k, usePath = usePath, prs_path = prs_path,  prs_from_loader = prs_from_loader, use_imputed = use_imputed, use_corrected = correct_for_age_gender, saveName = saveName, get_data_args = get_data_args, random_shuffle_prsLoader = random_shuffle_prsLoader, use_prsLoader = use_prsLoader)
        if dataprs is None:
            return None
        if test != "corrected_regression":
            dataprs_prs_index = dataprs.set_index("prs", drop=True)
        else:
            prs_excludify_map = get_exclusion_map()
            dataprs_prs_index = dataprs ##but the index is still 10K RegistrationCodes
            prs_id = dataprs_prs_index.columns.get_loc("prs")
            ##using the get method lets us do the dictionary lookup but also catch the keynotfound errors
            toExclude = prs_excludify_map.get(prs_from_loader) if prs_excludify_map.get(prs_from_loader) is not None else []
            dataprs_prs_index = dataprs_prs_index.loc[list(map(lambda id: id not in toExclude, dataprs_prs_index.index)), :]
        fundict = {}
        ###We also care about the column names
        varNames = {}
        for i in range((len(dataprs_prs_index.columns) // batch_width) + 1): ##also works with batch_width set to 100 if this breaks
            batch_ids = range(batch_width * i, min(batch_width * (i + 1), len(dataprs_prs_index.columns)))
        ##the first batch is still okay
            if test == "corrected_regression" and len(dataprs_prs_index.columns) > batch_width and i != 0:
                ##we still need the prs column in the batch
                ##prs is the first column of the dataframe so it only gets put in the first batch by default, manually include it if we have more than one batch and we're not in the first batch
                fundict[i] = q.method(manyTestsbatched, (dataprs_prs_index.iloc[:, list(batch_ids) + [prs_id]], test, tailsTest, False))
            else: ##otherwise even for corrected regression since we only have one batch the prs column is already included in batch_ids
                fundict[i] = q.method(manyTestsbatched, (dataprs_prs_index.iloc[:, batch_ids], test, tailsTest, False))
            varNames[i] = pd.Series(dataprs_prs_index.columns[batch_ids])
        fundict = {k: q.waitforresult(v) for k, v in fundict.items()}
    ps = pd.concat(fundict.values(), axis=0) ##in order of columns
    ps_corrected = ps
    varNames = pd.Series(varNames)
    finalres = pd.DataFrame(ps_corrected, index = pd.concat(varNames.values, axis = 0)) ##can also just use the names of the columns in the original data from the loader, but this method lets us skip columns between successive batches
    if prs_from_loader is not None:
        finalres = finalres.rename({0: "pvalue" + "_" + prs_from_loader}, axis = 1)
    else:
        finalres = finalres.rename({0: "pvalue"}, axis = 1)
    return finalres

if __name__ == "__main__":
    testedvalue = q_loop(None, index_is_10k = None, test = "corrected_regression", duplicate_rows = "mean", usePath = False, prs_path = "",  prs_from_loader = "6154_1", use_clustering = False, use_imputed = False, correct_for_age_gender = True, saveName = "DEXALoader.csv", get_data_args = None, tailsTest = "rightLeft", random_shuffle_prsLoader = False, use_prsLoader = True)
