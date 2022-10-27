import numpy as np
import pandas as pd
from LabData.DataLoaders.PRSLoader import PRSLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader ##for tests
from LabData.DataLoaders.SerumMetabolomicsLoader import data_cache_10k_RT_clustering_pearson08
from modified_tom_functions import correct_all_covariates

##general method for loading in data and agregating with prs file
##prs_from_loader corresponds to a column in PRSLoader.get_data().df
##call after qp.startpermanentrun to avoid this data getting sent premtively to the queue for no reason to do nothing
##data arguments
def preprocess_loader_data(loader, index_is_10k, keep_prs = False, usePath = True, prs_path = "/home/zacharyl/Desktop/intermed_prs.csv", duplicate_rows = "last", prs_from_loader = None, use_imputed = True, use_clustering = True, use_corrected = False, saveName = None, get_data_args = None, random_shuffle_prsLoader = False, use_prsLoader = True): ##we can either average values for duplicate rows, or just keep the last value for each
    if use_corrected:
        if saveName is None:
            data = correct_all_covariates(loader = loader, cluster = use_clustering, use_imputed_version = use_imputed, index_is_10k=index_is_10k, get_data_args = get_data_args)
        else:
            data = pd.read_csv("/net/mraid08/export/jasmine/zach/prs_associations/corrected_loaders/" + saveName)
    elif use_clustering and not use_imputed:
        loader_inst = loader()
        data = loader_inst.get_data(precomputed_loader_fname = data_cache_10k_RT_clustering_pearson08).df.copy()
    elif not use_clustering and not use_imputed:
        loader_inst = loader()
        if get_data_args is not None:
            data = loader_inst.get_data(study_ids=['10K']).df.copy()
        else:
            data = loader_inst.get_data(get_data_args, study_ids=['10K']).df.copy()
    elif use_clustering and use_imputed:
        ##use clustering and Tom's imputed data
        ##the imputed data is already the cluster data
        ##this data is in in 10k format, so set the argument to be true
        data = pd.read_csv("/net/mraid08/export/jafar/Tom/imputed_cluster_indcol.csv")
    if not index_is_10k and not use_corrected:
        data['RegistrationCode'] = list(map(lambda serum: '10K_' + serum.split('_')[0], data.index.values))
    elif index_is_10k and not use_imputed and not use_corrected:
        data = data.reset_index(drop=False)  # assume the index matches, but put registration code in as a column, because that's what reset index does by default, when drop = False
    else:
        ##index is 10k and we're either using the imputed data or corrected data (or both), both of which already have the index dropped and included as a column
        pass
    if usePath:
        if keep_prs:
            prs = pd.read_csv(prs_path)[["RegistrationCode", "prs"]]
        else:
            prs = pd.read_csv(prs_path)[["PRS_class", "RegistrationCode"]]
    else:
        if use_prsLoader:
            loaded_prs = PRSLoader().get_data().df.copy()[prs_from_loader]    #load prs with loader, assume that prs_from_loader is not none
        else:
            loaded_prs = pd.read_csv("/net/mraid08/export/jasmine/zach/scores/score_results/SOMAscan/scores_all_raw.csv").set_index("RegistrationCode")[prs_from_loader]    #load prs with loader, assume that prs_from_loader is not none
        if random_shuffle_prsLoader:
            ##shuffle all columns
            ##the line bwlow shuffles every column, don't do it
            #loaded_prs = loaded_prs.apply(np.random.permutation)
            ##we need to shuffle the index when its still the index, making it the column of the data frame doesn't work to shuffle for some reason
            loaded_prs.index = loaded_prs.index[np.random.permutation(len(loaded_prs))]
        loaded_prs = loaded_prs.reset_index() ##do this after we scramble the index. We can't shuffle index if it's a column for some reason.
        ##prs might be duplicated, if it is, keep only one
        loaded_prs = loaded_prs.loc[:,~loaded_prs.columns.duplicated()]
        trimsd = loaded_prs[(loaded_prs[prs_from_loader] < loaded_prs[prs_from_loader].quantile(0.95)) & (loaded_prs[prs_from_loader] > loaded_prs[prs_from_loader].quantile(0.05))][prs_from_loader].std()
        loaded_prs[prs_from_loader] = (loaded_prs[prs_from_loader] - loaded_prs[prs_from_loader].mean()) / trimsd
        ##should normalize the prs and grab other relevant info
        if keep_prs:
            ##the name of the prs right now is that prs - rename just to prs
            loaded_prs = loaded_prs.rename({prs_from_loader: "prs"}, axis = 1)
        else:
            lower_bound = loaded_prs[prs_from_loader].mean() + 2 * trimsd
            upper_bound = loaded_prs[prs_from_loader].mean() - 2 * trimsd
            loaded_prs.loc[:, ["PRS_class"]] = 0  ##0 if neither high nor low PRS
            loaded_prs.loc[loaded_prs[prs_from_loader]>= lower_bound, "PRS_class"] = 2  # 2 for high prs
            loaded_prs.loc[loaded_prs[prs_from_loader] <= upper_bound, "PRS_class"] = 1  # 1 for low prs
            loaded_prs = loaded_prs.drop(prs_from_loader, 1) ##If we're dropping the prs column, the name of it doesn't matter
        prs = loaded_prs
    # The first 10 digits of metabolomics registration codes, prefixed by "10_K" correspnd to 10K registration codes
    dataprs = pd.merge(prs, data, left_on="RegistrationCode", right_on="RegistrationCode", how="inner")
    if duplicate_rows == "mean":
        dataprs = dataprs.groupby('RegistrationCode').mean()
    elif duplicate_rows == "last":
        dataprs = dataprs.drop_duplicates(subset = 'RegistrationCode', keep = "last")
    return dataprs

def test_random_shuffle():
    dataprs = preprocess_loader_data(loader=BodyMeasuresLoader, use_clustering=True, duplicate_rows="last",
                                     keep_prs=False, index_is_10k=True, usePath=False, prs_path="",
                                     prs_from_loader="102_irnt", use_imputed=True,
                                     use_corrected=True, saveName= "body_corrected.csv", get_data_args=None,
                                     random_shuffle_prsLoader=True)
