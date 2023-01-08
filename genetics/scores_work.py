import numpy as np
import pandas as pd
from os.path import isfile, join
from GeneticsPipeline.helpers_genetic import read_status_table
from modified_tom_functions import correct_all_covariates
from loop_generate_prs_matrix import loop_generate_prs_matrix
from q_generate_prs_matrix import q_generate_prs_matrix
from run_gwas import loaders_list, read_loader_in, update_covariates, pre_filter, summarize_gwas
from manual_gwas import read_plink_bins_10K_index
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from LabUtils.addloglevels import sethandlers
# from GeneticsPipeline.config import gencove_logs_path
import os
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns
##Need these imported for the loader_assoc_plot
from LabData.DataLoaders.BloodTestsLoader import BloodTestsLoader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader
from LabData.DataLoaders.SubjectLoader import SubjectLoader
from LabData.DataLoaders.UltrasoundLoader import UltrasoundLoader
from LabData.DataLoaders.DietLoggingLoader import DietLoggingLoader
from LabData.DataLoaders.ABILoader import ABILoader
from LabData.DataLoaders.SerumMetabolomicsLoader import SerumMetabolomicsLoader
from LabData.DataLoaders.ItamarSleepLoader import ItamarSleepLoader
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.LifeStyleLoader import LifeStyleLoader
from LabData.DataLoaders.Medications10KLoader import Medications10KLoader
from LabData.DataLoaders.QuestionnairesLoader import QuestionnairesLoader
from LabData.DataLoaders.HormonalStatusLoader import HormonalStatusLoader
from LabData.DataLoaders.IBSTenkLoader import IBSTenkLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.ChildrenLoader import ChildrenLoader
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.CGMLoader import CGMLoader
from LabData.DataLoaders.PRSLoader import PRSLoader

##to remove circular dependencies, these are hardcoded everywhere
# if you want to change these, you're going to need to change these everywhere in the code that they are used, i.e preprocess_data_loader
##only load the status table once and pass it around to save on memory
status_table = read_status_table()
try:
    status_table = status_table[status_table.passed_qc].copy()
except ValueError:  ##In case the genetics pipeline is running
    status_table = status_table.dropna()
    status_table = status_table[status_table.passed_qc].copy()
raw_qtl_fname = "/net/mraid08/export/jasmine/zach/scores/score_results/SOMAscan/scores_all_raw.csv"
corrected_qtl_fname_base = "/net/mraid08/export/jasmine/zach/prs_associations/corrected_loaders/"
corrected_qtl_savename = "scores_all_corrected.csv"
corrected_qtl_fname = corrected_qtl_fname_base + corrected_qtl_savename
corrected_loader_save_path = "/net/mraid08/export/jasmine/zach/prs_associations/corrected_loaders/"
raw_matrices_save_path_prs = "/net/mraid08/export/jasmine/zach/prs_associations/uncorrected_matrices_prses/"
raw_matrices_save_path_pqtl = "/net/mraid08/export/jasmine/zach/prs_associations/uncorrected_matrices_pqtls/"


def get_all_result_files(SOMAscan=True,
                         cooldir="/net/mraid08/export/jasmine/zach/scores/score_results/"):
    if SOMAscan: allDir = cooldir + "SOMAscan/"
    onlyfiles = [f for f in os.listdir(allDir) if isfile(join(allDir, f)) and f.endswith(".sscore")]
    return onlyfiles


##combine all the raw score files and read them, saving the result
def combine_scores(SOMAscan=True,
                   cooldir="/net/mraid08/export/jasmine/zach/scores/score_results/"):
    i = 0
    all_score_files = get_all_result_files(SOMAscan=SOMAscan, cooldir=cooldir)
    numFiles = len(all_score_files)
    ##merge on Gencove ID (IID), populating the id list from a random results file first
    all_df = pd.read_csv(cooldir + "SOMAscan/" + "GSTP1.4911.49.2_model.txt.sscore", sep="\t").iloc[:, 0]
    all_df.name = "IID"
    for fileName in all_score_files:
        print("Now reading in: " + str(fileName) + ", which is: " + str(i) + "/" + str(numFiles))
        newdf = pd.read_csv(cooldir + "SOMAscan/" + fileName, sep="\t")
        newdf.columns = ["IID", fileName.split('_')[0]]
        all_df = pd.merge(all_df, newdf, left_on="IID", right_on="IID", how="inner")
        i += 1
    all_df['RegistrationCode'] = all_df['IID'].apply(
        status_table.set_index('gencove_id').RegistrationCode.to_dict().get)
    all_df = all_df.set_index("RegistrationCode").drop("IID", axis=1)
    all_df.to_csv(raw_qtl_fname)
    return all_df


###after calling combined_scores once, you can use this function to get at the saved result without having to wait for combine_scores to run again
def read_saved_combined_scores(fname=raw_qtl_fname):
    return pd.read_csv(fname).set_index("RegistrationCode")


def correct_all_loaders(loaders=None, correct_beforehand=False, plink_data=None, most_frequent_val_max_freq=0.95,
                        min_subject_threshold=3000):
    for loader in loaders:
        operative_loader = read_loader_in(loader)
        operative_loader = pre_filter(operative_loader, plink_data=plink_data,
                                      most_frequent_val_max_freq=most_frequent_val_max_freq,
                                      min_subject_threshold=min_subject_threshold)
        justname = str(loader).split(".")[2] + ".csv"
        saveName = corrected_loader_save_path + justname
        if correct_beforehand:
            correct_all_covariates(loader=None, use_precomputed_loader=True,
                                   precomputed_loader=operative_loader).to_csv(saveName)
            print("Wrote corrected: " + saveName)
        else:
            operative_loader.to_csv(saveName)
            print("Wrote uncorrected: " + saveName)


def make_test_all_loaders(loaders=None, loop=False, which="PQTLS", test="corrected_regression"):
    for loader in loaders:
        justname = str(loader).split(".")[2] + ".csv"
        if loop:
            matrix_gen_method = loop_generate_prs_matrix
        else:
            matrix_gen_method = q_generate_prs_matrix
        res_m_loader = matrix_gen_method(test=test, duplicate_rows="last", saveName=justname, tailsTest=None,
                                         random_shuffle_prsLoader=False, use_prsLoader=which != "PQTLS")
        if which == "PQTLS":
            res_m_loader.to_csv(raw_matrices_save_path_pqtl + justname)
            print("Wrote: " + raw_matrices_save_path_pqtl + justname)
        else:
            res_m_loader.to_csv(raw_matrices_save_path_prs + justname)
            print("Wrote: " + raw_matrices_save_path_prs + justname)


def stack_matrices_and_bonferonni_correct(results_dir=raw_matrices_save_path_prs, fillwithNA=True, orderbySig=False,
                                          include_mb_metab=False):
    if include_mb_metab:
        all_results_files = [f for f in os.listdir(results_dir) if isfile(join(results_dir, f))]
    else:
        all_results_files = [f for f in os.listdir(results_dir) if isfile(
            join(results_dir, f)) and "GutMBLoader" not in f and "SerumMetabolomicsLoader" not in f]
    dfs = []
    loader_col = []
    for res_file in all_results_files:
        tempdf = pd.read_csv(results_dir + res_file).set_index("Unnamed: 0")
        loader_col += list(np.repeat(res_file.split(".csv")[0], len(tempdf)))  ##we started with one element
        dfs.append(tempdf)
    res = pd.concat(dfs).fillna(1)
    ##we want to correct for all tests, so unwrap dataframe to be 1d then rewrap after
    ##corrected p values stay in, so this works in original order
    res_corrected = pd.DataFrame(
        multipletests(pvals=res.to_numpy().flatten(), method="bonferroni")[1].reshape(res.shape))
    res_corrected["Phenotype"] = res.index.values
    res_corrected["Loader"] = loader_col
    res_corrected = res_corrected.set_index(["Phenotype", "Loader"])
    res_corrected.columns = res.columns
    if fillwithNA:
        res_corrected = res_corrected.mask(res_corrected > 0.05, np.nan)  ##only store sig assocs
        if orderbySig:  ##very computationally costly
            sigMap = {}
            for pheno, loader in res_corrected.index:
                sigMap[(pheno, loader)] = res_corrected.loc[res_corrected.index.get_level_values(0) == pheno,
                                          :].isna().sum().sum()
            return res_corrected.loc[pd.Series(sigMap).sort_values().to_dict().keys(),
                   :]  ##return rows (phenotypes) sorted in order of most to least significant associations
        else:
            return res_corrected
    else:
        return res_corrected


def loader_assoc_plot(stacked_mat):
    loader_assoc_map = {}
    potential_loaders = list(set(stacked_mat.index.get_level_values(1)))
    for loader in potential_loaders:
        justloader = stacked_mat.loc[stacked_mat.index.get_level_values(1) == loader, :]
        justloader_flat = justloader.to_numpy().flatten()
        loader_assoc_map[loader] = len(justloader_flat[justloader_flat < 0.05])
    res = pd.Series(loader_assoc_map)
    backupindex = res.index
    res.index = list(map(lambda name: name.split("Loader")[0], res.index.values))  ##shorten the names so they fit
    ##map loader names to loader instances to read in and normalize by number of phenotypes
    numEachLoader = list(map(lambda loaderName: len(read_loader_in(eval(loaderName))), backupindex))
    plt.bar(res.index, res.values)
    plt.yscale("log")
    plt.xticks(rotation=45, fontsize=6, ha="right")
    plt.title("Significant Associations by Loader, Raw")
    plt.show()
    plt.bar(res.index, np.divide(res.values, numEachLoader))
    plt.yscale("log")
    plt.xticks(rotation=45, fontsize=6, ha="right")
    plt.title("Significant Associations by Loader, Normalized")
    plt.show()


##Stackmat should have fillNa = False
def make_clustermaps(stackmat):
    s_sig = stackmat.loc[(stackmat < 0.05).any(1), (stackmat < 0.05).any(0)]
    thedict = PRSLoader().get_data().df_columns_metadata
    thedict.index = list(map(lambda thestr: "pvalue_" + thestr, thedict.index))
    thedict = thedict.h2_description.to_dict()
    s_sig = s_sig.rename(dict(zip(list(s_sig.columns), [thedict.get(col) for col in s_sig.columns])), axis=1)
    mapper = list(map(lambda potential: type(potential) == str, s_sig.columns))
    s_sig_only_useful = s_sig.loc[:, mapper]
    s_sig_only_useful.columns = list(map(lambda thestr: thestr[0:20], s_sig_only_useful.columns))
    for theval in s_sig_only_useful.index.get_level_values(1).unique():
        sns.clustermap(-np.log10(s_sig_only_useful.loc[s_sig_only_useful.index.get_level_values(1) == theval, :]),
                       cmap="Blues")
        plt.savefig("/home/zacharyl/Desktop/scores_figures/" + theval + ".png")


##print PRSES clusters from all clusters
def prstoGwas(stackmat, threshold=0.7, do_corr=False, onlyMultiloaders=False):
    df = stackmat.loc[:, stackmat.ne(1).any()]
    df = df[df.ne(1).any(axis=1)]
    df_log = np.log(df).fillna(1)
    if do_corr:
        corr = df.corr('pearson')
        ##handle tiny numbers with integer overflow
        link = linkage(squareform(np.clip(1 - corr, a_min=0, a_max=None)), method='average')
    else:
        link = linkage(df, method="average")
    dn = dendrogram(link, no_plot=True)
    clst = fcluster(link, criterion='distance', t=threshold)
    if do_corr:
        clust_col_identity = pd.Series(index=corr.columns, data=clst).iloc[
            dn['leaves']]  ##for each PRS or phenotype, gives which cluster it's in
    else:
        clust_col_identity = pd.Series(index=df.index, data=clst).iloc[dn['leaves']]
    for identity in range(clust_col_identity.max()):
        if len(clust_col_identity[clust_col_identity.eq(identity)]) > 1:  ##only print clusters with size   > 1
            phenos, loaders = clust_col_identity[clust_col_identity.eq(identity)].index.get_level_values(0), \
                              clust_col_identity[clust_col_identity.eq(identity)].index.get_level_values(1)
            if len(set(loaders)) > 1 or not onlyMultiloaders:
                print("Found multi-loader cluster of, ", phenos, " from the loaders :", set(loaders))
                try:
                    print("Trying GWAS hit intersection")
                    summarize_gwas(phenos, use_clumped=True)
                except TypeError:
                    print("No sig hits for any of these phenotypes")
            else:
                print("Skipping single-loader cluster of, ", phenos)


def report_pheno(pheno, descriptionmap, stacked):
    print("Phenotype: ", pheno, ", Loader: ",
          stacked.index.get_level_values(1).values[stacked.index.get_level_values(0) == pheno][0])
    for prsentry in stacked.loc[pheno, :].T.dropna().index:
        print(descriptionmap[prsentry.split("pvalue_")[1]], ": ", stacked.loc[pheno, :].T.dropna().loc[prsentry].values)
    print("---------------------------------------------------------------")


if __name__ == "__main__":
    sethandlers()
    how = "q"
    ##needed to update the covariates
    ##only load the status table once and pass it around to save on memory
    min_subject_threshold = 2000
    most_frequent_val_max_freq = 0.95
    redo_collect_correct_pqtls = False
    redo_association_tests_prs = False
    redo_association_tests_pqtl = False
    redo_prs_pqtl_associations = False
    correct_beforehand = False  ##keep off to use the model with built in correction for age, gender, and PCS
    redo_loader_saving = False
    if redo_collect_correct_pqtls:
        scores = combine_scores()
        if correct_beforehand:
            scores = correct_all_covariates("", use_precomputed_loader=True, precomputed_loader=scores)
        scores.to_csv(corrected_qtl_fname)
    if redo_loader_saving:
        plink_data_loaded = read_plink_bins_10K_index()
        correct_all_loaders(loaders=loaders_list, correct_beforehand=correct_beforehand, plink_data=plink_data_loaded,
                            min_subject_threshold=min_subject_threshold,
                            most_frequent_val_max_freq=most_frequent_val_max_freq)
    ##then, treat the PQTLS as the PRSES and test with all the dataloaders
    if redo_association_tests_prs:
        plink_data_loaded = None  ##to avoid pickling this when we send things to the queue
        update_covariates(
            status_table=status_table)  ##We use the 10 PCS from the GWAS and age and gender as covariates in the model, so let's keep them current
        make_test_all_loaders(loaders=loaders_list, which="PRSES", loop=how == "loop")
    if redo_association_tests_pqtl:
        make_test_all_loaders(loaders=loaders_list, which="PQTLS", loop=how == "loop")
    if redo_prs_pqtl_associations:
        ##last, treat the PQTLS as dataloaders and tests with PRSES
        PQTLS_PRS_matrix = q_generate_prs_matrix(test="corrected_regression", duplicate_rows="last",
                                                 saveName=corrected_qtl_savename, tailsTest=None,
                                                 random_shuffle_prsLoader=False, use_prsLoader=True)
        PQTLS_PRS_matrix.to_csv("/net/mraid08/export/jasmine/zach/prs_associations/prs_pqtl_matrix.csv")
