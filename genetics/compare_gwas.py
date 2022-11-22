import os
from os.path import isfile, join
from run_gwas import read_loader_in
from LabUtils.addloglevels import sethandlers
import numpy as np
import pandas as pd
import subprocess
from LabData.DataLoaders.PRSLoader import PRSLoader
from scores_work import stack_matrices_and_bonferonni_correct
import csv
from scipy.stats.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from LabQueue.qp import qp
from GeneticsPipeline.config import qp_running_dir

##We need a specific conda installation for this so it can't run on the queue or with shellcommandexecute
def compareGwases(tenk_gwas_name, ukbb_gwas_name, mainpath = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/"):
    rg_arg = mainpath + "ukbb_gwases_munged/" + ukbb_gwas_name +".sumstats.gz," + mainpath + "tenk_gwases_munged/" + tenk_gwas_name + ".sumstats.gz"
    second_arg = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/eur_w_ld_chr/"
    third_arg =  "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/eur_w_ld_chr/"
    fourth_arg = mainpath + "all/" + "_tenK_" + tenk_gwas_name.split("batch0.")[-1].split(".glm.linear")[0] + "_UKBB_" + ukbb_gwas_name.split("/")[-1]
    subprocess.call(["~/PycharmProjects/genetics/do_ldsc_cmd.sh" + " " + rg_arg + " " + second_arg + " " + third_arg + " " + fourth_arg], shell=True)
    if is_ldsc_broken(*parse_single_ldsc_file(fourth_arg + ".log")):
        return -1
    else:
        return 0


##not using clumped files for now
def get_tenk_gwas_loc(pheno_name):
    basedir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/"
    if "Lipids" not in pheno_name and "fBin_" not in pheno_name:
        return basedir + "gwas_results/batch0." + pheno_name +".glm.linear"
    elif "Lipids" in pheno_name:
        return basedir + "metab/gwas_results_metab/batch0." + pheno_name +".glm.linear"
    elif "fBin_" in pheno_name:
        return basedir + "microbiome/gwas_results_mb/batch0." + pheno_name +".glm.linear"

def get_ukbb_gwas_loc(prs_name):
   return "/net/mraid08/export/genie/10K/genetics/PRSice/SummaryStatistics/Nealelab/v3/TransformedData/" + prs_name.split("pvalue_")[-1] + ".gwas.imputed_v3.both_sexes.tsv"

##From HDL, exported using R
def read_snp_dictionary():
    return pd.read_csv("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/snp_dictionary.csv").drop("Unnamed: 0", axis = 1).set_index("variant").rsid.to_dict()

def prepare_tenk_gwas(tenk_fname, mainpath = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/"):
    N = pd.read_csv(tenk_fname, sep = "\t")["OBS_CT"][0]
    first_arg = tenk_fname
    second_arg =  mainpath + "tenk_gwases_munged/" + tenk_fname.split("/")[-1].split("batch0.")[-1].split(".glm.linear")[0]
    third_arg = str(N)
    subprocess.call(["~/PycharmProjects/genetics/prepare_tenk_gwas.sh" + " " + first_arg + " " + second_arg + " " + third_arg], shell=True)
    return 0

def prepare_ukbb_gwas(ukbb_fname, mainpath = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/"):
    try:
        temp = pd.read_csv(ukbb_fname, sep = "\t")
    except FileNotFoundError:
        return -1
    snp_dict = read_snp_dictionary()
    ##both the dict and the original column are in the order of ref, alt
    ##replace the original col with rsids
    temp["variant"]  = pd.Series(list(map(lambda thestr: thestr.replace("[b37]", ":").replace(",", ":"), temp.variant))).apply(snp_dict.get)
    temp.to_csv(mainpath + "ukbb_gwases_with_rsid/" + ukbb_fname.split("/")[-1].split(".")[0] + ".csv", sep = "\t", quotechar = "", quoting = csv.QUOTE_NONE, index = False)
    first_arg = mainpath + "ukbb_gwases_with_rsid/" + ukbb_fname.split("/")[-1].split(".")[0]  + ".csv"
    second_arg = mainpath + 'ukbb_gwases_munged/' + ukbb_fname.split("/")[-1].split(".")[0]
    subprocess.call(["~/PycharmProjects/genetics/prepare_ukbb_gwas.sh" + " " + first_arg + " " + second_arg], shell=True)
    return 0

def ldsc_pipeline(tenk_fnames, ukbb_fnames):
    broken_tenk_phenos = list(pd.read_csv("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos.csv").pheno.values)
    ##make sure we only count phenotypes with actual summary statistics and not logs with no sumstats
    already_munged_tenk = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/", f)) and f.endswith(".sumstats.gz")]))
    already_munged_ukbb = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/", f)) and f.endswith(".sumstats.gz")]))
    already_done_ldsc = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "all/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "all/", f))]))
    already_done_pairs = [[x.split("_tenK_")[1].split("_UKBB_")[0], x.split("_tenK_")[1].split("_UKBB_")[1].split(".log")[0]] for x in already_done_ldsc]
    for ukbb_fname in ukbb_fnames:
        for tenk_fname in tenk_fnames:
            if tenk_fname not in broken_tenk_phenos and [tenk_fname, ukbb_fname] not in already_done_pairs:
                ukbb_munge_res = 0 ##default value indicating no errors
                print("Starting pair: ", tenk_fname, " and ", ukbb_fname)
                if tenk_fname.split("batch0.")[-1].split(".glm.linear")[0] not in already_munged_tenk:
                    print("Starting munging 10K GWAS: ", tenk_fname.split("batch0.")[-1].split(".glm.linear")[0])
                    prepare_tenk_gwas(tenk_fname)
                    print("Done munging 10K GWAS: ", tenk_fname.split("/")[-1].split(".")[0])
                if ukbb_fname.split("/")[-1].split(".")[0] not in already_munged_ukbb:
                    print("Starting munging UKBB GWAS: ", ukbb_fname.split("batch0.")[-1].split(".glm.linear")[0])
                    ukbb_munge_res = prepare_ukbb_gwas(ukbb_fname)
                    print("Done munging UKBB GWAS: ", ukbb_fname.split("/")[-1].split(".")[0])
                if ukbb_munge_res != -1:
                    print("Starting ldsc between the two")
                    ##If heritability of the 10K trait was found to be invalid in another ldsc run (with a different phenotype), skip reruns of it
                    ldsc_res = compareGwases(tenk_fname.split("batch0.")[-1].split(".glm.linear")[0], ukbb_fname.split("/")[-1].split(".")[0])
                    if ldsc_res == -1: ##indicating a broken run
                        with open("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos.csv", "a+") as f:
                            f.write(tenk_fname)
                            f.write(",")
                            f.write("\n")
                        print("Broken phenotype from 10K is ", tenk_fname.split("batch0.")[-1].split(".glm.linear")[0], " added to exclusion list")
                    print("Finished ldsc")

def compute_all_cross_corr(batch_width = 100):
    broken_tenk_phenos = list(pd.read_csv("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos.csv").pheno.values)
    os.chdir(qp_running_dir)
    res = {}
    with qp(jobname="ldsc", delay_batch = 30) as q:
        q.startpermanentrun()
        stacked = stack_matrices_and_bonferonni_correct(fillwithNA=False)
        all_tenk_fnames = list(map(get_tenk_gwas_loc, stacked.index.get_level_values(0)))
        all_tenk_fnames = [x for x in all_tenk_fnames if x not in broken_tenk_phenos and x != "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/batch0.prs.glm.linear"]
        all_ukbb_fnames = list(map(get_ukbb_gwas_loc, stacked.columns))
        tenk_fnames_batched = np.array_split(all_tenk_fnames, batch_width)
        ukbb_fnames_batched = np.array_split(all_ukbb_fnames, batch_width)
        i = 0
        for tenk_batch in tenk_fnames_batched:
            for ukbb_batch in ukbb_fnames_batched:
                res[i]  = q.method(ldsc_pipeline, (tenk_batch, ukbb_batch))
                i += 1
        res = {k: q.waitforresult(v) for k, v in res.items()}

def find_in_str_list(matchstr, thelist):
    i = 0
    for line in thelist:
        if matchstr in line:
            return i
        i += 1

def parse_single_ldsc_file(file, dir = ""):
    p, corr = "nan", "nan"
    with open(dir + file, "r") as f:
        contents = f.readlines()
        p_index = find_in_str_list("P: ", contents)
        if p_index is not None:  ##Indicating ldsc failed
            p = contents[p_index].split("P: ")[1].split("\n")[0]
            corr = contents[find_in_str_list("Genetic Correlation: ", contents)].split("Genetic Correlation: ")[1].split(" (")[0]
    return p, corr

def is_ldsc_broken(p, corr):
    if p != "nan" and corr != "nan" and p != 'nan (nan) (h2  out of bounds)' and corr != 'nan (nan) (h2  out of bounds)':
        return False
    else:
        return True

def read_all_ldsc(dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/all/"):
    ukbb_meaning_dict = PRSLoader().get_data().df_columns_metadata.h2_description.to_dict()
    all_files = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    res = {}
    for file in all_files:
            p, corr = parse_single_ldsc_file(file, dir = dir)
            if not is_ldsc_broken(p, corr):
                res[(file.split("tenK_")[-1].split("_UKBB")[0], ukbb_meaning_dict[file.split("UKBB_")[-1].split(".log")[0]])] = {"P": float(p), "Genetic Correlation": float(corr)}
    res = pd.DataFrame(res).T
    res["P"] = multipletests(pvals = res["P"], method = "bonferroni")[1]
    return res

def gen_feature_corr(stackmat, genmat):
    inversedict = PRSLoader().get_data().df_columns_metadata.reset_index().set_index("h2_description").phenotype_code.to_dict()
    for k,v in inversedict.items():
        inversedict[k] = "pvalue_" + inversedict[k]
    genmat_dict = genmat.to_dict()["P"]
    stackmat_dict = {}
    for k,v in genmat_dict.items():
        stackmat_dict[k] = float(stackmat.loc[k[0]].T.loc[inversedict[k[1]]].values)
    combined = pd.concat([pd.Series(stackmat_dict), pd.Series(genmat_dict)], 1)
    combined.columns = ["feature_space", "genetic_space"]
    combined = combined.dropna()
    return combined, pearsonr(combined.iloc[:, 1], combined.iloc[:, 0])

if __name__ == "__main__":
    sethandlers()
    do_all = True
    if do_all:
        compute_all_cross_corr()
