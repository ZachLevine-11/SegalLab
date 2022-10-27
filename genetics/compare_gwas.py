import os
from os.path import isfile, join
from run_gwas import read_loader_in
from LabData.DataLoaders.DEXALoader import DEXALoader
from scores_work import stack_matrices_and_bonferonni_correct
import pandas as pd
import subprocess
from LabData.DataLoaders.PRSLoader import PRSLoader
import csv

##We need a specific conda installation for this so it can't run on the queue or with shellcommandexecute
def compareGwases(tenk_gwas_name, ukbb_gwas_name, mainpath = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/"):
    subprocess.call("cd /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/ldsc-master && conda init && conda env create --file environment.yml ; source activate ldsc && ldsc-master/ldsc.py --rg "+ mainpath + "ukbb_gwases_munged/" + ukbb_gwas_name +".sumstats.gz," + mainpath + "tenk_gwases_munged/" + tenk_gwas_name + ".sumstats.gz --ref-ld-chr /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/eur_w_ld_chr/ --w-ld-chr " + mainpath + "eur_w_ld_chr/ --out " + mainpath + "all/" + "_tenK_" + tenk_gwas_name.split("batch0.")[-1].split(".glm.linear")[0] + "_UKBB_" + ukbb_gwas_name.split("/")[-1] + " --no-check-alleles; conda deactivate", shell = True)

##not using clumped files for now
def get_tenk_gwas_loc(pheno_name, loader):
    basedir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/"
    if "Lipids" not in pheno_name and "fBin_" not in pheno_name:
        return basedir + "gwas_results/batch0." + pheno_name +".glm.linear"
    elif "Lipids" in pheno_name:
        return basedir + "metab/gwas_results_metab/batch0." + pheno_name +".glm.linear"
    elif "fBin_" in pheno_name:
        return basedir + "microbiome/gwas_results_mb/batch0." + pheno_name +".glm.linear"

def get_ukbb_gwas_loc(prs_name):
   return "/net/mraid08/export/genie/10K/genetics/PRSice/SummaryStatistics/Nealelab/v3/TransformedData/" + prs_name.split("pvalue_")[-1] + ".gwas.imputed_v3.both_sexes.tsv"

def returnGwasPairs(stacked, thresh = 5*1e-10):
    pairs = []
    for pheno, loader in stacked.index:
        tenk_gwas_filename = get_tenk_gwas_loc(pheno, loader)
        entry = stacked.loc[pheno, :].T.dropna()
        entry = entry[entry < thresh]
        for prs in entry.index:
            pairs.append([tenk_gwas_filename, get_ukbb_gwas_loc(prs)])
    return pairs

##From HDL, exported using R
def read_snp_dictionary():
    return pd.read_csv("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/snp_dictionary.csv").drop("Unnamed: 0", axis = 1).set_index("variant").rsid.to_dict()

def prepare_tenk_gwas(tenk_fname, mainpath = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/"):
    N = pd.read_csv(tenk_fname, sep = "\t")["OBS_CT"][0]
    subprocess.call("cd /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/ldsc-master && conda init && conda env create --file environment.yml ; source activate ldsc && " + "ldsc-master/munge_sumstats.py --sumstats " + tenk_fname + " --out " + mainpath + "tenk_gwases_munged/" + tenk_fname.split("/")[-1].split("batch0.")[-1].split(".glm.linear")[0] + " --a2 AX --snp ID --N " + str(N) + "; conda deactivate", shell=True) ##remember that output only goes to the cmd line process running pycharm

def prepare_ukbb_gwas(ukbb_fname, snp_dict, mainpath = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/"):
    temp = pd.read_csv(ukbb_fname, sep = "\t")
    ##both the dict and the original column are in the order of ref, alt
    ##replace the original col with rsids
    temp["variant"]  = pd.Series(list(map(lambda thestr: thestr.replace("[b37]", ":").replace(",", ":"), temp.variant))).apply(snp_dict.get)
    temp.to_csv(mainpath + "ukbb_gwases_with_rsid/" + ukbb_fname.split("/")[-1].split(".")[0] + ".csv", sep = "\t", quotechar = "", quoting = csv.QUOTE_NONE, index = False)
    subprocess.call("cd /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/ldsc-master && conda init && conda env create --file environment.yml ; source activate ldsc && " + 'ldsc-master/munge_sumstats.py --sumstats ' + mainpath + "ukbb_gwases_with_rsid/" + ukbb_fname.split("/")[-1].split(".")[0]  + ".csv" + ' --out ' + mainpath + 'ukbb_gwases_munged/' + ukbb_fname.split("/")[-1].split(".")[0] + ' --snp variant --N 361194 --a1 ref_allele --a2 alt_allele; conda deactivate', shell=True)

def compute_all_cross_corr():
    stacked = stack_matrices_and_bonferonni_correct()
    pairs = returnGwasPairs(stacked)
    snp_dict = read_snp_dictionary()
    DexaPhenos = read_loader_in(DEXALoader).columns
    already_done_tenk = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/", f))]))
    already_done_ukbb = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/", f))]))
    for tenk_fname, ukbb_fname in pairs:
        if tenk_fname.split("batch0.")[-1].split(".glm.linear")[0] not in DexaPhenos: ##Skip DEXA phenotypes for now
            if tenk_fname.split("batch0.")[-1].split(".glm.linear")[0] not in already_done_tenk:
                prepare_tenk_gwas(tenk_fname)
            if ukbb_fname.split("/")[-1].split(".")[0] not in already_done_ukbb:
                prepare_ukbb_gwas(ukbb_fname, snp_dict)
            compareGwases(tenk_fname.split("batch0.")[-1].split(".glm.linear")[0], ukbb_fname.split("/")[-1].split(".")[0])

def find_in_str_list(matchstr, thelist):
    i = 0
    for line in thelist:
        if matchstr in line:
            return i
        i += 1

def read_all_ldsc(dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/all/"):
    ukbb_meaning_dict = PRSLoader().get_data().df_columns_metadata.h2_description.to_dict()
    all_files = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    res = {}
    for file in all_files:
        with open(dir + file, "r") as f:
            contents = f.readlines()
            p_index = find_in_str_list("P: ", contents)
            if p_index is not None: ##Indicating ldsc failed
                p = contents[p_index].split("P: ")[1].split("\n")[0]
                corr = contents[find_in_str_list("Genetic Correlation: ", contents)].split("Genetic Correlation: ")[1].split(" (")[0]
                if p != "nan" and corr != "nan" and p != 'nan (nan) (h2  out of bounds)' and corr != 'nan (nan) (h2  out of bounds)':
                    res[(file.split("tenK_")[-1].split("_UKBB")[0], ukbb_meaning_dict[file.split("UKBB_")[-1].split(".log")[0]])] = {"P": float(p), "Genetic Correlation": float(corr)}
    return pd.DataFrame(res).T

if __name__ == "__main__":
    do_all = False
    if do_all:
        compute_all_cross_corr()
    res = read_all_ldsc()