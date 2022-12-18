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
from scipy.stats.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from LabQueue.qp import qp
from GeneticsPipeline.config import qp_running_dir
from LabData.DataLoaders.DEXALoader import DEXALoader
from LabData.DataLoaders.CGMLoader import CGMLoader

##We need a specific conda installation for this so it can't run on the queue or with shellcommandexecute
def compareGwases(tenk_gwas_name, ukbb_gwas_name, mainpath = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/"):
    rg_arg = mainpath + "ukbb_gwases_munged/" + ukbb_gwas_name +".sumstats.gz," + mainpath + "tenk_gwases_munged/" + tenk_gwas_name + ".sumstats.gz"
    second_arg = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/eur_w_ld_chr/"
    third_arg =  "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/eur_w_ld_chr/"
    fourth_arg = mainpath + "all/" + "_tenK_" + tenk_gwas_name.split("batch0.")[-1].split(".glm.linear")[0] + "_UKBB_" + ukbb_gwas_name.split("/")[-1]
    try:
        subprocess.call(["~/PycharmProjects/genetics/do_ldsc_cmd.sh" + " " + rg_arg + " " + second_arg + " " + third_arg + " " + fourth_arg], shell=True)
    except Exception:
        return -1
    if is_ldsc_broken(*parse_single_ldsc_file(fourth_arg + ".log")):
        return -1
    else:
        return 0

def get_ukbb_gwas_loc(prs_name):
    prs_name = str(prs_name)
    return "/net/mraid08/export/genie/10K/genetics/PRSice/SummaryStatistics/Nealelab/v3/TransformedData/" + prs_name.split("pvalue_")[-1] + ".gwas.imputed_v3.both_sexes.tsv"

##From HDL, exported using R
def read_snp_dictionary():
    return pd.read_csv("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/snp_dictionary.csv").drop("Unnamed: 0", axis = 1).set_index("variant").rsid.to_dict()

def prepare_tenk_gwas(tenk_fname, mainpath = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/"):
    N = pd.read_csv(tenk_fname, sep = "\t")["OBS_CT"][0]
    first_arg = tenk_fname
    second_arg =  mainpath + "tenk_gwases_munged/" + tenk_fname.split("/")[-1].split("batch0.")[-1].split(".glm.linear")[0]
    third_arg = str(N)
    try:
        subprocess.call(["~/PycharmProjects/genetics/prepare_tenk_gwas.sh" + " " + first_arg + " " + second_arg + " " + third_arg], shell=True)
    except Exception:
        return -1
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
    try:
        subprocess.call(["~/PycharmProjects/genetics/prepare_ukbb_gwas.sh" + " " + first_arg + " " + second_arg], shell=True)
    except Exception:
        return -1
    return 0

def munge_tenk_batched(tenk_batch):
    tenk_munge_results = {}
    broken_tenk_phenos = [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos/")if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos/", f))]
    already_munged_tenk = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/", f))]))
    for tenk_fname in tenk_batch:
        tenk_munge_res = 0 ##So that if the file already was munged and we don't munge it again, it indicates success
        if tenk_fname not in broken_tenk_phenos and ".glm.linear" in tenk_fname: ##Catch plink log files
            if tenk_fname.split("batch0.")[-1].split(".glm.linear")[0] not in already_munged_tenk:
                print("Starting munging 10K GWAS: ", tenk_fname.split("batch0.")[-1].split(".glm.linear")[0])
                tenk_munge_res  = prepare_tenk_gwas(tenk_fname)
                print("Done munging 10K GWAS: ", tenk_fname.split("/")[-1].split(".")[0])
        tenk_munge_results[tenk_fname] = tenk_munge_res
    return tenk_munge_results

def munge_ukbb_batched(ukbb_batch):
    ukbb_munge_results = {}
    ##Count log files too because sometimes we only have log files (i.e if a run failed)
    already_munged_ukbb = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/", f))]))
    for ukbb_fname in ukbb_batch:
        ukbb_munge_res = 0 ##So that if the file already was munged and we don't munge it again, it indicates success
        if ukbb_fname.split("/")[-1].split(".")[0] not in already_munged_ukbb:
            print("Starting munging UKBB GWAS: ", ukbb_fname.split("batch0.")[-1].split(".glm.linear")[0])
            ukbb_munge_res = prepare_ukbb_gwas(ukbb_fname)
            print("Done munging UKBB GWAS: ", ukbb_fname.split("/")[-1].split(".")[0])
        ukbb_munge_results[ukbb_fname] = ukbb_munge_res
    return ukbb_munge_results

def do_log(tenk_fname):
    with open("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos/" +
              tenk_fname.split("batch0.")[-1].split(".glm.linear")[0], "w") as f:
        f.write("")
    print("Broken phenotype from 10K is ", tenk_fname.split("batch0.")[-1].split(".glm.linear")[0],
          " added to exclusion list")

def compare_gwases_batched(tenk_fnames, ukbb_fnames, exclude_broken_tenk_phenos = False):
    if exclude_broken_tenk_phenos:
        broken_tenk_phenos  = [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos/", f))]
    else:
        broken_tenk_phenos = []
    already_done_ldsc = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "all/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "all/", f))]))
    already_done_pairs = [[x.split("_tenK_")[1].split("_UKBB_")[0], x.split("_tenK_")[1].split("_UKBB_")[1].split(".log")[0]] for x in already_done_ldsc if "_tenK_" in x and "_UKBB_" in x]
    for ukbb_fname in ukbb_fnames:
        ##Maps fname:munge res for this batch
        for tenk_fname in tenk_fnames:
            try:
                if tenk_fname not in broken_tenk_phenos and [tenk_fname, ukbb_fname] not in already_done_pairs:
                    print("Starting ldsc between the two")
                    ##If heritability of the 10K trait was found to be invalid in another ldsc run (with a different phenotype), skip reruns of it
                    ldsc_res = compareGwases(tenk_fname.split("batch0.")[-1].split(".glm.linear")[0],ukbb_fname.split("/")[-1].split(".")[0])
                    if ldsc_res == -1:  ##indicating a broken run
                        do_log(tenk_fname)
            except Exception:
                do_log(tenk_fname)

def unstack_matrix(ldsc_mat):
    resg = ldsc_mat["Genetic Correlation"]
    resg_2d = resg.loc[~resg.index.duplicated(), :].unstack(level=1)
    return resg_2d

##Truncated phenotype names will always be re munged because the phenotype name doesn't match the munged phenotype name.
##To avoid munging anything do already_munged_all = True
##There's no way around this right now, but it's only three phenotypes
def compute_all_cross_corr(already_munged_all = True, exclude_broken_tenk_phenos = False, batch_width = 10, containing_dirs = ["/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/", "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/metab/gwas_results_metab/", "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/microbiome/gwas_results_mb/"]):
    if exclude_broken_tenk_phenos:
        broken_tenk_phenos  = [f for f in os.listdir("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos/") if isfile(join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/broken_tenk_phenos/", f))]
    else:
        broken_tenk_phenos = []
    res_munging = {}
    res_ldsc = {}
    with qp(jobname="ldsc", delay_batch = 30) as q:
        q.startpermanentrun()
        if not already_munged_all:
            all_tenk_fnames = []
            for containing_dir in containing_dirs:
                all_tenk_fnames += [containing_dir + f for f in os.listdir(containing_dir) if isfile(join(containing_dir, f))]
            all_tenk_fnames = [x for x in all_tenk_fnames if x not in broken_tenk_phenos and x != "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/batch0.prs.glm.linear" and "clumpheader" not in x]
            all_ukbb_fnames = list(map(get_ukbb_gwas_loc, PRSLoader().get_data().df.columns))
            already_munged_ukbb = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir(
                "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/") if isfile(
                join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/", f))]))
            already_munged_tenk = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir(
                "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/") if isfile(
                join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/", f))]))
            all_tenk_fnames_not_munged = []
            for x in all_tenk_fnames:
                if ".glm.linear" in x:
                    if x.split("batch0.")[-1].split(".glm.linear")[0] not in already_munged_tenk:
                        all_tenk_fnames_not_munged.append(x)
                elif ".glm.logistic" in x:
                    if x.split("batch0.")[-1].split(".glm.logistic")[0] not in already_munged_tenk:
                        all_tenk_fnames_not_munged.append(x)
            if len(all_tenk_fnames_not_munged) == 0:
                print("All TenK Gwases have been munged already")
            else:
                all_tenk_fnames_not_munged_batched = np.array_split(all_tenk_fnames_not_munged, max(len(all_tenk_fnames_not_munged)//batch_width, 1))
                print("Dispatching remaining " , len(all_tenk_fnames_not_munged), " TenK munging files")
                for tenk_batch in all_tenk_fnames_not_munged_batched:
                    ##Keep the arguments as tuples so that the qp internal check (is empty) works, instead of doing the array comparison and it not working
                    res_munging[tuple(tenk_batch)]  = q.method(munge_tenk_batched, (tenk_batch,))
            all_ukbb_fnames_not_munged = [x for x in all_ukbb_fnames if x.split("/")[-1].split(".")[0] not in already_munged_ukbb]
            if len(all_ukbb_fnames_not_munged) == 0:
                print("All UKBB Gwases have been munged already")
            else:
                all_ukbb_fnames_not_munged_batched = np.array_split(all_ukbb_fnames_not_munged, max(len(all_ukbb_fnames_not_munged) // batch_width, 1))
                print("Dispatching remaining " , len(all_ukbb_fnames_not_munged), " UKBB munging files")
                for ukbb_batch in all_ukbb_fnames_not_munged_batched:
                    res_munging[tuple(ukbb_batch)]  = q.method(munge_ukbb_batched, (ukbb_batch,))
            res_munging = {k: q.waitforresult(v) for k, v in res_munging.items()}
            status_table = pd.Series(res_munging)
            status_table.to_csv("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/munging_status_table.csv")
        else:
            print("Not munging anything as per selection")
        ##Now each batch of tenk_phenos and ukbb_phenos lives in res_munging, which we do in the same dict to wait for both at the same time
        ##To grab the result of a single UKBB munge, grab the results corresponding to the batch and then the results corresponding to the filename
        ##While they were technically munged and we don't want to redo them, so we count them above, now ignore gwases that don't
        ##Only give ldscore GWASes that have been munged with actual corresponding .sumstats.gz files where we want them
        sumstats_exists_ukbb = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir(
            "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/") if isfile(
            join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "ukbb_gwases_munged/",
                 f)) and f.endswith(".sumstats.gz")]))
        sumstats_exists_tenk = list(map(lambda thestr: thestr.split(".")[0], [f for f in os.listdir(
            "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/") if isfile(
            join("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/" + "tenk_gwases_munged/",
                 f)) and f.endswith(".sumstats.gz")]))
        sumstats_exists_ukbb_batched = np.array_split(sumstats_exists_ukbb, len(sumstats_exists_ukbb)//batch_width)
        sumstats_exists_tenk_batched = np.array_split(sumstats_exists_tenk, len(sumstats_exists_tenk)//(4*batch_width))
        ##Only operate on GWASES with actual sumstats.gz files at this point, when we've considered all traits
        i = 0
        print("Starting ldscore")
        for ukbb_batch in sumstats_exists_ukbb_batched:
            for tenk_batch in sumstats_exists_tenk_batched:
                res_ldsc[i] = q.method(compare_gwases_batched, (tenk_batch, ukbb_batch, exclude_broken_tenk_phenos))
                i += 1
        for k, v in res_ldsc.items():
            try:
                res_ldsc[k] =  q.waitforresult(v)
            except Exception:
                print("Job fell off: ", k)

def find_in_str_list(matchstr, thelist):
    i = 0
    for line in thelist:
        if matchstr in line:
            return i
        i += 1

def parse_single_ldsc_file(file, dir = ""):
    p, corr, her = "nan", "nan", "nan"
    with open(dir + file, "r") as f:
        contents = f.readlines()
        p_index = find_in_str_list("P: ", contents)
        if p_index is not None:  ##Indicating ldsc failed
            p = contents[p_index].split("P: ")[1].split("\n")[0]
            corr = contents[find_in_str_list("Genetic Correlation: ", contents)].split("Genetic Correlation: ")[1].split(" (")[0]
            ##The 10K trait is always phenotype 2/2 in the ldsc report
            her = contents[find_in_str_list("Heritability of phenotype 2/2", contents)+2].split("Total Observed scale h2: ")[1].split(" (")[0]
    return p, corr, her

def is_ldsc_broken(p, corr, her):
    if p != "nan" and corr != "nan" and p != 'nan (nan) (h2  out of bounds)' and corr != 'nan (nan) (h2  out of bounds)':
        return False
    else:
        return True

def whichLoader(phenoName, dexacols, cgmcols):
    if phenoName in cgmcols:
        return "Insulin"
    elif phenoName in dexacols:
        return "DEXA"
    else:
        return phenoName

def read_all_ldsc(dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/all/"):
    ukbb_meaning_dict = PRSLoader().get_data().df_columns_metadata.h2_description.to_dict()
    all_files = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    res = {}
    for file in all_files:
            p, corr, her = parse_single_ldsc_file(file, dir = dir)
            if not is_ldsc_broken(p, corr, her):
                res[(file.split("tenK_")[-1].split("_UKBB")[0], ukbb_meaning_dict[file.split("UKBB_")[-1].split(".log")[0]])] = {"P": float(p), "Genetic Correlation": float(corr), "10K Trait Heritability": float(her)}
            else:
                res[(file.split("tenK_")[-1].split("_UKBB")[0], ukbb_meaning_dict[file.split("UKBB_")[-1].split(".log")[0]])] = {"P": None, "Genetic Correlation": None, "10K Trait Heritability": None}
    res = pd.DataFrame(res).T
    res.loc[~res.P.isna(), "P"] = multipletests(pvals = res.loc[~res.P.isna(), "P"], method = "bonferroni")[1]
    return res

def gen_feature_corr(stackmat, genmat):
    inversedict = PRSLoader().get_data().df_columns_metadata.reset_index().set_index("h2_description").phenotype_code.to_dict()
    for k,v in inversedict.items():
        inversedict[k] = "pvalue_" + inversedict[k]
    genmat_dict = genmat.to_dict()["P"]
    stackmat_dict = {}
    for k,v in genmat_dict.items():
        try: ##In case the PRS meaning doens't exist, like if its not in the columns metadata
            stackmat_dict[k] = float(stackmat.loc[k[0]].T.loc[inversedict[k[1]]].values)
        except KeyError:
            pass
    combined = pd.concat([pd.Series(stackmat_dict), pd.Series(genmat_dict)], 1)
    combined.columns = ["feature_space", "genetic_space"]
    combined = combined.dropna()
    return combined, spearmanr(combined.iloc[:, 1], combined.iloc[:, 0])

if __name__ == "__main__":
    sethandlers()
    os.chdir(qp_running_dir)
    do_all = True
    if do_all:
        compute_all_cross_corr(containing_dirs=["/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/"])
