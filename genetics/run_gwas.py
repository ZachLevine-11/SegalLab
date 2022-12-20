import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin, write_plink1_bin
import os
import ast
from os.path import isfile, join, getmtime
import re
import string
import random
from LabQueue.qp import qp
import datetime
from LabUtils.Scripts.shell_commands_execute import ShellCommandsExecute
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
from LabData.DataLoaders.RetinaScanLoader import RetinaScanLoader
from LabData.DataLoaders.HormonalStatusLoader import HormonalStatusLoader
from LabData.DataLoaders.IBSTenkLoader import IBSTenkLoader
from LabData.DataLoaders.GutMBLoader import GutMBLoader
from LabData.DataLoaders.ChildrenLoader import ChildrenLoader
from LabData.DataLoaders.CGMLoader import CGMLoader
from GeneticsPipeline.helpers_genetic import read_status_table, run_plink2, required_memory_gb
from LabUtils.Scripts.shell_commands_execute import ShellCommandsExecute
from LabData.DataUtils.DataProcessing import NormDistCapping
from LabUtils.addloglevels import sethandlers
from GeneticsPipeline.config import plink19_bin, plink2_bin, qp_running_dir
from pandas_plink import write_plink1_bin
from functools import reduce
import matplotlib.pyplot as plt
from itertools import chain, combinations
import sys

##Not using Questionnaires (broken according to Nastya), DietLogging (not useful)
##Gut MB and Metab get done separately
#loaders_list = [CGMLoader, UltrasoundLoader, ABILoader, ItamarSleepLoader, HormonalStatusLoader, DEXALoader, RetinaScanLoader]

loaders_list = [SerumMetabolomicsLoader, GutMBLoader]

sleepPhenos =  ["batch0.AHI.glm.linear", "batch0.MeanSatValue.glm.linear", "batch0.TotalNumberOfApneas.glm.linear"]
liverPhenos = ["batch0.Average_SSP_Plus.glm.linear", "batch0.Average_ATT_Plus.glm.linear"]
dexaPhenos = ["batch0.body_spine_bmd.glm.linear"]

def run_plink1(plink_cmd, jobname, mem_required_gb, threads=32, queue=qp):
    os.chdir(qp_running_dir)
    with queue(jobname=jobname, _mem_def=f'{mem_required_gb}G',_trds_def=threads, _tryrerun=True) as q:
        q.startpermanentrun()
        res = q.method(lambda: ShellCommandsExecute().run(cmd=plink_cmd, cmd_name=jobname))
        q.waitforresult(res)

short_name_dict_fname = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/shortened_name_table_current.csv"

##returns original plink bins in their unadaltered form
def read_original_plink_bins():
    original_bin = read_plink1_bin(bed="/net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc.bed",
                               bim="/net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc.bim",
                               fam="/net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc.fam")
    return(original_bin)

def extract_height(dir="/net/mraid08/export/jasmine/zach/height_gwas/"):
    df = BodyMeasuresLoader().get_data(study_ids=["10K"]).df.copy()
    height_data = df.reset_index().loc[:, ["height", "RegistrationCode"]]
    height_data.columns = ["height", "IID"]
    height_data["FID"] = "0"  ##to allign with binary bed file
    ##order matters for plink
    height_data = height_data.loc[:, ["FID", "IID", "height"]]
    ##the binary files uses gencove ids, not RegistrationCodes. Make the switch
    height_data['IID'] = height_data['IID'].apply(status_table.set_index('RegistrationCode').gencove_id.to_dict().get)
    height_data.dropna(inplace=True)
    ##round floating point values so that plink will accept them
    height_data.height = height_data.height.astype("int")
    height_data = height_data[~height_data.IID.duplicated(keep='first')]
    height_data.to_csv(dir + "height_pheno.txt", sep="\t", index=False, header=True)

def make_height_gwas_command():
    cmd = "/net/mraid08/export/genie/Bin/plink2a/plink2 --bfile /net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc --pheno /net/mraid08/export/genie/10K/genetics/TestsCache/zach/height_gwas/height_pheno.txt --pheno-name height --linear cols=+ax --allow-no-sex --out /net/mraid08/export/genie/10K/genetics/TestsCache/zach/height_gwas/height_pheno.txt --covar /net/mraid08/export/genie/10K/genetics/TestsCache/zach/height_gwas/covariates_with_age_gender.txt --covar-col-nums 3-14 --covar-variance-standardize"

def tryConversion(col):
    try: ##don't do setattr because sometimes we have column names like "mean" which fight default methods in numpy
        ##we also still need the data to be a data frame though and square bracket indexing turns it into a Series, so make sure its a DataFrame after
        col = col.astype("float")
    except ValueError:
        pass
    return col

def write_pheno(col, dir = None, use_short_names = True):
    pheno_name = col.name
    pheno_data = col.reset_index().loc[:, [pheno_name, "RegistrationCode"]]
    pheno_data.columns = [pheno_name, "IID"]
    pheno_data["FID"] = "0"  ##to allign with binary bed file
    ##order matters for plink
    pheno_data = pheno_data.loc[:, ["FID", "IID", pheno_name]]
    ##the binary files uses gencove ids, not RegistrationCodes. Make the switch
    pheno_data['IID'] = pheno_data['IID'].apply(status_table.set_index('RegistrationCode').gencove_id.to_dict().get)
    pheno_data.dropna(inplace=True)
    ##round floating point values so that plink will accept them
    pheno_data[pheno_name] = tryConversion(pheno_data[pheno_name])
    if not use_short_names:
        pheno_shortname = pheno_name
    else:
        letters = string.ascii_lowercase
        random_name = ("".join(random.choice(letters) for i in range(100)))
        pheno_shortname = random_name
        ##the "|" in the column name breaks shellcommandexecute because it thinks we are using a pipe.
        ##replace the entire name with a random string and keep track of that string.
        shortened_name_table[pheno_name] = pheno_shortname
        pheno_data.columns = [*pheno_data.columns[0:2], pheno_shortname]
    if dir is not None:
        pheno_data.to_csv(dir + pheno_shortname + "_pheno.txt", sep="\t", index=False, header=True)
        print("Wrote: " + pheno_name)
    else:
        return pheno_data[pheno_shortname]

##for ABILoader and ItamarSleepLoader
def get_elapsed_seconds(time_str):
    if pd.isna(time_str): return np.nan
    h,m,s = time_str.split(":")
    return float(h)*3600 + float(m) * 60 + float(s)

##for ItamarSleepLoader
def PhysicalTime_seconds(date_time_str):
    if pd.isna(date_time_str): return np.nan
    elapsedTime = date_time_str.split(" ")[-1]
    return get_elapsed_seconds(elapsedTime)

def fix_birth_weight(comma_int):
    if isinstance(comma_int, str):
        try:
            ans = float(comma_int.replace(",", ""))
        except ValueError:
            ans =  np.nan
        return ans

def encode_dummy_variables_children(childrendf):
    def get_broken_dummy_names_children(col_list):
        col_list = np.array(col_list)
        return col_list[list(map(lambda col_name: col_name != col_name.split("_[")[0], col_list))]
    prefixes = {}
    children_dummy = pd.get_dummies(childrendf)
    broken_dummies = get_broken_dummy_names_children(children_dummy.columns)
    new_columns = []
    for broken_column in broken_dummies:
        column_meaning_list = ast.literal_eval("[" + broken_column.split("_[")[-1])
        for col in column_meaning_list:
            prefixes[col] = broken_column.split("_[")[0] + "_["
            new_columns.append(col)
    new_columns = list(set(new_columns))
    new_columns_properly_prefixed = list(map(lambda colName: prefixes[colName]  + colName + "]", new_columns))
    for i in range(len(new_columns)):
        parent_cols = broken_dummies[list(map(lambda parent_col: parent_col.replace(new_columns[i], "") != parent_col, broken_dummies))]
        harmonized_disease_column = children_dummy.loc[:, parent_cols].any(axis = 1)
        children_dummy[new_columns_properly_prefixed[i]] = harmonized_disease_column ##overwrite the existing columns because we already accounted for them in the parent columns
    res = children_dummy.drop(broken_dummies, axis = 1)
    return res

##norm_dist_capping obliterates the type of all columns, turning floating point columns into "objects," make sure this doesn't happen
##restore the original type of columns after norm dist capping
def fix_norm_dist_capping_type_conversion(notCappedDf, cappedDf):
    df_col_types_before_norm_dist_capping = notCappedDf.dtypes
    for i in range(len(cappedDf.columns)):
        if cappedDf.dtypes.values[i] != df_col_types_before_norm_dist_capping[i]:
            cappedDf[cappedDf.columns[i]] = cappedDf[cappedDf.columns[i]].astype(df_col_types_before_norm_dist_capping[i])
    return cappedDf

def read_loader_in(loader, numeric_cols = "notstrict", groupby = "latest", sample_size_frac = 0.95, remove_sigmas = 5, do_log_log_metab = False): ##Call data for a loader and set the index of the df to be the 10k RegistrationCodes if it isn't already
    ##Because CGMLoader needs these as numbers
    norm_dist_capping = {"sample_size_frac": sample_size_frac, "remove_sigmas": remove_sigmas}
    if loader == "Noam": ##The only time this can take a string
        df = pd.read_csv("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/biological_age/raw_data_from_noam/residuals_df_equalzscore.csv").set_index("RegistrationCode")
        df = df.drop("gender", 1) ##Don't GWAS gender, and we already have gender in the covariates file too
    elif loader == SerumMetabolomicsLoader:
        df = fix_norm_dist_capping_type_conversion(loader().get_data(precomputed_loader_fname = "metab_10k_data_RT_clustering_pearson08_present05_baseline", study_ids=["10K"], groupby_reg = groupby).df.copy(), loader().get_data(precomputed_loader_fname = "metab_10k_data_RT_clustering_pearson08_present05_baseline", study_ids=["10K"], groupby_reg = groupby, norm_dist_capping = norm_dist_capping).df.copy())
        df["RegistrationCode"] = list(map(lambda serum: '10K_' + serum.split('_')[0], df.index.values))
        df = df.set_index("RegistrationCode")
    elif loader == GutMBLoader:
        df_all_data = fix_norm_dist_capping_type_conversion(loader().get_data("segal_species", study_ids = ["10K"], groupby_reg = groupby).df.copy(), loader().get_data("segal_species", study_ids = ["10K"], groupby_reg = groupby, norm_dist_capping = norm_dist_capping).df.copy())
        dfmeta = loader().get_data("segal_species", study_ids = ["10K"], groupby_reg = groupby).df_metadata
        df = df_all_data.reset_index(drop = False)
        df["RegistrationCode"] = df.reset_index(drop = False).SampleName.apply(dfmeta.RegistrationCode.to_dict().get)
        df = df.set_index("RegistrationCode").drop("SampleName", axis = 1)
    elif loader == CGMLoader: ##connection ID is a useless field, get rid of it because otherwise it's going to get written as a categorical one
        ##Ayya's new version
        df = pd.read_csv("/net/mraid08/export/genie/LabData/Analyses/ayyak/CGM/iglu/iglu_no_tails.csv").set_index("id").drop("Unnamed: 0", axis=1)
        ##throw away second part of index
        df.index = list(map(lambda longName: longName.split("/")[0], df.index.values))
        df.index.name = "RegistrationCode"
        ##perform the same outlier removal as above
        df = fix_norm_dist_capping_type_conversion(pd.read_csv("/net/mraid08/export/genie/LabData/Analyses/ayyak/CGM/iglu/iglu_no_tails.csv").set_index("id").drop("Unnamed: 0", axis=1), cappedDf = NormDistCapping(sample_size_frac=sample_size_frac, remove_sigmas = remove_sigmas).fit_transform(df))
        ##perform the same grouping of multiple entries as above
        if groupby == "latest":
            df = df.loc[~df.index.duplicated(keep = "last"),:]
        else: ##assume we want the mean
            df = df.groupby("RegistrationCode").mean()
        ##Ayya's old version
        #df = pd.read_pickle("/net/mraid08/export/genie/LabData/Cache/CGMMeasures/cgmquantify_features.cch").set_index("RegistrationCode").drop("ConnectionID", 1)
    elif loader == RetinaScanLoader:
        ##Encode left and right eye measurements as separate columns instead of indexes
        df = RetinaScanLoader().get_data(study_ids = ["10K"]).df.copy().unstack() ##Don't groupby reg here because the extra index (right vs left eye) will mess things up
        df.columns = list(map(lambda col_tuple: str(col_tuple[0]) + "_" + str(col_tuple[1]), df.columns))
        df = fix_norm_dist_capping_type_conversion(df, cappedDf = NormDistCapping(sample_size_frac=sample_size_frac, remove_sigmas = remove_sigmas).fit_transform(df))
        if groupby == "latest":
            df = df.loc[~df.index.get_level_values(0).duplicated(keep="last"), :]
        else:  ##assume we want the mean
            df = df.groupby("RegistrationCode").mean()
    else:
        df = fix_norm_dist_capping_type_conversion(notCappedDf = loader().get_data(study_ids=["10K"]).df.copy(), cappedDf = loader().get_data(study_ids=["10K"], norm_dist_capping = norm_dist_capping, groupby_reg = groupby).df.copy())
    if loader == QuestionnairesLoader: ##cols 734 to 804 in Questionnaires are na for everyone (empty), remove them to avoid future difficulties
        df = df.drop(df.columns[range(733, 805, 1)], axis = 1)
    elif loader == Medications10KLoader:
        df = df.reset_index().set_index("RegistrationCode").drop("Date", 1).drop("Start", 1) ##just consider anyone who took a medication in the past
    else:
        pass
    if loader in [LifeStyleLoader, QuestionnairesLoader, ChildrenLoader]: ##force some columns to be numeric if we know already that they are, the rest get encoded as dummies
        if numeric_cols == "strict":
            force_numeric_cols_list = [x for x in df.columns if "weight" in x or "age" in x or "number" in x or "times" in x or "distance" in x or "people_living_together" in x or "number" in x or "how_often" in x or "duration" in x or "minute" in x or "hour" in x or "day" in x or "week" in x or "month" in x or "year" in x]
        else:
            force_numeric_cols_list = [x for x in df.columns if "weight" in x or "age" in x or "distance" in x or "duration" in x or "minute" in x]
        for col in force_numeric_cols_list:
            if col != "birth_weight":
                try:
                    df[col] = df[col].astype("float64")
                except ValueError: ##also we might include some columns that aren't numeric above, so fix that below
                    pass
            ##there are commas in childrenloader columns that should be int, i.e birthweight, try and catch this
            else:
                df[col] = df[col].apply(fix_birth_weight).astype("float64")
    ##encode all for sure numeric variables as either dummy variables if there is no ordering in the levels, or numeric if there is
    print("There are " + str(len(df.dtypes.index[df.dtypes != "float64"])) + " categorical columns in " + str(loader) + ", converting now")
    if loader == ABILoader:##these object based columns are duration based, so convert them to elapsed time
        for col in df.dtypes.index[df.dtypes != "float64"]:
            df[col] = df[col].apply(get_elapsed_seconds) ##drop lists from the dataframe if they're an entry, an edge case
    elif loader == ItamarSleepLoader:
        ##very similar thing as above
        df["PhysicalSleepTime"] = df["PhysicalSleepTime"].apply(PhysicalTime_seconds)
        df["PhysicalWakeTime"] = df["PhysicalWakeTime"].apply(PhysicalTime_seconds)
    else:
        for col in df.dtypes.index[df.dtypes != "float64"]:
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)##drop lists from the dataframe if they're an entry, an edge case
    if loader == ItamarSleepLoader:
        df = df.drop(["BraceletMessage", "StudyStartTime", "StudyEndTime"], axis = 1)
    ##take out duplicate columns (mainly for ItamarSleepLoader)
    df = df.loc[:, ~df.columns.duplicated()]
    if loader != ChildrenLoader:
        df = pd.get_dummies(df)
    else:
        df = encode_dummy_variables_children(df)
    df.index = df.index.get_level_values("RegistrationCode")
    ##using dummy variables puts spaces in file names which breaks phenotype writing. Replace the spaces with underscores and keep going
    df.columns = df.columns.str.replace(" ", "_")
    ##remove columns with NA colnames
    numNaNames = pd.isna(df.columns).sum()
    if numNaNames != 0:
        ##can't index nans in column names to drop for some reason, so this works
        ##rename the nans to someting else identifiable and unique, and then drop those nans
        df.columns = df.columns.fillna("NA")
        df = df.drop(df.columns.fillna("NA")[df.columns == "NA"], 1) ##can't index nas in column names to drop for some reason, so this works
    for col in df.columns:
        if col[0] == "\n" or len(col.replace("_","")) < 2: ##if we only have one characters in the column name (that wasn't space), get rid of it
            df = df.drop(col, axis = 1)
    ##take out duplicate columns (mainly for ItamarSleepLoader)
    df = df.loc[:, ~df.columns.duplicated()]
    ##remove backslashes so plink doesn't break
    df.columns = pd.Series(list(df.columns)).apply(lambda colName: colName.replace("/",""))
    if loader == GutMBLoader: ##log10 correction, detection limit is 10^(-4), so exclude below that
        for col in df.columns:
            df[col] = df[col].apply(lambda val: np.log10(val) if np.log10(val) > -4 else None)
    if do_log_log_metab: ##The RT cluster is already logged, don't do it again
        if loader == SerumMetabolomicsLoader:
            for col in df.columns: ##no cutoff here
                df[col] = df[col].apply(lambda val: np.log10(val))
                df.loc[df[col] == -np.inf, col] = None
    df = df * 1  ##Multiply by one to map booleans to integers as 1:True, 0:False
    if loader == UltrasoundLoader:
        ##All q_box derived measurements start with q_box, if there's a second q_box in the name of the trait don't gwas it
        ##That would just be the technical input parameter for the q_box itself which we don't care about
        ##There are only 5 values for this trait anyway
        ##"qbox" in a column name (without an underscore) designates a technical parameter, while "q_box" designates a paramter derived from the qbox.
        df = df.loc[:, [x for x in df.columns if "qbox" not in x]]
    return df

def extract_all_pheno(loader, dir="/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos/"):
    df = read_loader_in(loader)
    df.apply(lambda col: write_pheno(col, dir, True), axis=0)
    print("Finished loader: " + str(loader))

def write_all_loaders(singleBatch = True, loaders = loaders_list):
    if not singleBatch:
        for loader in loaders:
            extract_all_pheno(loader)
        ##Save the map of short to long (original) names
        pd.Series(shortened_name_table).to_csv(short_name_dict_fname)
        print("Wrote " + short_name_dict_fname)
    if singleBatch:
        print("Using a single batch - phenotypes are not written to a file first")

##assumes use_covariates = True
def make_plink1_command(pheno_name):
    cmd = plink19_bin + ' --bfile /net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc --pheno /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos/' + pheno_name + '_pheno.txt' + '--pheno-name' +  pheno_name + '--linear no-x-sex --allow-no-sex --noweb --out /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/' + pheno_name + ' --covar /net/mraid08/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt --covar-number 1-12'
    return cmd

def make_plink2_command(pheno_name, use_short_names = False, short_names_table_fname = short_name_dict_fname, batched = True, i = 0, use_pfilter = True, ldmethod = "clump", howmanyPCs = 10, do_noam = False):
    if ldmethod == "clump":
        bfile_loc = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_custom"
    else:
        bfile_loc = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_extra_qc_extra_before_king"
    if not do_noam:
        if batched and use_pfilter:
            cmd = plink2_bin +' --bfile ' + bfile_loc  + ' --king-cutoff 0.22 --pheno /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos_batched/batch' + str(i) + '.txt' + ' --1 --mac 20 --linear no-x-sex hide-covar cols=+ax -allow-no-sex --out /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/batch' + str(i) + ' --covar /net/mraid08/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt --covar-col-nums 2-'+ str(howmanyPCs+3) + ' --variance-standardize --pfilter 0.00000005'
        elif batched and not use_pfilter:
            cmd = plink2_bin +' --bfile '+ bfile_loc + ' --king-cutoff 0.22 --pheno /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos_batched/batch' + str(i) + '.txt' + ' --1 --mac 20 --linear no-x-sex hide-covar cols=+ax -allow-no-sex --out /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/batch' + str(i) + ' --covar /net/mraid08/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt --covar-col-nums 2-'+ str(howmanyPCs+3) + ' --variance-standardize'
        elif not use_short_names:
            cmd = plink2_bin + ' --bfile ' + bfile_loc + ' --king-cutoff 0.22  --pheno /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos/' + pheno_name + '_pheno.txt' + ' --1 --mac 20 --pheno-name ' + pheno_name + ' --linear no-x-sex hide-covar cols=+ax --allow-no-sex --out /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/' + pheno_name + ' --covar /net/mraid08/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt --covar-col-nums 2-'+ str(howmanyPCs+3) + ' --variance-standardize --pfilter 0.00000005'
        else:
            short_names_df = pd.read_csv(short_names_table_fname)
            short_names_df.columns = ["long", "short"]
            short_file_name = short_names_df.loc[short_names_df["long"] == pheno_name, "short"].values[0]  ##the file is named with a short name, which is also the name of the phenotype (column) inside it
            cmd = plink2_bin + ' --bfile ' + bfile_loc + ' --king-cutoff 0.22  --pheno /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos/' + short_file_name + '_pheno.txt' + ' --1 --mac 20 --pheno-name ' + short_file_name + ' --linear no-x-sex hide-covar cols=+ax --allow-no-sex --out /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/' + short_file_name + ' --covar /net/mraid08/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt --covar-col-nums 2-'+ str(howmanyPCs+3) + ' --variance-standardize --pfilter 0.00000005'
    else: ##Switch directories and hardcore the number of PCs to use to avoid using age.
        ##The last column is age, ignore it. We already have gender.
        cmd = plink2_bin + ' --bfile ' + bfile_loc + ' --king-cutoff 0.22 --pheno /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/biological_age/phenos_batched/batch' + str(i) + '.txt' + ' --1 --mac 20 --linear no-x-sex hide-covar cols=+ax -allow-no-sex --out /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/biological_age/gwas_results/batch' + str(i) + ' --covar /net/mraid08/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt --covar-col-nums 2-' + str(12) + ' --variance-standardize'
    return cmd

def update_covariates(dir="/net/mraid08/export/jasmine/zach/height_gwas/", status_table = None, keep_fid = False):
    df = SubjectLoader().get_data(study_ids = ["10K"]).df.copy()
    new_covar_data = df.reset_index().loc[:, ["gender", "age", "RegistrationCode"]]
    new_covar_data.columns = ["gender", "age", "IID"]
    if keep_fid:
        new_covar_data["FID"] = "0"  ##to allign with binary bed file
    ##order matters for plink
        new_covar_data = new_covar_data.loc[:, ["FID", "IID", "gender", "age"]]
    else:
        new_covar_data = new_covar_data.loc[:, ["IID", "gender", "age"]]
    ##the binary files uses gencove ids, not RegistrationCodes. Make the switch
    new_covar_data['IID'] = new_covar_data['IID'].apply(status_table.set_index('RegistrationCode').gencove_id.to_dict().get)
    new_covar_data.dropna(inplace=True)
    ##round floating point values so that plink will accept them
    new_covar_data.gender = new_covar_data.gender.astype("int")
    new_covar_data.age = new_covar_data.age.astype("int")
    old_covar_data = pd.read_csv("/net/mraid08/export/genie/10K/genetics/Gencove/covariates/covariates.eigenvec", sep  = "\t")
    save_covar_data = pd.merge(old_covar_data, new_covar_data, left_on="IID", right_on="IID", how="inner").drop("#FID", axis = 1)
    if keep_fid:
        save_covar_data = save_covar_data[["FID", "IID", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10","gender", "age"]]
    else:
        save_covar_data = save_covar_data[["IID", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10","gender", "age"]]
    save_covar_data.to_csv(dir + "covariates_with_age_gender.txt", sep="\t", index=False, header=True)

def existsAlready(colName, resultDir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/", use_short_names = False):
    if not use_short_names:
        potential_path = resultDir + colName + "." + colName + ".glm.linear" ##the covariate name is repeated in the output, so do this to properly detect it
    else:
        short_names_df = pd.read_csv(short_name_dict_fname)
        short_names_df.columns = ["long", "short"]
        short_file_name = short_names_df.loc[short_names_df["long"] == colName, "short"].values[0]  ##the file is named with a short name, but inside the file we maintain the long phenotype name
        potential_path = resultDir + short_file_name + "." + short_file_name + ".glm.linear" ##the covariate name is repeated in the output, so do this to properly detect it
    return isfile(potential_path)

def all_GWAS(overwrite = True, batched = True, num_batches = 1, use_pfilter = True, ldmethod = "clump", howmanyPCs = 10, do_noam = False):
    os.chdir("/net/mraid08/export/mb/logs/")
    if not batched:
        for loader in loaders_list:
            df = read_loader_in(loader)
            for col in df.columns:
                if overwrite or not overwrite and not existsAlready(col): ##don't use the column name as the job name because of the special characters in it
                    run_plink2(make_plink2_command(col, use_short_names = True, ldmethod = ldmethod, howmanyPCs = howmanyPCs), "gwas", required_memory_gb("/net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc.bed"))
    else:
        for i in range(num_batches):
            run_plink2(make_plink2_command(pheno_name = None, batched = True, i = i, use_pfilter = use_pfilter, ldmethod = ldmethod, howmanyPCs = howmanyPCs, do_noam = do_noam), "gwas", required_memory_gb("/net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc.bed"), threads = 32)
    print("Ran all gwases")

def make_unique_plink_bins():
    thebin = read_original_plink_bins()
    theinds = (~pd.Series(thebin.snp).duplicated()).to_dict()
    thebin_unique = thebin.loc[:,list(theinds.values())]
    thebin_unique.data_array = thebin_unique
    write_plink1_bin(thebin_unique, "/net/mraid08/export/jasmine/zach/height_gwas/all_samples_qc_just_unique.bed", "/net/mraid08/export/jasmine/zach/height_gwas/all_samples_qc_just_unique.bim", "/net/mraid08/export/jasmine/zach/height_gwas/all_samples_qc_just_unique.fam")

def make_already_done(gwas_output_dir, wholeName = False):
    already_done = [f for f in os.listdir(gwas_output_dir) if isfile(join(gwas_output_dir, f))]
    if wholeName:
        return already_done
    # pull away the file prefix to get at just the phenotype names
    already_done = list(map(lambda gwas_file_name: gwas_file_name.split(".glm.linear")[0],
                            already_done))  ##when using batches we don't repeat the phenotype name in the file, so use the .glm.linear files to see which we've already done
    ##also remove the batch number prefix
    already_done = list(map(lambda gwas_file_name: re.split("batch" + "(\d)" + ".", gwas_file_name)[-1], already_done))
    ##sometimes there are still extra periods, pull them away if they're there
    already_done = list(map(lambda gwas_file_name: gwas_file_name.split(".")[-1], already_done))
    return already_done

##call after the phenotypes have been written to batch them
def make_batches(overwrite = False, batch_size = 300, gwas_output_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/", single_pheno_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos/"):
    all_phenos = [f for f in os.listdir(single_pheno_dir) if isfile(join(single_pheno_dir, f))]
    all_phenos = list(map(lambda pheno_file_name: pheno_file_name.split("_pheno.txt")[0], all_phenos))
    already_done = make_already_done(gwas_output_dir)
    if not overwrite:
        all_phenos = list(set(all_phenos) - set(already_done)) ##both have the shorter/replaced (random, sometimes) names, remove ones we've already done
    cols_batches = np.array_split(np.array(all_phenos), len(all_phenos)//batch_size)
    return cols_batches

def order_cols(batch_df, keep_fid):
    batch_df.reset_index(inplace=True)
    if keep_fid:
        batch_df["FID"] = np.zeros(len(batch_df))  ##for backwards compatibility with plink1 (which needs FID), even though plink2 does not, you can  keep the FID column at the begining of the dataframe
    all_other_cols = batch_df.columns.values[batch_df.columns.values != "IID"]
    if keep_fid:
        all_other_cols = all_other_cols[all_other_cols != "FID"]
        batch_df = batch_df.loc[:, ["FID", "IID", *all_other_cols]]
    else:
        batch_df = batch_df.loc[:, ["IID", *all_other_cols]]
    return batch_df

##separated out do do the test_plot
def secondary_filter(all_loaders, plink_data, most_frequent_val_max_freq = 0.95, min_subject_threshold = 2000):
    ##check whether the most frequent value in each column max(col.value_counts) occurs more than 95% of the time
    cols_imbalanced = all_loaders.columns[list(
        ##the denominator should be those people that are not missing, not all people.
        map(lambda colName: max(all_loaders[colName].value_counts().values) > most_frequent_val_max_freq * len(
            all_loaders[colName].dropna()), all_loaders))]
    all_loaders = all_loaders.drop(labels=cols_imbalanced, axis=1)
    print("Dropped " + str(len(cols_imbalanced)) + " cols based on imbalance")
    ##exclude phenotypes with less than min_subject_threshold people also having SNP data
    ##both indices need to be in the same format (gencove or 10K) this to work
    ##right now all_loaders has gencove indices so use the unaltered index from the plink data in the intersection
    cols_not_enough_genetics_data = all_loaders.columns[list(map(lambda colName: len(
        set(all_loaders[colName].index).intersection(set(plink_data.sample.values))) < min_subject_threshold,
                                                                 all_loaders.columns))]
    all_loaders = all_loaders.drop(labels=cols_not_enough_genetics_data, axis=1)
    print("Dropped " + str(len(cols_not_enough_genetics_data)) + " cols without enough matching SNP data")
    return all_loaders

##Pre filter a data frame
def pre_filter(all_loaders, plink_data, most_frequent_val_max_freq = 0.95, min_subject_threshold = 2000, exclusion_filter_fname = None):
    if exclusion_filter_fname is not None:
        exclusion_filter = set(pd.read_csv(exclusion_filter_fname, sep = "\t").IID)
        all_loaders = all_loaders.loc[set(all_loaders.index.values) - exclusion_filter,:]
    ##do all the filtering only on the effective people used by PLINK, i.e those without genetics data
    print("Removing people without genetics data")
    num_original = len(all_loaders.copy())
    all_loaders = all_loaders.loc[list(set(all_loaders.index).intersection(plink_data.sample.values)), :]
    print("Removed " + str(num_original - len(all_loaders)) + " people without matching genetics data")
    print("Filtering features now")
    all_loaders = all_loaders[~all_loaders.index.isna()]
    orig_n_cols = len(all_loaders.columns)
    all_loaders = all_loaders.dropna(axis=1, thresh=min_subject_threshold)
    print("Dropped " + str(orig_n_cols - len(all_loaders.columns)) + " cols based on too many missing values")
    all_loaders = secondary_filter(all_loaders, plink_data, most_frequent_val_max_freq=most_frequent_val_max_freq, min_subject_threshold = min_subject_threshold)
    return all_loaders

def compare_with_ukbb(ukbb_fname, tenk_fname):
    ukbb_gwas, tenk_gwas = pd.read_csv(ukbb_fname, sep = "\t"), pd.read_csv(tenk_fname, sep = "\t")
    ukbb_gwas_sig, tenk_gwas_sig = ukbb_gwas.loc[ukbb_gwas.pval < 5*10**-8], tenk_gwas.loc[tenk_gwas.P < 5*10**-8]
    neale_newindex = list(map(lambda index: index.split("[")[0], ukbb_gwas_sig["variant"]))
    ukbb_gwas_sig["ID"] = neale_newindex
    ukbb_gwas_sig ["ID"] = ukbb_gwas_sig["ID"].str.replace(":", "") + ukbb_gwas_sig[""]
    ukbb_gwas_sig["ID"] = ukbb_gwas_sig["ID"] + ukbb_gwas_sig["ref_allele"] + ukbb_gwas_sig["alt_allele"]
    tenk_gwas_sig["newid"] = list(map(lambda theint: str(theint), tenk_gwas_sig["POS"])) + tenk_gwas_sig["AX"] + tenk_gwas_sig["A1"]
    tenk_gwas_sig["newid"] = list(map(lambda theint: str(theint), tenk_gwas_sig["#CHROM"])) + tenk_gwas_sig["newid"]
    res = pd.merge(ukbb_gwas_sig, tenk_gwas_sig, left_on="ID", right_on="newid", how="inner")
    return res

##Use gencove indices as the index
def make_extra_sleep_cols():
    s = read_loader_in(ItamarSleepLoader)
    s["Average_Apnea_Length"] = [s.loc[tenK_index, "TotalApneaSleepTime"]/s.loc[tenK_index, "TotalNumberOfApneas"] if s.loc[tenK_index, "TotalNumberOfApneas"] != 0 else 0 for tenK_index in s.index]
    return s["Average_Apnea_Length"]

def make_extra_liver_cols():
    att_plus_phenos = ["att_plus_ssp_plus_db_cm_mhz_1_att_plus", "att_plus_ssp_plus_db_cm_mhz_2_att_plus", "att_plus_ssp_plus_db_cm_mhz_3_att_plus"]
    ssp_plus_phenos = ["att_plus_ssp_plus_m_s_3_ssp_plus", "att_plus_ssp_plus_m_s_2_ssp_plus", "att_plus_ssp_plus_m_s_3_ssp_plus"]
    us = read_loader_in(UltrasoundLoader)
    us["Average_SSP_Plus"] = us[ssp_plus_phenos].mean(1)
    us["Average_ATT_Plus"] = us[att_plus_phenos].mean(1)
    return us[["Average_SSP_Plus", "Average_ATT_Plus"]]

def make_extra_cols_gwas():
    df_10K = pd.merge(make_extra_liver_cols(), make_extra_sleep_cols(), left_on = "RegistrationCode", right_on = "RegistrationCode", how = "outer")
    temp_index = df_10K.reset_index().RegistrationCode.apply(status_table.set_index('RegistrationCode').gencove_id.to_dict().get)
    temp_index.name = "IID"
    df_10K = df_10K.set_index(temp_index)
    return df_10K

def write_all_batches(singleBatch = True, keep_fid = False, single_pheno_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos/", batched_pheno_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos_batched/", min_subject_threshold = 2000, most_frequent_val_max_freq = 0.95, plink_data = None, exclusion_filter_fname = None, only_write_noam = False):
    if not singleBatch:
        cols_batches = make_batches(single_pheno_dir = single_pheno_dir)
        i = 0
        for batch in cols_batches:
            j = 1
            print("Now starting batch: " + str(i))
            firstfile = pd.read_csv(single_pheno_dir + batch[0] + "_pheno.txt", sep="\t").drop("FID", 1).set_index("IID")
            while j < len(batch):
                newpheno = pd.read_csv(single_pheno_dir + batch[j] + "_pheno.txt", sep="\t").drop("FID", 1).set_index("IID")
                if len(newpheno) != 0: ##catch the empty phenos from dummy variables
                    firstfile = firstfile.merge(newpheno, left_index=True, right_index=True, how = "outer")
                j += 1
            batch_df = order_cols(firstfile, keep_fid)
            batch_df.to_csv(batched_pheno_dir + "batch" + str(i) + ".txt", sep="\t", index=False, header=True)
            print("Wrote batch: " + str(i))
            i += 1
        return len(cols_batches)
    else:
        print("Single batch option selected")
        if only_write_noam:
            print("Only writing Noam's phenotypes to batch")
            all_loaders = read_loader_in("Noam")
        else:
            all_loaders =  read_loader_in(loaders_list[0])
            for j in range(1, len(loaders_list)):
                print("Reading loader: " + str(loaders_list[j]))
                all_loaders = all_loaders.merge(read_loader_in(loaders_list[j]), left_index=True, right_index=True, how="outer")
        temp_index = all_loaders.reset_index().RegistrationCode.apply(status_table.set_index('RegistrationCode').gencove_id.to_dict().get)
        temp_index.name = "IID"
        all_loaders = all_loaders.set_index(temp_index)
        all_loaders = pre_filter(all_loaders, plink_data, most_frequent_val_max_freq, min_subject_threshold, exclusion_filter_fname = exclusion_filter_fname)
        print("Manually adding artificial columns now")
        ##Both indices are called IID so this should work
        all_loaders = all_loaders.merge(make_extra_cols_gwas(), left_index = True, right_index = True, how = "outer")
        all_loaders = order_cols(all_loaders, keep_fid)
        print("Writing all loaders to batch now")
        ##using fillNA is much slower than using na_rep at the time we write the batch csv file
        all_loaders.to_csv(batched_pheno_dir + "batch" + str(0) + ".txt", sep="\t", index=False, header=True, na_rep = "NA")
        print("Wrote batch to csv, N = " , str(len(all_loaders)))
        return 1

def rename_results(gwas_output_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/", gwas_output_renamed_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results_renamed/"):
    already_done_whole = make_already_done(gwas_output_dir, wholeName = True) ##actual names of the saved files
    already_done_short = make_already_done(gwas_output_dir, wholeName = False) ##names in the renaming table
    short_name_table_inverse = pd.read_csv(short_name_dict_fname).set_index("0")["Unnamed: 0"].to_dict()
    numResults = len(already_done_whole)
    for i in range(numResults): ##lists are in the same order
        long_filename = already_done_whole[i]
        ##exclude the log files
        if "log" not in long_filename:
            gwas = pd.read_csv(gwas_output_dir+long_filename, sep = "\t")
            try:
                true_name = short_name_table_inverse[already_done_short[i]].replace("/", "") ##just remove the backslashes from the true names
            except KeyError:
                true_name = long_filename
            gwas.to_csv(gwas_output_renamed_dir + true_name + ".glm.linear", sep = "\t", index = False, header = True)
            print("Renamed GWAS for phenotype: ", true_name + ", " + str(i + 1) + "/" + str(numResults)) ##direpancy is due to skipping plink2 log files

def get_broken_keys_medications(key_list):
    key_list = np.array(key_list)
    return list(key_list[key_list != list(map(lambda key: key.replace("medication_{", ""), key_list))])

##Use a hybrid (generated) but also unique index for each SNP because we don't have rsids for many of them
##hybrid index by default is "CHROM":"POS":"REF":"ALT"
def generate_hybrid_index(hits_df, cols = ["CHROM", "POS", "REF", "ALT"]):
    ##convert each column to a string and then concatenate them
    return set(hits_df[cols].apply(lambda col: col.astype(str)).apply(":".join, axis = 1))

def summarize_gwas(onlythesecols = None, forHist = False, threshold = False, use_clumped = True, containing_dir =  "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/", clump_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results_clumped/", singleBatch = True):
    if use_clumped:
        containing_dir = clump_dir
        sep = "\s+|\t+|\s+\t+|\t+\s+"
    else:
        sep = "\t"
    if use_clumped:
        all_gwases = [f for f in os.listdir(containing_dir) if isfile(join(containing_dir, f)) and f.endswith(".clumped")]
    else:
        all_gwases = [f for f in os.listdir(containing_dir) if isfile(join(containing_dir, f))]
    numGwases = len(all_gwases) ##correct for all the GWASES that we did, not just the ones from each loader.
    if forHist:
        numGwases = 1
        threshold = True
    if onlythesecols is not None:
        ##intersect the columns from the loader with the gwases we actually have
        if singleBatch and not use_clumped:
            all_gwases = list(set(all_gwases).intersection(set(list(map(lambda res_name: "batch0." + res_name + ".glm.linear", onlythesecols)))))
        elif not singleBatch and not use_clumped:
            all_gwases = list(set(all_gwases).intersection(set(list(map(lambda res_name: res_name + ".glm.linear", onlythesecols)))))
        else:
            all_gwases = list(set(all_gwases).intersection(set(list(map(lambda res_name: "batch0." + res_name + ".clumped", onlythesecols)))))
    hits = {}
    rank = dict(zip([long_filename.split(".glm.linear")[0] for long_filename in all_gwases], [0 for gwas in all_gwases]))
    i = 1
    for long_filename in all_gwases:
        if (".clumped" in long_filename and use_clumped) or (not use_clumped and ".glm.linear" in long_filename):
            gwas = pd.read_csv(containing_dir + long_filename, sep=sep)
            if len(gwas.P) != 0: #ignore gwases with no significant associations
                gwas.columns = ["CHROM", *gwas.columns[range(1, len(gwas.columns))]] ##fix the hashtag in the name of the chromosome column
                if not threshold:
                    hits[long_filename.split(".glm.linear")[0]] = gwas
                else:
                    hits[long_filename.split(".glm.linear")[0]] = gwas.loc[gwas.P < (5*10**(-8))/numGwases,:]
                rank[long_filename.split(".glm.linear")[0]] = len(gwas.loc[gwas.P < (5*10**(-8))/numGwases,:])
            print(str(i) + "/" + str(len(all_gwases)))
        i += 1
    hits = pd.Series(hits)
    rank = pd.Series(rank).sort_values(ascending = False)
    return hits, rank

def stack_our_gwases(use_clumped = True, onlythesecols = None):
    hits, rank = summarize_gwas(use_clumped=use_clumped, onlythesecols=onlythesecols)
    for k, v in hits.items():
        hits[k]["pheno"] = k.split("batch0.")[1].split(".clumped")[0].lower()
    ans = pd.concat(hits.values)
    ans = ans.rename({"P": "p_ours"}, axis = 1)
    ans = ans.sort_values("p_ours")
    return ans

def merge_with_pheno_team(use_clumped = True):
    ans = stack_our_gwases(use_clumped = use_clumped, onlythesecols = read_loader_in(CGMLoader).columns)
    theirs = pd.read_csv("~/Desktop/pheno_cgm_assoc.csv")
    theirs["phenotype"] = list(map(lambda thestr: thestr.lower().replace(".", ""), theirs["phenotype"]))
    theirs = theirs.rename({"P": "p_theirs"}, axis = 1)
    tog = pd.merge(ans, theirs, left_on  = ["SNP", "pheno"], right_on = ["SNP", "phenotype"], how = "inner")
    return tog

def plot_with_pheno_team():
    tog = merge_with_pheno_team()
    groups = tog.groupby("phenotype")
    for name, group in groups:
        plt.plot(group.p_ours, group.p_theirs, marker = "o", linestyle = "", label = name)
        plt.legend()
        plt.show()

##prebuilt qc pipelines
def genetics_qc(sexcheck = False, ldmethod = "clump"):
    ##Remove people with broken sex
    ##doesn't work without X chromosome data, which we don't currently have.
    def gwas_first_pass_sexcheck(covariates_file="/net/mraid08/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt", containing_dir="/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/"):
        base_covars = pd.read_csv(covariates_file, sep="\t")[["IID", "gender"]]
        base_covars["FID"] = 0
        base_covars = base_covars.loc[:, ["FID", "IID", "gender"]]
        base_covars.to_csv(containing_dir + "gender.txt", sep="\t", index=False, header=False)
        add_gender_qc_cmd = plink19_bin + ' --bfile /net/mraid08/export/genie/10K/genetics/Gencove/allsamples --mind --geno --maf 0.03 --hwe 0.000001 --update-sex ' + containing_dir + 'gender.txt  --make-bed --out ' + containing_dir + "allsamples_qc_with_gender"
        ShellCommandsExecute().run(add_gender_qc_cmd)
        ##we need a separate genotype file with sex encoded for this to work
        check_sex_cmd = plink19_bin + ' --bfile ' + containing_dir + "allsamples_qc_with_gender" + ' --check-sex --out ' + containing_dir + "gender_check"
        ShellCommandsExecute().run(check_sex_cmd)
        sexcheck = pd.read_csv(containing_dir + "gender_check.sexcheck", delim_whitespace=True)
        sexcheck_partial = sexcheck[["IID", "STATUS"]]
        sexcheck_partial_problem = sexcheck_partial[sexcheck_partial["STATUS"] != "OK"]
        sexcheck_partial_problem["FID"] = 0
        broken_people = sexcheck_partial_problem[["FID", "IID"]]
        broken_people_fname = containing_dir + "brokenpeople.txt"
        broken_people.to_csv(broken_people_fname, sep="\t", index=False)
        exclude_broken_people_cmd = plink19_bin + ' --bfile /net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc --remove ' + broken_people_fname + ' --make-bed --out ' + containing_dir + "allsamples_qc_custom"
        ShellCommandsExecute().run(exclude_broken_people_cmd)
    ##GWAS QC pipeline without sex check, identical
    def gwas_qc_first_pass_no_sexcheck(containing_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/"):
        qc_cmd = plink19_bin + ' --bfile /net/mraid08/export/genie/10K/genetics/Gencove/allsamples --mind --geno --maf 0.03 --hwe 0.000001 --make-bed --out ' + containing_dir + "allsamples_qc_custom"
        ShellCommandsExecute().run(qc_cmd)
    def gwas_ld_calc(containing_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/"):
        ld_cmd = plink19_bin + ' --bfile ' + containing_dir + 'allsamples_qc_custom' + ' --indep-pairwise 100 5 0.2 --out ' + containing_dir + "ld"
        ShellCommandsExecute().run(ld_cmd)
    def gwas_ld_prune(containing_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/"):
        extract_ld_variants_cmd = plink19_bin + ' --bfile ' + containing_dir + 'allsamples_qc_custom' + ' --exclude ' + containing_dir + "ld.prune.out" + ' --make-bed --out ' + containing_dir + "allsamples_extra_qc_extra_before_king"
        ShellCommandsExecute().run(extract_ld_variants_cmd)
    print("Starting first pass genetics QC")
    if sexcheck:
        gwas_first_pass_sexcheck()
    else:
        gwas_qc_first_pass_no_sexcheck()
    if ldmethod == "prune":
        gwas_ld_calc()
        gwas_ld_prune()
    else:
        pass


##use plink2 for this because the output from plink1 is not readable
def check_read_freq(containing_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/"):
        check_freq_cmd = plink2_bin + ' --bfile ' + containing_dir + 'allsamples_extra_qc_extra_before_king --freq --out ' + containing_dir + 'freq'
        run_plink2(check_freq_cmd, "freq", required_memory_gb("/net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc.bed"))

def get_duplicate_ids(qc_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/", outfilename = "duplicate_id_snps.snplist"):
    write_all_cmd = plink19_bin + ' --bfile ' + qc_dir + 'allsamples_qc_custom --write-snplist --out ' + qc_dir + 'all_snps'
    run_plink1(write_all_cmd, "write_all_snpids", required_memory_gb("/net/mraid08/export/genie/10K/genetics/Gencove/allsamples_qc.bed"))
    extract_duplicate_id_snps_cmd = 'cat ' + qc_dir + 'all_snps.snplist' + ' | sort | uniq -d > ' + qc_dir + outfilename
    ShellCommandsExecute().run(extract_duplicate_id_snps_cmd, "get_duplicate_snpids")

def wasGwasedToday(fn):
    return datetime.datetime.fromtimestamp(os.path.getmtime(fn)).day == datetime.date.today().day

##clumping procedure for gwases with at least one significant hit
def clump(qc_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/", gwas_results_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/", clump_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results_clumped/", duplicate_id_snps_fname = "duplicate_id_snps.snplist"):
    print("Clumping from ", gwas_results_dir, " clumping into directory ", clump_dir)
    all_gwases = [f for f in os.listdir(gwas_results_dir) if isfile(join(gwas_results_dir, f))]
    alreadyclumped = list(map(lambda fname: fname.split(".")[1], [f for f in os.listdir(clump_dir) if isfile(join(clump_dir, f))]))
    gwases_from_today = [x for x in all_gwases if wasGwasedToday(gwas_results_dir + x)]
    numGwases = len(all_gwases) ##correct for all the GWASES that we did, not just the ones from each loader.
    res = {}
    os.chdir(qp_running_dir)
    with qp(jobname= "clump", _mem_def=f'{required_memory_gb("/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_extra_qc/allsamples_qc_custom.bed")}G',_trds_def=32, delay_batch = 30, _tryrerun=True) as q:
        q.startpermanentrun()
        ##Penalize based on all gwases, but only clump gwases done today
        for long_filename in gwases_from_today:
            if long_filename.split(".")[1] not in alreadyclumped and "clumpheader" not in long_filename: ##Don't treat clumpheader files as new traits, we only need them for plink
                gwas = pd.read_csv(gwas_results_dir + long_filename, sep="\t")
                if "P" in gwas.columns:
                    if len(gwas.P) != 0:  # ignore gwases with no significant associations
                        gwas.columns = ["CHROM", *gwas.columns[range(1, len(gwas.columns))]]  ##fix the hashtag in the name of the chromosome column
                        if len(gwas.loc[gwas.P < (5 * 10 ** (-8)) / numGwases, :]) >= 1:
                            print("At least one significant hit found for " + long_filename.split(".glm.linear")[0] + " clumping now.")
                            ##the clump CMD needs the snp id column to be named 'SNP', not 'ID' it is supposed to be by plink.
                            ##To fix this, change the header name manually of files we're clumping
                            ###Don't overwrite the original, save this one with a special prefix
                            gwas.columns = ["CHROM", "POS", "SNP", *gwas.columns[range(3, len(gwas.columns))]]
                            gwas.to_csv(gwas_results_dir + "clumpheader" + long_filename,  index = False, header = True, sep = "\t")
                            clump_cmd = plink19_bin + ' --bfile ' + qc_dir + 'allsamples_qc_custom --clump ' + gwas_results_dir + "clumpheader" + long_filename + ' --exclude ' + qc_dir + duplicate_id_snps_fname + ' --out ' + clump_dir + long_filename.split(".glm.linear")[0]
                            res[long_filename] = q.method(lambda: ShellCommandsExecute().run(cmd=clump_cmd, cmd_name=long_filename.split(".")[1]))
        res = {k:q.waitforresult(v) for k, v in res.items()}

if __name__ == "__main__":
    ##avoid breaking scores_work and other modules that use the queue and import this file
    sethandlers()
    ##only load the status table once and pass it around to save on memory
    status_table = read_status_table()
    status_table = status_table[status_table.passed_qc].copy()
    shortened_name_table = {}
    ##Since encoding dummy variables also breaks file names because theres spaces in the factor levels, use random names for all gwases
    do_batched = True
    min_subject_threshold = 2000
    singleBatch = True ##much faster
    redo_setup = False
    ##Can use the close relations pruning list from a previous GWAS run
    #exclusion_filter_fname = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/batch0.king.cutoff.out.id"
    exclusion_filter_fname = None
    remake_batches = False
    do_GWAS = False
    do_noam = False
    lenbatches = 1
    do_renaming = False
    redo_genetics_qc = False
    ldmethod = "clump"
    do_clumping = False
    use_pfilter = False
    pass_cmd = False
    howmanyPCs = 10
    redo_get_duplicate_ids = False ##for clumping, right now it's just the ID "."
    if redo_genetics_qc:
        genetics_qc()
    if redo_setup:
        update_covariates(status_table=status_table)
        write_all_loaders(singleBatch = singleBatch)
    if remake_batches and do_batched:
        plink_data_loaded = read_original_plink_bins()
        if do_noam:
            batched_pheno_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/biological_age/phenos_batched/"
        else:
            batched_pheno_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/phenos_batched/"
        lenbatches = write_all_batches(singleBatch = singleBatch, min_subject_threshold = min_subject_threshold, plink_data = plink_data_loaded, exclusion_filter_fname = exclusion_filter_fname, batched_pheno_dir = batched_pheno_dir, only_write_noam = do_noam)
    if do_GWAS:
        ##The plink data can't be pickled, so get rid of it to avoid q.startpermanentrun errors
        plink_data_loaded = None
        all_GWAS(overwrite = True, batched = do_batched, num_batches = lenbatches, use_pfilter = use_pfilter, ldmethod = ldmethod, howmanyPCs=howmanyPCs, do_noam = do_noam)
    if redo_get_duplicate_ids:
        get_duplicate_ids()
    if ldmethod == "clump" and do_clumping:
        if pass_cmd == True:
            gwas_results_dir = sys.argv[1]
            clump_dir = sys.argv[2]
        else:
            if do_noam:
                gwas_results_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/biological_age/gwas_results/"
                clump_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/biological_age/gwas_results_clumped/"
            else:
                gwas_results_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results/"
                clump_dir = "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results_clumped/"
        clump(gwas_results_dir=gwas_results_dir, clump_dir=clump_dir)
    if do_renaming and not singleBatch:
        rename_results()
