import scipy.stats
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr
from GeneticsPipeline.helpers_genetic import read_status_table

##test can be "t" for t test, or "r" for regression
##t testing expects an index of PRS_class = 0,1,2
##takes a single batch and computes a test for people split into high and lowprs for each column
##regression expects the index to be the prs itself
##the swap argument swaps y and x after data selection, which is useful for inverting the model, and is only supported for test = "r"
def manyTestsbatched(batch, test, tailsTest, swap = False):
    pd.options.mode.use_inf_as_na = True ##treat inf as na values to save number of checks
    batchtypes = dict(batch.dtypes)
    if test == "corrected_regression":
        covars = pd.read_csv("/net/mraid08/export/jasmine/zach/height_gwas/covariates_with_age_gender.txt", sep="\t")#if using plink1 covariates, need to .drop("FID", axis=1)
        status_table = read_status_table()
        try:
            status_table = status_table[status_table.passed_qc].copy()
        except ValueError:  ##In case the genetics pipeline is running
            status_table = status_table.dropna()
            status_table = status_table[status_table.passed_qc].copy()
        covars["RegistrationCode"] = covars["IID"].apply(status_table.set_index("gencove_id").RegistrationCode.to_dict().get)
        covars = covars.drop("IID", axis = 1).set_index("RegistrationCode", drop = True)
    pvals = {}
    ##fix numberofwakes type errors with duplicate columns
    ##the value at index var corresponds to whether var is a repeated column or not.
    duplicated_cols = pd.DataFrame(batch.columns.duplicated(), index = batch.columns)
    # exclude missing data by default
    operative_cols = [x for x in batch.columns if x not in ["prs"]]
    for var in operative_cols: ##for corrected_regression the index is 10K so prs is a column, ignore it and use it separately
        ##if the column is not repeated. We have to do this before indexing the columns to avoid indexing errors.
        if sum(duplicated_cols[duplicated_cols.index == var][0]) == 0:
            ## if the data is non numeric or entirely missing, skip
            if test == "t" or test == "m":
                should_skip = batchtypes[var] != "float64" or batch[batch.index == 1][var].isna().sum() == len(batch[batch.index == 1]) or batch[batch.index == 2][var].isna().sum() == len(batch[batch.index == 2])
            else:
                ##the condition is different because for regression, we don't have the prs classes
                should_skip = batchtypes[var] != "float64" or batch[var].isna().sum() == len(batch)
            if should_skip:
                pvals[var] = None
            else:
                ##we have a numeric variable, so run tests
                ##w're indexing by columns, we always want the same rows
                if test == "t":
                    if tailsTest == "rightLeft":
                        test_res = ttest_ind(batch[batch.index == 1][var].dropna(), batch[batch.index == 2][var].dropna())
                    else:
                        test_res = ttest_ind(batch[batch.index == 2][var].dropna(), batch[batch.index != 2][var].dropna())
                    pvals[var] = test_res[1]
                elif test == "m":
                    if tailsTest == "rightLeft":
                        test_res = mannwhitneyu(batch[batch.index == 1][var].dropna(), batch[batch.index == 2][var].dropna())
                    else:
                        test_res = mannwhitneyu(batch[batch.index == 2][var].dropna(), batch[batch.index != 2][var].dropna())
                    pvals[var] = float(test_res.pvalue)
                elif test == "corrected_regression":
                    batch_droppedna_this_var = batch.dropna(axis = 0, subset = [var]).merge(covars, left_index = True, right_index = True, how = "inner")
                    ##special characters in any variable names breaks hypothesis testing, so do the testing with a different name and then save under the original one
                    batch_droppedna_this_var["y"] = batch_droppedna_this_var[var]
                    ##Use a formula so we can access these names in hypothesis testing later
                    formula = "y ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + gender + age + prs"
                    ##test whether the PRS coefficient is different from zero
                    try:
                        model = ols(formula,batch_droppedna_this_var).fit()  ##remember that we need the formula-based version of this function
                        hypotheses = "(prs = 0)"
                        test_ = model.f_test(hypotheses)
                        pval = float(test_.pvalue)
                        pvals[var] = pval
                    except Exception as e: ##catch hypothesis testing errors
                        pvals[var] = None
        else: ##add None the same time as the number of repeats to allign the indexes with column names of the batch
            NumberofNones = sum(duplicated_cols[duplicated_cols.index == var][0]) + 1
            for NoneNumber in range(NumberofNones):
                p_label = var + "_" + str(NoneNumber) ##avoid duplicate column names in the output
                pvals[p_label] = None
    return pd.Series(pvals)
