
##useful chunk
library("dplyr")
LD.path <- "/net/mraid08/export/genie/10K/genetics/TestsCache/zach/height_gwas/UKB_imputed_SVD_eigen99_extraction"

LD.files <- list.files(LD.path)
if(any(grepl(x = LD.files, pattern = ".*_snp_counter.*"))){
  snp_counter_file <- LD.files[grep(x = LD.files, pattern = ".*_snp_counter.*")]
  snp_list_file <- LD.files[grep(x = LD.files, pattern = ".*_snp_list.*")]
  load(file=paste(LD.path, snp_counter_file, sep = "/"))
  load(file=paste(LD.path, snp_list_file, sep = "/"))
  if("nsnps.list.imputed" %in% ls()){
    snps.name.list <- snps.list.imputed.vector
    nsnps.list <- nsnps.list.imputed
  }
  if(is.null(names(nsnps.list))) names(nsnps.list) <- as.character(1:length(nsnps.list))
} else{
  error.message <- "It seems this directory does not contain all files needed for HDL. Please check your LD.path again. The current version of HDL only support pre-computed LD reference panels."
  if(output.file != ""){
    cat(error.message, file = output.file, append = T)
  }
  stop(error.message)
}
gwas <- read.table("/net/mraid08/export/genie/LabData/Data/10K/genetics/PRSice/SummaryStatistics/Nealelab/v3/RawData/50_irnt.gwas.imputed_v3.both_sexes.tsv", head = TRUE)
dictionary_file <- LD.files[grep(x = LD.files, pattern = "snp.dictionary.*")]
load(file=paste(LD.path, dictionary_file, sep = "/"))
gwas.hdl.df <- gwas %>%
  inner_join(snp.dictionary %>% filter(rsid %in% snps.name.list), by = "variant")  %>%
  select(rsid, alt, ref, n_complete_samples, tstat, pval) %>%
  rename(SNP = rsid, A1 = alt, A2 = ref, N = n_complete_samples, Z = tstat,P = pval)
write.table(gwas.hdl.df, "/net/mraid08/export/genie/10K/genetics/TestsCache/zach/height_gwas/height_ukbb.sumstats",quote = FALSE, row.names = FALSE, sep = " ")

##final attempt
gwas_res_new <- read.table("/net/mraid08/export/genie/10K/genetics/TestsCache/zach/height_gwas/height_pheno.txt.height.glm.linear", head = TRUE)
colnames(gwas_res_new) <- c("CHROM",	"POS",	"ID",	"REF",	"ALT",	"A1",  "AX",	"TEST",	"OBS_CT",	"BETA",	"SE",	"T_STAT",	"P")
gwas_res_new_just_test <- gwas_res_new[gwas_res_new$TEST == "ADD",] ##exlude PCS
gwas_res_new_just_test$rsid <- gwas_res_new_just_test$id <- gwas_res_new_just_test$SNP
#gwas_res_new_just_test$hybrid_index <- paste0(gwas_res_new_just_test$CHR, ":", gwas_res_new_just_test$BP, ":" , gwas_res_new_just_test$A1)
maybe <- gwas_res_new_just_test
#maybe <- maybe[!duplicated(maybe),]
maybe <- maybe[,c("ID", "A1", "AX", "OBS_CT", "T_STAT", "P")]
colnames(maybe) <- c("SNP", "A1", "A2", "N", "Z", "P")
maybe <- maybe[maybe$A1 %in% c("A","T","C","G"),] ##QC for ldsc to work, remove "ATAT" type entries
maybe <- maybe[maybe$A2 %in% c("A","T","C","G"),]
write.table(maybe, "/net/mraid08/export/genie/10K/genetics/TestsCache/zach/height_gwas/height_tenk.sumstats",quote = FALSE, row.names = FALSE, sep = " ")
