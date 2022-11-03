library(qqman)
##generate Manhattan plots for any traits which have any SNPS that pass correction
library(genio)
library(patchwork)
library(ggplotify)

basepath <- "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results_clumped/"
figpath_individual <- "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results_figures/"
figpath_panels <- "/net/mraid08/export/jasmine/zach/height_gwas/all_gwas/gwas_results_figures_panels/"
all_gwases <- list.files(basepath)
numGwases = length(all_gwases) ##correct for all the GWASES that we did, not just the ones from each loader.

read_clumped <- function(fname){
  read_in <- do.call(rbind, lapply(strsplit(readLines(fname), "\\s+|\\t+|\\s+\\t+|\\t+\\s+"), function(x){as.data.frame(t(x))}))
  read_in <- read_in[, 1:6]
  colnames(read_in) <- read_in[1,]
  read_in <- read_in[-1,] ##Drop the column names as the first entry
  read_in$CHR <- as.numeric(read_in$CHR)
  read_in$"F" <- as.numeric(read_in$"F")
  read_in$BP <- as.numeric(read_in$BP)
  read_in$P <- as.numeric(read_in$P)
  read_in
}

##make all plots for traits with significant hits and put them in a directory
make_all_gwas_plots_individual <- function(){
    for (long_filename in all_gwases){
    if (endsWith(long_filename, ".clumped")){
      gwas <- read_clumped(paste0(basepath, long_filename))
      print(head(gwas))
      if (length(gwas$P) != 0){
        if (nrow(gwas[gwas$P < (5*10**(-8))/numGwases,]) != 0){
          jpeg(stringr::str_replace_all(paste0(figpath_individual, long_filename, ".jpg"), "%", ""))
          manhattan(gwas, chr = "CHR",
                    bp = "BP",
                    snp = "SNP",)
          dev.off()

    }
  
      }
    }
  }
}

##from https://danielroelfs.com/blog/how-i-create-manhattan-plots-using-ggplot/
ggplot_manhattan <- function(gwas_data, theTitle = "Manhattan Plot"){
  gwas_data$p <- gwas_data$P
  gwas_data$chr <- gwas_data$X.CHROM
  gwas_data$bp <- gwas_data$POS
  library(ggplot2)
  library(dplyr)
  data_cum <- gwas_data %>% 
    group_by(chr) %>% 
    summarise(max_bp = max(bp)) %>% 
    mutate(bp_add = lag(cumsum(as.numeric(max_bp)), default = 0)) %>% 
    select(chr, bp_add)
  
  gwas_data <- gwas_data %>% 
    inner_join(data_cum, by = "chr") %>% 
    mutate(bp_cum = bp + bp_add)
  axis_set <- gwas_data %>% 
    group_by(chr) %>% 
    summarize(center = mean(bp_cum))
  
  ylim <- gwas_data %>% 
    filter(p == min(p)) %>% 
    mutate(ylim = abs(floor(log10(p))) + 2) %>% 
    pull(ylim)
  
  sig <- 5e-8
  manhplot <- ggplot(gwas_data, aes(x = bp_cum, y = -log10(p), 
                                    color = as.factor(chr), size = -log10(p))) +
    geom_hline(yintercept = -log10(sig), color = "grey40", linetype = "dashed") + 
    geom_point(alpha = 0.75) +
    scale_x_continuous(label = axis_set$chr, breaks = axis_set$center) +
    scale_y_continuous(expand = c(0,0), limits = c(0, ylim)) +
    scale_color_manual(values = rep(c("#276FBF", "#183059"), unique(length(axis_set$chr)))) +
    scale_size_continuous(range = c(0.5,3)) +
    labs(x = NULL,
         #https://sites.google.com/view/stuck-in-the-shallow-end/home/generate-manhattan-plots-with-ggplot2-and-ggrastr
         title = theTitle) + 
    ylab(expression(paste(-log[10],"(", italic(P), ")"))) +
    theme_minimal() +
    theme( 
      legend.position = "none",
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      axis.text.x = element_text(angle = 60, size = 8, vjust = 0.5)
    )
  manhplot
}

panel_plots_panelby_loader<- function(loader = "metabolomics"){
  library("patchwork")
  if (loader == "bloodtests"){
    operative_gwases <- c('batch0.bt__protein_total.glm.linear',
      'batch0.bt__bilirubin_total.glm.linear',
      'batch0.bt__uric_acid.glm.linear',
      'batch0.bt__basophils_%.glm.linear',
      'batch0.bt__glucose.glm.linear',
      'batch0.bt__monocytes_abs.glm.linear',
      'batch0.bt__vitamin_d.glm.linear',
      'batch0.bt__hdl_cholesterol.glm.linear',
      'batch0.bt__folic_acid.glm.linear',
      'batch0.bt__mean_platelet_volume.glm.linear',
      'batch0.bt__alkaline_phosphatase.glm.linear',
      'batch0.bt__tsh.glm.linear',
      'batch0.bt__monocytes_%.glm.linear')
  }
  else if (loader == "cgm"){
    operative_gwases <- c('batch0.Median.glm.linear',
      'batch0.lbgi.glm.linear',
      'batch0.j_index.glm.linear',
      'batch0.AUC.glm.linear',
      'batch0.adrr.glm.linear',
      'batch0.1st_Qu..glm.linear',
      'batch0.Min..glm.linear',
      'batch0.Mean.glm.linear',
      'batch0.3rd_Qu..glm.linear',
      'batch0.grade_hypo.glm.linear',
      'batch0.ea1c.glm.linear',
      'batch0.hypo_index.glm.linear',
      'batch0.gmi.glm.linear')
  }
  else if (loader == "ultrasound"){
    operative_gwases <- c('batch0.q_box_mean_mm_qbox_diameter.glm.linear',
      'batch0.q_box_mm_2_qbox_diameter.glm.linear',
      'batch0.q_box_median_mm_qbox_diameter.glm.linear',
      'batch0.q_box_mm_1_qbox_diameter.glm.linear',
      'batch0.q_box_mm_3_qbox_diameter.glm.linear')
    
  }
  else if (loader  == "metabolomics"){
    operative_gwases <- c('batch0.Lipids_POS_951.7732_412.1897_348.8896',
                           'batch0.Lipids_POS_772.5478_312.7962_299.3659',
                           'batch0.Lipids_POS_878.6730_357.8089_331.5334',
                           'batch0.Lipids_POS_925.7570_412.2867_343.8190',
                           'batch0.Lipids_POS_934.7356_388.2823_344.5864',
                           'batch0.Lipids_NEG_736.6819_427.4490_307.8527',
                           'batch0.Lipids_POS_587.5473_371.7741_270.3309',
                           'batch0.Lipids_POS_692.6319_447.7308_310.2593',
                           'batch0.Lipids_NEG_310.8379_48.6147_133.8410',
                           'batch0.Lipids_POS_644.6326_403.8752_292.8943',
                           'batch0.Lipids_NEG_720.6505_403.8259_300.8752',
                           'batch0.Lipids_POS_955.8045_433.6478_354.5816',
                           'batch0.Lipids_POS_722.7121_419.3648_314.3103',
                           'batch0.Lipids_NEG_1089.8691_431.3047_412.3494',
                           'batch0.Lipids_NEG_698.6211_419.3794_292.0184',
                           'batch0.Lipids_POS_953.7890_422.5644_352.0154',
                           'batch0.Lipids_NEG_984.8941_499.7480_364.6446',
                           'batch0.Lipids_POS_975.7739_407.4706_353.2466',
                           'batch0.Lipids_POS_927.7719_422.3191_347.1757')
  }
  ps <- list()
  i <- 1
  for (short_filename in operative_gwases){
    if (loader == "metabolomics"){
      long_filename <- paste0(short_filename, ".glm.linear")
    }
    else{
      ##for these loaders, the name is complete as is is
      long_filename <- short_filename
    }
      gwas <- read.csv(paste0(basepath, long_filename), sep="\t")
      if (length(gwas$P) != 0){
          p <- ggplot_manhattan(gwas, theTitle = stringr::str_replace(short_filename, pattern = "batch.0", replacement = ""))
          ps[[i]] <- p
          i <- i + 1
      }
  }
  ##do this here
  jpeg(stringr::str_replace_all(paste0(figpath_panels, loader, ".jpg"), "%", ""), width = 1920, height = 3000)
  cowplot::plot_grid(plotlist = ps)
  dev.off()
}


