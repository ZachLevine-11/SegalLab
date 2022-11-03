#!/bin/csh

cd /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/ldsc-master
conda activate ldsc
ldsc-master/munge_sumstats.py --sumstats $1 --out $2 --snp variant --N 361194 --a1 ref_allele --a2 alt_allele
conda deactivate
