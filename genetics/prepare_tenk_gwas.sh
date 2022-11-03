#!/bin/csh

cd /net/mraid08/export/jasmine/zach/height_gwas/all_gwas/ldsc/ldsc-master
conda activate ldsc
ldsc-master/munge_sumstats.py --sumstats $1 --out $2 --a2 AX --snp ID --N $3
conda deactivate