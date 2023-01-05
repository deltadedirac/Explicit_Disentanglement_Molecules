#!/bin/bash

#NEWLINE=$'\n'
#result=$(grep -Eo 'RSA\w{2}' BLAT/alignments/BLAT_ECOLX_1_b0.5.a2m | head -n 20)

grep -Eo 'RSA\w{7}' BLAT/alignments/BLAT_ECOLX_1_b0.5.a2m | sort -u | head -n 20 > tmpseqs.txt
awk '{print "> seq" FNR ORS $0}' tmpseqs.txt > BLAT20.fasta
