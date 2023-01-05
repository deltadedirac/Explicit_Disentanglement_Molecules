#!/bin/bash

grep -v '#' PF00397.alignment.seed | awk '{print ">"$1; print $2}' > WW_domain_dataset_seed.fasta
