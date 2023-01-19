#!/bin/bash

INPUT=$1
TARGET=$2

grep '^[a-zA-Z]' $INPUT | awk '{header=">"$1 ; seq=$2; gsub(/\./,"\-", seq); print header;print seq}' > $TARGET

python3 -c "
import sys
list_args=sys.argv

assert len(list_args)!=0, f'No allowed'
with open(list_args[1]) as f:
        for line in f:
                line=line.strip()
                if '>' in line: print(line); continue

                gapcat = ''.join( ['-']*line.count('-') )
                line = line.replace('-','')
                line += gapcat
                print(line)
" "$TARGET" > RAWseqs.fasta
