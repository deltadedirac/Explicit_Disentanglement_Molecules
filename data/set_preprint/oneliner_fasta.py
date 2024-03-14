
import os, sys
from tqdm import tqdm


if __name__ == "__main__":
    
    # Check if the number of command-line arguments is correct
    if len(sys.argv) != 3:
        print("Usage: python script.py <input fasta> <output fasta>")
        sys.exit(1)
        
    input = sys.argv[1]#'BLAT_ECOLX_1_b0.5_labeled_noX.fasta'
    output = sys.argv[2]
    
    fasta = ''
    with open(input) as f:
        gg=f.readlines()
        
    for line in  tqdm(gg, desc="Processing"):
        if '>' in line and fasta=='': fasta+=line
        elif '>' in line and len(fasta)>0: fasta+='\n'+line
        else: fasta+=line.rstrip('\n')
        
    with open(output, "w") as file:
        # Write the string to the file
        file.write(fasta)
