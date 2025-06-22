import pickle, json, os
import numpy as np
import math
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from bioservices import UniProt
from tqdm import tqdm
import requests

import ipdb; ipdb.set_trace()



# Function to save a dictionary to a text file
def save_dict_to_txt(dictionary, file_path):
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, indent=4)  # Save as JSON formatted text

# Function to load a dictionary from a text file
def load_dict_from_txt(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)



def clustal_msa(fasta_string, file_path):
    """
    Perform multiple sequence alignment using the Clustal Omega web service.

    Parameters:
        fasta_string (str): A string in FASTA format containing the sequences to align.

    Returns:
        str: The aligned sequences in FASTA format.
    """
    import ipdb; ipdb.set_trace()
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File '{file_path}' exists. Loading its content.")
        with open(file_path, "r") as fasta_MSA:
            return fasta_MSA.read()

    # Clustal Omega web service endpoint
    url = "https://www.ebi.ac.uk/Tools/services/rest/clustalo/run/"

    # Parameters for the Clustal Omega service
    data = {
        "sequence": fasta_string,
        "email": "deltadedirac@gmail.com",  # Replace with your email if required
        "format": "fasta"
    }

    print("Submitting sequences for MSA...")
    response = requests.post(url, data=data)

    if response.status_code != 200:
        raise Exception(f"Error submitting sequences to Clustal Omega: {response.text}")

    job_id = response.text.strip()
    print(f"Job submitted successfully. Job ID: {job_id}")

    # Check job status
    status_url = f"https://www.ebi.ac.uk/Tools/services/rest/clustalo/status/{job_id}"

    while True:
        status_response = requests.get(status_url)
        if status_response.status_code != 200:
            raise Exception(f"Error checking job status: {status_response.text}")

        status = status_response.text.strip()
        if status == "FINISHED":
            print("Alignment completed.")
            break
        elif status in ["RUNNING", "PENDING"]:
            print("Job is still running...")
        else:
            raise Exception(f"Unexpected job status: {status}")

    # Retrieve results
    #result_url = f"https://www.ebi.ac.uk/Tools/services/rest/clustalo/result/{job_id}/aln-clustal_num"
    result_url = f"https://www.ebi.ac.uk/Tools/services/rest/clustalo/result/{job_id}/fa"
    result_response = requests.get(result_url)

    if result_response.status_code != 200:
        raise Exception(f"Error retrieving alignment results: {result_response.text}")

    with open(file_path, "w") as fasta_file:
        fasta_file.write(result_response.text)
    
    print(f"File '{file_path}' created successfully.")
    return result_response.text


def get_substring(input_string, start_pos, end_pos=None):
    """
    Extracts a substring from a string.

    Parameters:
    - input_string (str): The original string.
    - start_pos (int): The starting position (0-based index).
    - end_pos (int or None): The ending position (0-based index). If None, goes to the end of the string.

    Returns:
    - str: The extracted substring.
    """
    if end_pos is None:
        return input_string[start_pos:]
    return input_string[start_pos:end_pos]

import random

def split_protein_data(tags, domain_seqs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=None, shuffle_enable=False):
    """
    Splits protein data into training, validation, and test sets.
    
    Parameters:
    - tags (list): List of protein names (tags).
    - domain_seqs (list): List of corresponding protein sequences.
    - train_ratio (float): Proportion of data for the training set.
    - val_ratio (float): Proportion of data for the validation set.
    - test_ratio (float): Proportion of data for the test set.
    - seed (int or None): Random seed for reproducibility.
    
    Returns:
    - dict: Dictionary containing 'train', 'val', and 'test' keys with corresponding tag-sequence pairs.
    """
    # Validate inputs
    if len(tags) != len(domain_seqs):
        raise ValueError("The length of tags and domain_seqs must be the same.")
    if not (0 < train_ratio + val_ratio + test_ratio <= 1):
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be between 0 and 1.")
    
    # Set random seed
    if seed is not None:
        random.seed(seed)
    
    # Combine tags and sequences
    data = list(zip(tags, domain_seqs))
    
    # Shuffle the data
    if shuffle_enable == True:
        random.shuffle(data)
    
    # Compute split indices
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Unzip data into separate lists
    train_tags, train_seqs = zip(*train_data) if train_data else ([], [])
    val_tags, val_seqs = zip(*val_data) if val_data else ([], [])
    test_tags, test_seqs = zip(*test_data) if test_data else ([], [])
    
    # Return splits as a dictionary
    return {
        "train": {"tags": list(train_tags), "domain_seqs": list(train_seqs)},
        "val": {"tags": list(val_tags), "domain_seqs": list(val_seqs)},
        "test": {"tags": list(test_tags), "domain_seqs": list(test_seqs)},
    }

def get_subset_from_fasta_str(fasta_str, seq_amount, ref_length, filter_bylength=None):
    import ipdb; ipdb.set_trace()
    tmp = fasta_str.split("\n")


    if filter_bylength is not None:
        # Get the length of the first sequence
        first_sequence_length = ref_length
        filtered_list=[]
        count=0
        # Use list comprehension to filter the sequences efficiently
        for i in range(0, len(tmp)-2, 2):
            #print('iteration {}'.format(i))
            #if i==63014:
            #    print('from here'); ipdb.set_trace()
            tag = tmp[i]
            sequence = tmp[i + 1]
            if len(sequence) <= first_sequence_length:
                if count >= seq_amount:
                    break
                filtered_list.extend([tag, sequence])
                count+=1
        tmp = filtered_list

    ipdb.set_trace()
    
    subset = '\n'.join(tmp[0:2*seq_amount])
    return subset

def download_sequences_per_batch(set_idx, batch_size=10):

    print( "starting sequences downloading......" )

    import pandas as pd
    import ipdb; ipdb.set_trace()

    # Remove repetitions
    idx_ids = list(set(set_idx))
    df_seqs = pd.DataFrame({'Entry': pd.Series(dtype='object'),
                        'Sequence': pd.Series(dtype='object')})

    for i, batch in enumerate(np.array_split(idx_ids, math.ceil(len(idx_ids)/batch_size))):

        print('iteration {} \n\n'.format(i))
        u = UniProt(verbose=True)
        df = u.get_df(batch, limit=None)
        #print(len(batch), len(batch))
            # print(list(df.columns))
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        result = df[['Entry', 'Sequence']]

        #ipdb.set_trace()
        df_seqs = pd.concat([df_seqs, result])

        #pd.concat([df1, df2])
    result_dict = df_seqs.set_index('Entry')['Sequence'].to_dict()
    save_dict_to_txt(result_dict, 'YAP1_seqs_dict.txt')
    print('DoNE')
    return result_dict


def create_or_load_fasta(names, sequences, file_path):
    """
    Create a FASTA file from two lists or load the existing file if it exists.
    
    Parameters:
    - names (list of str): List of sequence names (tags).
    - sequences (list of str): List of sequences corresponding to the names.
    - file_path (str): Path to save or load the FASTA file.
    
    Returns:
    - str: The FASTA content as a string.
    
    Raises:
    - ValueError: If the lengths of names and sequences do not match.
    """
    # Check if the file already exists
    if os.path.exists(file_path):
        print(f"File '{file_path}' exists. Loading its content.")
        with open(file_path, "r") as fasta_file:
            return fasta_file.read()

    # Validate input lengths
    if len(names) != len(sequences):
        raise ValueError("The lengths of 'names' and 'sequences' must match.")
    
    # Create the FASTA file and save the content
    fasta_content = ""
    for name, sequence in zip(names, sequences):
        fasta_content += f">{name}\n{sequence}\n"
    
    with open(file_path, "w") as fasta_file:
        fasta_file.write(fasta_content)
    
    print(f"File '{file_path}' created successfully.")
    return fasta_content


def equalize_fasta_sequence_lengths(fasta_str, ref_max_length=None, gap_char="?"):
    """
    Adjusts the lengths of sequences in a FASTA string by padding shorter sequences with a gap character.

    Parameters:
    - fasta_str (str): A string containing the content of a FASTA file.
    - gap_char (str): The character to use for padding shorter sequences.

    Returns:
    - str: A new FASTA string with all sequences padded to the same length.
    """
    import ipdb; ipdb.set_trace()
    def parse_fasta(fasta_str):
        """Parses a FASTA string into a list of (header, sequence) tuples."""
        fasta_lines = fasta_str.strip().splitlines()
        sequences = []
        header = None
        seq = []
        
        for line in fasta_lines:
            if line.startswith(">"):
                if header:  # Save the previous sequence
                    sequences.append((header, "".join(seq)))
                header = line.strip()
                seq = []
            else:
                seq.append(line.strip())
        if header:  # Save the last sequence
            sequences.append((header, "".join(seq)))
        
        return sequences

    def format_fasta(sequences):
        """Formats a list of (header, sequence) tuples into a FASTA string."""
        return "\n".join(f"{header}\n{seq}" for header, seq in sequences)

    # Parse the input FASTA string
    parsed_data = parse_fasta(fasta_str)

    # Extract only the sequences
    headers, sequences = zip(*parsed_data)
    
    # Find the maximum sequence length
    if ref_max_length is not None:
        max_length=ref_max_length
    else:
        max_length = max(len(seq) for seq in sequences)
    
    # Pad all sequences to the maximum length using the gap character
    padded_sequences = [seq.ljust(max_length, gap_char) for seq in sequences]
    
    # Combine the headers and padded sequences
    equalized_fasta = list(zip(headers, padded_sequences))
    
    # Format back into a FASTA string
    return format_fasta(equalized_fasta)


def preprocess_domain_def(hash_prots, ids, domain_def):

    import ipdb; ipdb.set_trace()

    dom_names=[]; seq_domains=[]
    id_domain_pair = list(zip(ids,domain_def))
    for set_seqinfo in tqdm( id_domain_pair, desc="Processing sequences..."):
        
        if set_seqinfo[0] not in hash_prots:                
            continue 

        #if set_seqinfo[0]=='A0A0P5R957':
        #if set_seqinfo[0]=='G1N2I2':
        #    ipdb.set_trace()
        #    print('from here')

        #print(set_seqinfo[0])    
        tag = set_seqinfo[0]; dom_def = set_seqinfo[1]
        if isinstance(hash_prots[tag],float): continue # when dictionary get NaN, see why is happening that, is it for bioservices request?

        seq = hash_prots[tag]
        dom_seq = get_substring(seq, dom_def[0]-1, end_pos=dom_def[1])

        # clean up the domain definitions that are not existing 
        # on Uniprot sequence, or going beyond the sequence length
        if len(dom_seq)==0:
            continue

        dom_names.append( tag+'/'+'-'.join( list(map(str, dom_def)) ) )
        seq_domains.append( dom_seq )

    ipdb.set_trace()
    dict_sets = split_protein_data(dom_names, seq_domains, train_ratio=0.5, val_ratio=0.3, 
                                                test_ratio=0.2, seed=None, shuffle_enable=False)
    
    '''ADD CODE TO SAVE THE SPLITTED PARTITIONS'''
    
    #return {
    #    "train": {"tags": list(train_tags), "domain_seqs": list(train_seqs)},
    #    "val": {"tags": list(val_tags), "domain_seqs": list(val_seqs)},
    #    "test": {"tags": list(test_tags), "domain_seqs": list(test_seqs)},
    #}
    YAP1_fasta_str_TRAIN = create_or_load_fasta( dict_sets['train']['tags'], dict_sets['train']['domain_seqs'], 'YAP1_filtered_train.fasta' )
    YAP1_fasta_str_TEST = create_or_load_fasta( dict_sets['test']['tags'], dict_sets['test']['domain_seqs'], 'YAP1_filtered_test.fasta' )
    YAP1_fasta_str_VAL = create_or_load_fasta( dict_sets['val']['tags'], dict_sets['val']['domain_seqs'], 'YAP1_filtered_val.fasta' )

    YAP1_fasta_str_full = create_or_load_fasta(dom_names, seq_domains, 'YAP1_filtered_FULLsequences.fasta')

    return YAP1_fasta_str_full, YAP1_fasta_str_TRAIN, YAP1_fasta_str_TEST, YAP1_fasta_str_VAL 




if __name__ == "__main__":

    lexem='UniRef100';ids = [];domain_map=[]; wt_seq='';wt_tag=''; info_set =[]
    for record in SeqIO.parse('alignments/YAP1_HUMAN_1_b0.5.a2m', 'fasta'):

        if record.id.split('/')[0] == 'YAP1_HUMAN':
            wt_seq = str(record.seq); wt_tag = 'YAP1_HUMAN'

        idx = record.id.split('/')[0].split('_')[1]
        domain_def = list(map(int, 
                        record.id.split('/')[1].split('-') ))
        
        ids.append(idx)
        domain_map.append( domain_def)
        info_set.append([idx, domain_def])

    # skip first one
    wt_ids = ids[0]; wt_map = domain_map[0]
    ids = ids[1:]; domain_map = domain_map[1:]
    #print(ids)

    info_map = {}
    fasta_entries = []

    import ipdb; ipdb.set_trace()
    if os.path.exists('YAP1_seqs_dict.txt'):
        YAP1_dict = load_dict_from_txt('YAP1_seqs_dict.txt')
    else:
        YAP1_dict = download_sequences_per_batch(ids, batch_size=100)

    YAP1_fasta_str_full, YAP1_fasta_str_TRAIN, \
            YAP1_fasta_str_TEST, YAP1_fasta_str_VAL  = preprocess_domain_def(YAP1_dict, ids, domain_map)

    ipdb.set_trace()
    # take the sequence filtered by the length of wild type, i.e. lower or equal length as wt
    YAP1_subset_tmp = get_subset_from_fasta_str(YAP1_fasta_str_full, 1999, ref_length =len(wt_seq), filter_bylength=True)
    
    # take the sequence without filtering fot infering transformations under very low similarity patterns,
    # i.e. domains of WW with long loops and so on.
    YAP1_subset_density = get_subset_from_fasta_str(YAP1_fasta_str_VAL, 600, ref_length =len(wt_seq))

    # Adding wild type for testing
    YAP1_subset_tmp = f">{wt_tag}\n{wt_seq.upper()}\n" + YAP1_subset_tmp
    #MSA_YAP1_set_prior = clustal_msa(YAP1_subset_tmp, 'YAP1_MSA500.a2m')
    MSA_YAP1_set_prior = clustal_msa(YAP1_subset_tmp, 'YAP1_MSA2000.a2m')

    import ipdb; ipdb.set_trace()
    #ref_max_length = len(MSA_YAP1_set_prior.split("\n")[1])
    ref_max_length = len( ''.join(MSA_YAP1_set_prior.split('>')[1].split('\n')[1:]) ) # take the max length of wild type including gaps
    ipdb.set_trace()
    raw_seqs_for_T = equalize_fasta_sequence_lengths(YAP1_subset_density, ref_max_length=ref_max_length, gap_char="?")
    with open('YAP1_density_to_train_padded600.fasta', "w") as fasta_file:
        fasta_file.write(raw_seqs_for_T)
    


