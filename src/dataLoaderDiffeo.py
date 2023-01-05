import pdb
import torch
import pickle
import numpy as np
import pandas as pd
#from one_hot_encoding import *
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio import AlignIO

def _predefine_encoding(alphabet):
		char_to_int = dict((c, i) for i, c in enumerate(alphabet))
		int_to_char = dict((str(i), c) for i, c in enumerate(alphabet))
		int_to_int = dict((str(i), i) for i, c in enumerate(alphabet))
		return char_to_int, int_to_char, int_to_int

def onehot_by_chunks(t, num_classes, padded_vals):

        amino_idx = (t != padded_vals).nonzero(as_tuple=True)[0]
        padding_idx = (t == padded_vals).nonzero(as_tuple=True)[0]

        one_hot_chunks = torch.cat( ( F.one_hot(t[amino_idx], num_classes), 
                                        torch.zeros(len(padding_idx), num_classes)  ) )
 
        return padding_idx, one_hot_chunks

def define_dataset_variablelength_seqs(path, **kargs):#, alphabets):
		# Read your fasta file
		identifiers = []
		lengths = []
		seqs = []
		seqs_numeric = []

		with open(path) as fasta_file:  # Will close handle cleanly
				for title, sequence in SimpleFastaParser(fasta_file):
						identifiers.append(title.split(None, 1)[0])  # First word is ID
						lengths.append(len(sequence))
						seqs.append(np.array(list(sequence)))

		# To obtain the number of clases, as well as the alphabet and the specific numpy array of numpy arrays right into it.
		#pdb.set_trace()
		seqs = np.array(seqs)
		seqs_np = np.concatenate(np.array(seqs))

		#alphabet = np.unique(seqs_np).tolist()
		if 'alphabet' in kargs:
			alphabet = kargs['alphabet']
		else:
			alphabet = np.unique(seqs_np).tolist()

		num_classes = len(alphabet)#np.unique(seqs_np).shape[0]
		# To create a dictionary of correspondences between the aminoacids and the numerical order.
		c2i, i2c, i2i = _predefine_encoding(alphabet)

		#family_seqs = np.array([ np.array([ self.c2i[elem] for elem in seq ]) for seq in seqs ] )
		family_seqs = [ torch.from_numpy(np.array([ c2i[elem] for elem in seq ])) for seq in seqs ]
		#pdb.set_trace()
		
		if '-' in c2i and len(family_seqs)>1:
			family_seqs = torch.nn.utils.rnn.pad_sequence(family_seqs, batch_first=True, padding_value=c2i["-"])
			prot_space = [ onehot_by_chunks(family_seqs[i], num_classes, c2i["-"]) \
                                                            for i in range(0,family_seqs.shape[0]) ]  

			max_length = family_seqs.shape[1]                                     
			#padding_indexes = [ list(range( 0, i[0].item()) ) if len(i[0])!=0 else list(range( 0, max_length) ) \
            #                                                                                    for i in prot_space ]
			#non_padded_idx = [ list( set(range(0, max_length)) - set(i) ) for i in padding_indexes ] 
			padding_indexes = []; non_padded_idx =[]

		else:
			family_seqs =  torch.stack(family_seqs)
			prot_space = [ F.one_hot(family_seqs[i], num_classes) for i in range(0,family_seqs.shape[0]) ]			
			
			padding_indexes = [] #[ list(range( 0, i[0].item()) ) if len(i[0])!=0 else list(range( 0, max_length) ) for i in prot_space ]
			non_padded_idx = []#[ list( set(range(0, max_length)) - set(i) ) for i in padding_indexes ]
		#num_classes = np.unique(family_seqs).shape[0]
		#alphabet = np.unique(family_seqs).tolist()
		#prot_space = [ F.one_hot(family_seqs[i], num_classes) for i in range(0,family_seqs.shape[0]) ]
        
		prot_space = [ F.one_hot(family_seqs[i], num_classes) for i in range(0,family_seqs.shape[0]) ]	
		#prot_space2= torch.cat(prot_space).view(family_seqs.shape[0],num_classes,-1)
		prot_space2=torch.stack(prot_space, dim=0)
		return prot_space2, identifiers, lengths, num_classes, alphabet, padding_indexes, non_padded_idx

def seq2num(seqs,c2i, i2c, i2i):
    return [ torch.from_numpy(np.array([ c2i[elem] for elem in seq ])) for seq in seqs ]

def read_clustal_align_output(path, **kargs):
    #pdb.set_trace()
    msa = []

    align = AlignIO.read(path, "clustal")
    for sequence in align:
        msa.append(np.array(list(sequence.seq)))

    #msa = np.array(msa)
    if 'alphabet' in kargs and bool(kargs['alphabet']):
        alphabet = kargs['alphabet']
    else:
        alphabet = np.unique(msa).tolist()
    # To create a dictionary of correspondences between the aminoacids and the numerical order.
    c2i, i2c, i2i = _predefine_encoding(alphabet)

    msa_tensor_numeric = torch.stack(seq2num(msa,c2i, i2c, i2i))
    msa_tensor_onehot = torch.stack([ F.one_hot(msa_tensor_numeric[i], len(alphabet)) for i in range(0,msa_tensor_numeric.shape[0]) ])
    print("tmp")
    return msa_tensor_numeric, msa_tensor_onehot, alphabet, c2i, i2c, i2i, msa

class datasetLoader(torch.utils.data.Dataset):

    prot_space = []

    def __init__(self, **kwargs):
        #pdb.set_trace()
        enable_var_seqs = kwargs.get("enable_variable_length")
        alphabet = kwargs.get("alphabet")

        if enable_var_seqs == True:
            self.prot_space, self.identifiers, self.lengths, self.num_classes, \
                self.alphabet,self.padded_idx, self.non_padded_idx = define_dataset_variablelength_seqs(kwargs.get("pathBLAT_data"),alphabet=alphabet)
            #self.prot_space, self.identifiers, self.lengths, self.num_classes, self.alphabet = define_dataset_variablelength_seqs(kwargs.get("pathBLAT_data"))
            self.shape_dataset = self.prot_space.shape
        else:
            #self.seqs =  pickle.load(open(kwargs.pop("pathBLAT_data_MSA"),'rb'))
            self.alphabet = kwargs.get("alphabet")
            self.family_seqs, self.test_seqs, self.cell_grow_ampiciline = self.parse_BLAT(kwargs.get("pathBLAT_data"))
            self.dataset_extraction()

    def parse_BLAT(self, pathBLAT_data):
        with open(pathBLAT_data, "rb") as infile:
                blat_df = pickle.load(infile)

        # stored values without assay entries are BLAT TEM1 ECOLX family data
        family_df = blat_df[blat_df.assay.isna()]
        test_blat_df = blat_df[~blat_df.assay.isna()]
        # cast sequence labels to int
        family_seqs = np.array([[int(elem) for elem in seq] for seq in family_df.seqs])
        family_seqs = np.minimum(family_seqs, 21)
        test_seqs = np.array([[int(elem) for elem in seq] for seq in test_blat_df.seqs])
        test_y = np.array(test_blat_df.assay, dtype=float)
        return family_seqs, test_seqs, test_y


    def dataset_extraction(self):

        self.theta =0.2

        self.n, self.length = self.family_seqs.shape
        self.test_n = self.test_seqs.shape[0]
        self.num_classes = np.unique(self.family_seqs).shape[0]

        self.prot_space = F.one_hot(torch.from_numpy(self.family_seqs), num_classes=self.num_classes).to(float)

        self.shape_dataset = self.prot_space.shape

        self.validation_space = F.one_hot(torch.from_numpy(self.test_seqs),num_classes=self.num_classes).to(float)
        self.shape_val_dataset = self.validation_space.shape


    def is_num_nparray(self,a):
        flag=True
        try:
            a.astype(int)
        except:
            flag=False
        return flag

    def __len__(self):
        return self.prot_space.shape[0]  # required

    def __getitem__(self, idx):
        return self.prot_space[idx]


