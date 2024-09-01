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
import itertools
from collections import *

def Sequence_Data_Loader(dataset_train, dataset_test=None, batch_size=16, sampler=None):

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, sampler=sampler)
    return trainloader, testloader


class seqsReader():

    @staticmethod
    def _predefine_encoding(alphabet):
            char_to_int = dict((c, i) for i, c in enumerate(alphabet))
            int_to_char = dict((str(i), c) for i, c in enumerate(alphabet))
            int_to_int = dict((str(i), i) for i, c in enumerate(alphabet))
            return char_to_int, int_to_char, int_to_int
    
    @staticmethod
    def onehot_by_chunks(t, num_classes, padded_vals):

        amino_idx = (t != padded_vals).nonzero(as_tuple=True)[0]
        padding_idx = (t == padded_vals).nonzero(as_tuple=True)[0]
        one_hot_chunks = torch.cat( ( F.one_hot(t[amino_idx], num_classes), 
                                        torch.ones(len(padding_idx), num_classes)*(1/num_classes)  ) )
                
 
        return padding_idx, one_hot_chunks

    @staticmethod
    def onehot_by_chunk_no_Ud(t, num_classes, padded_vals):
        #import ipdb; ipdb.set_trace()
        amino_idx = (t != padded_vals).nonzero(as_tuple=True)[0]
        padding_idx = (t == padded_vals).nonzero(as_tuple=True)[0]
        one_hot_chunks = torch.cat( ( F.one_hot(t[amino_idx], num_classes), 
                                        torch.ones(len(padding_idx), num_classes)  ) )
                
 
        return padding_idx, one_hot_chunks

    @staticmethod
    def padding_strategy(list_sequences, value, path_strategy='constant'):
        import math
        max_l_seq = max([ i.shape[1] for i in list_sequences])
        padded_seqs = []
        for tt in list_sequences:

            len_seq = math.abs(tt.shape[1] - max_l_seq)
            pad = ( (np.floor(len_seq/2) , np.round(len_seq/2)), \
                    (0,len_seq)  )[ path_strategy=='constant' ]

            padded_seqs.append(F.pad(tt, pad, path_strategy, value))

        return padded_seqs



    @staticmethod
    def define_dataset_variablelength_seqs(path, **kargs):
            # Read your fasta file
            identifiers = []
            lengths = []
            seqs = []
            seqs_numeric = []
            
            if 'alphabet'  in kargs:
                alphabet = kargs['alphabet']
            else:
                alphabet = np.unique(seqs_np).tolist()
            num_classes = len(alphabet)#np.unique(seqs_np).shape[0]
            # To create a dictionary of correspondences between the aminoacids and the numerical order.
            c2i, i2c, i2i = seqsReader._predefine_encoding(alphabet)
            if kargs['padding']!=None:
                padded_val = c2i[kargs['padding']]
            else: 
                padded_val = None

            #et_trace()
            # Read of file source to extract seq

            seq_names = []
            dict_seqs=defaultdict(str)
            filtered_dict={}
            INPUT = open(path, "r")
            for i, line in enumerate(INPUT):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    seq_names.append(name)
                else:
                    dict_seqs[name] += line.upper().replace('.','-')
            INPUT.close()

            # Convert to list based on the order of keys
            #import ipdb; ipdb.set_trace()
            seqs = [ np.array(list(dict_seqs[key])) for key in seq_names]  # Extract the values in sorted order
            print('Ok')

            # To obtain the number of clases, as well as the alphabet and the specific numpy array of numpy arrays right into it.
            seqs = np.array(seqs,dtype=object)
            seqs_np = np.concatenate(np.array(seqs))
            #pdb.set_trace()

            family_seqs = [ torch.from_numpy(np.array([ c2i[elem] for elem in seq ])) for seq in seqs ]

            if padded_val!=None:
                family_seqs = torch.nn.utils.rnn.pad_sequence(family_seqs, batch_first=True, padding_value = padded_val)
            else:
                family_seqs = torch.stack(family_seqs)

            prot_space = [ F.one_hot(family_seqs[i], len(alphabet) ) for i in range(0,family_seqs.shape[0]) ]
    
            prot_space = torch.stack(prot_space, dim=0)
            return prot_space, identifiers, lengths, num_classes, alphabet, c2i, i2c, i2i, None, None


    @staticmethod
    def seq2num(seqs,c2i, i2c, i2i):
        return [ torch.from_numpy(np.array([ c2i[elem] for elem in seq ])) for seq in seqs ]


    @staticmethod
    def read_clustal_align_output(path, **kargs):
        #pdb.set_trace()
        msa = []

        align = AlignIO.read(path, "clustal")
        for sequence in align:
            msa.append(np.array(list(sequence.seq)))

        if 'alphabet'  in kargs:
            alphabet = kargs['alphabet']
        else:
            alphabet = np.unique(msa).tolist()

        # To create a dictionary of correspondences between the aminoacids and the numerical order.
        c2i, i2c, i2i = seqsReader._predefine_encoding(alphabet)

        msa_tensor_numeric = torch.stack(seqsReader.seq2num(msa,c2i, i2c, i2i))
        msa_tensor_onehot = torch.stack([ F.one_hot(msa_tensor_numeric[i], len(alphabet)) for i in range(0,msa_tensor_numeric.shape[0]) ])
        return msa_tensor_numeric, msa_tensor_onehot, alphabet, c2i, i2c, i2i, msa




class seqsDatasetLoader(torch.utils.data.Dataset):


    def __init__(self, **kwargs):

        #import ipdb; ipdb.set_trace()
        if 'padding' in kwargs:
            padd = kwargs['padding']
        else:
            padd = None
        self.prot_space, self.identifiers, self.lengths, self.num_classes, \
            self.alphabet, self.c2i, self.i2c, self.i2i,self.padded_idx, \
            self.non_padded_idx = \
                    seqsReader.define_dataset_variablelength_seqs(\
                        kwargs.get("pathBLAT_data"), alphabet = kwargs.get("alphabet"), \
                          padding=padd )

        if 'device' in kwargs:
            device = kwargs['device']
            if type(device)==str:
                device = torch.device("cuda") if device=="gpu" or device=="cuda" else torch.device("cpu") if device=='cpu' else torch.device('mps')
        self.prot_space = torch.tensor(self.prot_space, dtype=torch.float32, device=device)

    def is_num_nparray(self,a):
        flag=True
        try:
            a.astype(int)
        except:
            flag=False
        return flag
    
    def get_paddings_per_batch(self, n_batch, batch_size, offset = 0):
        return self.padded_idx[n_batch*batch_size : (n_batch + 1)* batch_size - offset], \
               self.non_padded_idx[n_batch*batch_size : (n_batch + 1)* batch_size - offset]

    def __len__(self):
        return self.prot_space.shape[0]  # required

    def __getitem__(self, idx):
        '''DONT FORGET TO PUT THE PADDING INDEXES IN THE BATCH MODE, IT IS SUPERIMPORTANT'''
        return self.prot_space[idx]
