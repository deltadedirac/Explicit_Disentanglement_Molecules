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


def Sequence_Data_Loader(dataset_train, dataset_test=None, batch_size=16):

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)
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
        amino_idx = (t != padded_vals).nonzero(as_tuple=True)[0]
        padding_idx = (t == padded_vals).nonzero(as_tuple=True)[0]
        one_hot_chunks = torch.cat( ( F.one_hot(t[amino_idx], num_classes), 
                                        torch.ones(len(padding_idx), num_classes)  ) )
                
 
        return padding_idx, one_hot_chunks


    @staticmethod
    def define_dataset_variablelength_seqs(path, **kargs):
            # Read your fasta file
            identifiers = []
            lengths = []
            seqs = []
            seqs_numeric = []
            padded_val = 22

            with open(path) as fasta_file:  # Will close handle cleanly
                    for title, sequence in SimpleFastaParser(fasta_file):
                            identifiers.append(title.split(None, 1)[0])  # First word is ID
                            lengths.append(len(sequence))
                            seqs.append(np.array(list(sequence)))


            # To obtain the number of clases, as well as the alphabet and the specific numpy array of numpy arrays right into it.
            seqs = np.array(seqs,dtype=object)
            seqs_np = np.concatenate(np.array(seqs))
            #pdb.set_trace()
            if 'alphabet'  in kargs:
                alphabet = kargs['alphabet']
            else:
                alphabet = np.unique(seqs_np).tolist()
            num_classes = len(alphabet)#np.unique(seqs_np).shape[0]
            # To create a dictionary of correspondences between the aminoacids and the numerical order.
            c2i, i2c, i2i = seqsReader._predefine_encoding(alphabet)

            #family_seqs = np.array([ np.array([ self.c2i[elem] for elem in seq ]) for seq in seqs ] )
            family_seqs = [ torch.from_numpy(np.array([ c2i[elem] for elem in seq ])) for seq in seqs ]
            #pdb.set_trace()
            
            family_seqs = torch.nn.utils.rnn.pad_sequence(family_seqs, batch_first=True, padding_value = padded_val)

            #num_classes = np.unique(family_seqs).shape[0]
            #alphabet = np.unique(family_seqs).tolist()

            max_length = family_seqs.shape[1]
            #prot_space = [ F.one_hot(family_seqs[i], num_classes + 1) for i in range(0,family_seqs.shape[0]) ]
            #prot_space = [ seqsReader.onehot_by_chunks(family_seqs[i], num_classes, padded_val) for i in range(0,family_seqs.shape[0]) ]            
            prot_space = [ seqsReader.onehot_by_chunk_no_Ud(family_seqs[i], num_classes, padded_val) for i in range(0,family_seqs.shape[0]) ]            


            #pdb.set_trace()
            non_padded_idx = [] #[ list(range( 0, i[0].item()) ) if len(i[0])!=0 else list(range( 0, max_length) ) for i in prot_space ]
            padding_indexes = [] #[ list(set(range(0, max_length)) - set(non_padded_idx[i])) for i in range(0,len(non_padded_idx)) ]  #[ list( set(range(0, max_length)) - set(i) ) for i in padding_indexes ] 


            prot_space = [ i[1] for i in prot_space]
            #prot_space2= torch.cat(prot_space).view(family_seqs.shape[0],num_classes,-1)
            prot_space = torch.stack(prot_space, dim=0)
            return prot_space, identifiers, lengths, num_classes, alphabet, c2i, i2c, i2i, padding_indexes, non_padded_idx


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

        self.prot_space, self.identifiers, self.lengths, self.num_classes, \
            self.alphabet, self.c2i, self.i2c, self.i2i,self.padded_idx, self.non_padded_idx = seqsReader.define_dataset_variablelength_seqs(\
                                                                                                       kwargs.get("pathBLAT_data"), alphabet = kwargs.get("alphabet") )

        if 'device' in kwargs:
            device = kwargs['device']
            if type(device)==str:
                device = torch.device("cuda") if device=="gpu" else torch.device("cpu")
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
