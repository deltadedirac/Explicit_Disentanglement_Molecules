#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:18:28 2018

@author: nsde
"""

#%%
import pdb
import torch, os
import argparse, datetime
#from torchvision import transforms
#pdb.set_trace()

from src.gp_cpab.src.transformation.gp_cpab import gp_cpab
from src.gp_cpab.src.transformation.configManager import configManager

from src.models.trainer import vae_trainer
from src.unsuper.unsuper.data.mnist_data_loader import mnist_data_loader
from src.unsuper.unsuper.data.perception_data_loader import perception_data_loader
from src.unsuper.unsuper.helper.utility import model_summary
from src.models.encoder_decoder import get_encoder, get_decoder
from src.models import get_model

from src.seqsDataLoader import seqsReader, seqsDatasetLoader, Sequence_Data_Loader

#%%
def argparser():
    """ Argument parser for the main script """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model settings
    ms = parser.add_argument_group('Model settings')
    ms.add_argument('--model', type=str, default='vitae_ci', help='model to train')
    ms.add_argument('--ed_type', type=str, default='mlp', help='encoder/decoder type')
    ms.add_argument('--stn_type', type=str, default='gp_cpab', help='transformation type to use')
    ms.add_argument('--beta', type=float, default=16.0, help='beta value for beta-vae model')
    
    # Training settings
    ts = parser.add_argument_group('Training settings')
    ts.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
    ts.add_argument('--eval_epoch', type=int, default=1000, help='when to evaluate log(p(x))')
    ts.add_argument('--batch_size', type=int, default=2, help='size of the batches') # batch=10,warmup=10, epochs=100
    ts.add_argument('--warmup', type=int, default=20, help='number of warmup epochs for kl-terms')
    ts.add_argument('--lr', type=float, default=1e-5, help='learning rate for adam optimizer')
    
    # Hyper settings
    hp = parser.add_argument_group('Variational settings')
    hp.add_argument('--latent_dim', type=int, default=20, help='dimensionality of the latent space')
    hp.add_argument('--density', type=str, default='gaussian', help='output density')  # bernoulli  
    hp.add_argument('--eq_samples', type=int, default=1, help='number of MC samples over the expectation over E_q(z|x)')
    hp.add_argument('--iw_samples', type=int, default=1, help='number of importance weighted samples')
    
    # Dataset settings
    ds = parser.add_argument_group('Dataset settings')
    ds.add_argument('--classes','--list', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='classes to train on')
    ds.add_argument('--num_points', type=int, default=10000, help='number of points in each class')
    ds.add_argument('--logdir', type=str, default='beta_final16_3', help='where to store results')
    ds.add_argument('--dataset', type=str, default='mnist', help='dataset to use')
    
    # Parse and return
    args = parser.parse_args()
    return args

#%%
if __name__ == '__main__':
    # Input arguments
    args = argparser()
    std = configManager("configs/setup2.yaml")

    # Logdir for results
    if args.logdir == '':
        logdir = 'res/' + args.model + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    else:
        logdir = 'res/' + args.model + '/' + args.logdir
    
    # Load data
    print('Loading data')
    #pdb.set_trace()

    batches = args.batch_size #16 # 448
    # Raw Sequences, to see if we can align the sequences somehow
    dataset_msa = seqsDatasetLoader( pathBLAT_data = std.parserinfo('*/PathMSAalignment') )
    # initially the dimension is [448, 34, 21]. However as it is necesary to ignore 
    # the batch size, I just create a tuple, by taking just the last 2 components from the size
    trainloader, testloader = Sequence_Data_Loader(dataset_msa, dataset_test=None, batch_size=batches)
    seq_size = ( [*dataset_msa.prot_space.shape][1:])
    #seq_size = (1,*[*dataset_msa.prot_space.shape][1:])

    pdb.set_trace()

    # Construct model
    model_class = get_model(args.model)
    model = model_class(input_shape = seq_size, #img_size,
                        config = std, 
                        latent_dim = args.latent_dim, 
                        encoder = get_encoder(args.ed_type), 
                        decoder = get_decoder(args.ed_type), 
                        outputdensity = args.density,
                        ST_type = args.stn_type,
                        alphabet_size = len(dataset_msa.c2i) )
    
    # Summary of model
    #model_summary(model)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model_name = '/trained_model_softmax.pt'

    if os.path.isfile(logdir + model_name):
        print ("Loading Deformation Model")
        model.load_state_dict(torch.load(logdir + model_name))
        testloader, _ = Sequence_Data_Loader(dataset_msa, dataset_test=None, batch_size=448)
        #testloader = next(iter(testloader))[0].to(torch.float32).to('cpu')
    else:
        # Train model
        Trainer = vae_trainer(seq_size, model, optimizer)
        Trainer.fit2(trainloader=trainloader, 
                    n_epochs=args.n_epochs, 
                    warmup=args.warmup, 
                    logdir=logdir,
                    testloader=None,#testloader,
                    eq_samples=args.eq_samples, 
                    iw_samples=args.iw_samples,
                    beta=args.beta,
                    eval_epoch=args.eval_epoch,
                    padded_idx = dataset_msa.padded_idx,
                    non_padded_idx =  dataset_msa.non_padded_idx )
        

        # Save model
        torch.save(model.state_dict(), logdir + model_name)

    model.eval()
    pdb.set_trace()
    recon_data_train = model(dataset_msa.prot_space, 1, 1, 1.)
    #recon_data_train = model(dataset_msa.prot_space.reshape(-1,1,31,20),1,1,1.)

    

    xmean = recon_data_train[0].squeeze(1)
    cpab1 = model.x_trans1

    cpabZ2 = model.xmean_before_reverse

    pdb.set_trace()

    print('Done')

