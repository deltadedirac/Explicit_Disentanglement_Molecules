#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:18:45 2018

@author: nsde
"""
#%%
import torch
import numpy as np
from torchvision.utils import make_grid
from tqdm import tqdm
import time, os, datetime, gc, pdb

from tensorboardX import SummaryWriter
from ..unsuper.unsuper.helper.losses import vae_loss
from .LossFunctionsAlternatives import LossFunctionsAlternatives


#%%
class vae_trainer:
    """ Main class for training the vae models 
    Arguments:
        input_shape: shape of a single image
        model: model (of type torch.nn.Module) to train
        optimizer: optimizer (of type torch.optim.Optimizer) that will be used 
            for the training
    Methods:
        fit - for training the network
        save_embeddings - embeds data into the learned spaces, saves to tensorboard
    """
    def __init__(self, input_shape, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.outputdensity = model.outputdensity
        self.use_cuda = True
        
        # Get the device
        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        # Move model to gpu (if avaible)
        if torch.cuda.is_available() and self.use_cuda:
            self.model.cuda()

        self.model.train()


    #%%
    def fit2(self, trainloader, n_epochs=10, warmup=1, logdir='',
            testloader=None, eq_samples=1, iw_samples=1, beta=1.0, eval_epoch=10000, **kargs):
        """ Fits the supplied model to a training set 
        Arguments:
            trainloader: dataloader (of type torch.utils.data.DataLoader) that
                contains the training data
            n_epochs: integer, number of epochs to run
            warmup: integer, the KL terms are weighted by epoch/warmup, so this
                number determines the number of epochs before the KL-terms are 
                fully activated in the loss function
            logdir: str, where to store the results
            testloader: dataloader (of type torch.utils.data.DataLoader) that
                contains the test data
            eq_samples: integer, number of equality samples which the expectation
                is calculated over
            iw_samples: integer, number of samples the mean-log is calculated over
            eval_epoch: how many epochs that should pass between calculating the
                L5000 loglikelihood (very expensive to do)
        """

        # Assert that input is okay
        assert isinstance(trainloader, torch.utils.data.DataLoader), '''Trainloader
            should be an instance of torch.utils.data.DataLoader '''
        assert warmup <= n_epochs, ''' Warmup period need to be smaller than the
            number of epochs '''
    
        # Print stats
        print('Number of training points: ', len(trainloader.dataset.prot_space))
        if testloader: print('Number of test points:     ', len(testloader.dataset))

        if 'ref' in kargs:
            ref = kargs['ref']
        else:
            ref = None

        if 'ref2' in kargs:
            ref2 = kargs['ref2']
        else:
            ref2 = None
        
        # Dir to log results
        logdir = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if logdir is None else logdir
        if not os.path.exists(logdir): os.makedirs(logdir)
        loss_function = LossFunctionsAlternatives()

        # Summary writer
        writer = SummaryWriter(log_dir=logdir)

        # Main loop
        start = time.time()
        for epoch in range(1, n_epochs+1):
            progress_bar = tqdm(desc='Epoch ' + str(epoch) + '/' + str(n_epochs), 
                                total=len(trainloader.dataset), unit='samples')
            train_loss = 0
            # Training loop
            #self.model.train()
            for i, data in enumerate(trainloader):
                # Zero gradient
                self.optimizer.zero_grad()
                #import pdb;pdb.set_trace()
                # Feed forward data
                data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)

                switch = 1.0 if epoch > warmup else 0.0
                out = self.model(data, eq_samples, iw_samples, switch)
                
                # Calculat loss
                #pdb.set_trace()
                #loss = loss_function(method = 'Guided_Soft_Label_KLD', input = out[0].squeeze(1), target = data, forw_per=(0,2,1), variational = out, target2=ref)
                loss = loss_function(method = 'Soft_Label_KLD', input = out[5], target = ref, forw_per=(0,2,1))

                #pdb.set_trace()
                #loss = loss_function(method = 'CE', input = out[0].squeeze(1), target = data, forw_per=(0,2,1)) #- out[7] - out[8]

                
                # Backpropegate and optimize
                # We need to maximize the bound, so in this case we need to
                # minimize the negative bound
                #(-loss).backward()
                loss.backward()
                self.optimizer.step()
                
                # Write to consol
                progress_bar.update(data.size(0))
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Save to tensorboard
                iteration = epoch*len(trainloader) + i
                writer.add_scalar('train/total_loss', loss, iteration)
                #writer.add_scalar('train/recon_loss', recon_term, iteration)
                
                #for j, kl_loss in enumerate(kl_terms):
                #    writer.add_scalar('train/KL_loss' + str(j), kl_loss, iteration)
                #del loss, recon_term, kl_loss, out
                #del loss, out


                gc.collect()
                torch.cuda.empty_cache()
            #progress_bar.set_postfix({'Average ELBO': train_loss / len(trainloader)})
            progress_bar.close()
        
        print('Total train time', time.time() - start)
        #import pdb;pdb.set_trace()
        # Save the embeddings
        print('Saving embeddings, maybe?')
        with torch.no_grad():
            try:
                self.save_embeddings(writer, trainloader, name='train')
            except Exception as e:
                print('Did not save embeddings for training set')
                print(e)
            if testloader: 
                try:
                    self.save_embeddings(writer, testloader, name='test')
                except Exception as e:
                    print('Did not save embeddings for test set')
                    print(e)

        # Close summary writer

        writer.close()

        
    #%%
    def save_embeddings(self, writer, loader, name='embedding'):
        # Constants
        N = len(loader.dataset)
        m = self.model.latent_spaces
        
        # Data structures for holding the embeddings
        all_data = torch.zeros(N, *self.input_shape, dtype=torch.float32)
        #all_label = torch.zeros(N, dtype=torch.int32)
        all_latent = [ ]
        for j in range(m):
            all_latent.append(torch.zeros(N, self.model.latent_dim, dtype=torch.float32))
        
        # Loop over all data and get embeddings
        counter = 0
        for i, data in enumerate(loader):
            n = data.shape[0]
            data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
            #label = label.to(self.device)
            z = self.model.latent_representation(data)
            all_data[counter:counter+n] = data.cpu()
            for j in range(m):
                all_latent[j][counter:counter+n] = z[j].cpu()
            #all_label[counter:counter+n] = label.cpu()
            counter += n
            
        # Save the embeddings
        for j in range(m):
            
            # Embeddings with dim < 3 needs to be appended extra non-informative dimensions
            N, n = all_latent[j].shape
            if n == 1:
                all_latent[j] = torch.cat([all_latent[j], torch.zeros(N, 2)], dim = 1)
            if n == 2:
                all_latent[j] = torch.cat([all_latent[j], torch.zeros(N, 1)], dim = 1)
            
            # Maximum bound for the sprite image
            writer.add_embedding(mat = all_latent[j],
                                 metadata = None,
                                 label_img = all_data,
                                 tag = name + '_latent_space' + str(j))
        
        
