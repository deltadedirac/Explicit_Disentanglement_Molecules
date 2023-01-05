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
import logging


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
    def __init__(self, input_shape, model, optimizer, log_dir='../logs'):
        self.model = model
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.outputdensity = model.outputdensity
        self.use_cuda = True
        self.logger = self.setup_LOGGER(log_dir)
        
        # Get the device
        if torch.cuda.is_available() and self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        # Move model to gpu (if avaible)
        if torch.cuda.is_available() and self.use_cuda:
            self.model.cuda()

        self.model.train()

    def setup_LOGGER(self, filename_path):
        import pdb; pdb.set_trace()
        LOG_FORMAT = " %(asctime) - %(message)"
        logging.basicConfig(filename = filename_path+"/log_test.log",
                            level = logging.DEBUG,
                            #format = LOG_FORMAT,
                            filemode='w')
        return logging.getLogger()

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
        pdb.set_trace()
        #masked_idx = trainloader.dataset.padded_idx
        #masked_idx = np.delete(np.arange(trainloader.dataset.prot_space.shape[1]),masked_idx).tolist()
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
                ''' 
                TO ORGANIZE LATER, THE SETUP OF PADDINGS IN THE TRANSFORMATION MUST NOT 
                DEPEND ON THE BATCH PER LOOP, YOU SHOULD MAKE IT WITH A WRAPPER/INTERFACE
                '''
                #pdb.set_trace()
                padded_idx_batch, non_padded_idx_batch = trainloader.dataset.get_paddings_per_batch( i, trainloader.batch_size, 
                                                                        offset = trainloader.batch_size - data.shape[0])

                self.model.stn.setup_ST_GP_CPAB(x = data, padding_option = 'padding_weights', 
                                                padded_idx = padded_idx_batch, non_padded_idx = non_padded_idx_batch)
                #self.model.stn.setup_ST_GP_CPAB(x = data)
                #data = data.to(torch.float32).to(self.device)
                switch = 1.0 if epoch > warmup else 0.0
                out = self.model(data, eq_samples, iw_samples, switch)
                
                # Calculat loss
                #pdb.set_trace()
                # SHITY EXAMPLE
                #loss = loss_function(method = 'CustomVariational_1', input = out[0].squeeze(1), target = data, forw_per=(0,2,1), variational = out, target2=ref)
                #pdb.set_trace()
                loss = loss_function(method = 'JSD', input = out[0].squeeze(1), target = data, forw_per=(0,2,1))

                
                #loss = loss_function(method = 'Soft_Label_KLD', input = out[0].squeeze(1), target = ref, forw_per=(0,2,1))
                #loss = loss_function(method = 'CE', input = out[0].squeeze(1), target = ref, forw_per=(0,2,1))
                #loss = loss_function(method = 'CE', input = out[0].squeeze(1), target = data, forw_per=(0,2,1))

                '''
                self.logger.info("EPOCH {0} :\n\n".format(epoch))
                self.logger.info("LOSS VALUE {0}".format(loss))
                self.logger.info("LOSS BY COMPONENTS: ")
                #self.logger.info(loss_function.component_vals)
                self.logger.info(loss_function.component_vals1)
                self.logger.info(loss_function.component_vals2)
                seq_length = data.shape[1]
                self.logger.info("ORIGINAL INPUT: \n {0}".format(data))
                orig_grid = self.model.stn.st_gp_cpab.make_grids_for_Regresion(batch_size = data.shape[0])
                inv_trans = self.model.stn.st_gp_cpab.transform_grid( orig_grid, -out[6])
                dir_trans = self.model.stn.st_gp_cpab.transform_grid( orig_grid, out[6])
                self.logger.info("ORIGINAL GRID: \n {0}".format(orig_grid))
                self.logger.info("ORIGINAL GRID SCALED: \n {0}".format(orig_grid*(seq_length-1) ))
                self.logger.info("INVERSE TRANSFORMED GRID: \n {0}".format(inv_trans))
                self.logger.info("INVERSE TRANSFORMED GRID SCALED: \n {0}".format(inv_trans*(seq_length-1)))
                self.logger.info("DIRECT TRANSFORMED GRID: \n {0}".format(dir_trans))
                self.logger.info("DIRECT TRANSFORMED GRID SCALED: \n {0}".format(dir_trans*(seq_length-1)) )
                self.logger.info("TRANSFORMED IMPUT FROM CPAB: \n {0}".format(out[5].squeeze(1)))
                self.logger.info("TARGET: \n {0}".format(ref))
                self.logger.info("FINAL OUTPUT FROM PGM: \n {0}".format(out[0].squeeze(1)))
                self.logger.info("--------------------------------------------------------------------------------\n\n" )
                '''


                #pdb.set_trace()
                '''
                loss = loss_function(method = 'CustomVariational_2', input = out[0].squeeze(1), target = ref,  forw_per=(0,2,1),
                                                                                                        target2=ref2, input2 = out[6] )
                '''
                # ---------------------------------------------------------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------------------------------
                #loss = loss_function(method = 'KL', input = out[6], target = ref2, forw_per=(0,2,1))



                # ---------------------------------------------------------------------------------------------------------------------
                # ---------------------------------------------------------------------------------------------------------------------


                '''
                loss, recon_term, kl_terms = vae_loss(data, *out, 
                                                      eq_samples, iw_samples, 
                                                      self.model.latent_dim, 
                                                      epoch, warmup, beta,
                                                      self.outputdensity)
                '''
                
                '''train_loss += float(loss.item())'''
                
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
        self.logger.info("""NOTE: remember, both inverse and direct transformation are made over the same original grid, not the
                normal configuration T(deformed_grid), because we are emulating the behavior of not having the historical information about
                the deformed grid (normaly when we transformed the new data, this is not possible). That's why we are using as a solution 
                just to make the interpolation strategy in inverse transforme just using the GP and for direct transform just using the linear
                interpolation  """)
        writer.close()




    #%%
    def fit(self, trainloader, n_epochs=10, warmup=1, logdir='',
            testloader=None, eq_samples=1, iw_samples=1, beta=1.0, eval_epoch=10000):
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
        
        # Dir to log results
        logdir = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if logdir is None else logdir
        if not os.path.exists(logdir): os.makedirs(logdir)
        
        # Summary writer
        writer = SummaryWriter(log_dir=logdir)
        
        # Main loop
        start = time.time()
        for epoch in range(1, n_epochs+1):
            progress_bar = tqdm(desc='Epoch ' + str(epoch) + '/' + str(n_epochs), 
                                total=len(trainloader.dataset), unit='samples')
            train_loss = 0
            # Training loop
            self.model.train()
            for i, data in enumerate(trainloader):
                # Zero gradient
                self.optimizer.zero_grad()
                #import pdb;pdb.set_trace()
                # Feed forward data
                data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
                #data = data.to(torch.float32).to(self.device)
                switch = 1.0 if epoch > warmup else 0.0
                out = self.model(data, eq_samples, iw_samples, switch)
                
                # Calculat loss
                loss, recon_term, kl_terms = vae_loss(data, *out, 
                                                      eq_samples, iw_samples, 
                                                      self.model.latent_dim, 
                                                      epoch, warmup, beta,
                                                      self.outputdensity)
                train_loss += float(loss.item())
                
                # Backpropegate and optimize
                # We need to maximize the bound, so in this case we need to
                # minimize the negative bound
                (-loss).backward()
                self.optimizer.step()
                
                # Write to consol
                progress_bar.update(data.size(0))
                progress_bar.set_postfix({'loss': loss.item()})
                
                # Save to tensorboard
                iteration = epoch*len(trainloader) + i
                writer.add_scalar('train/total_loss', loss, iteration)
                writer.add_scalar('train/recon_loss', recon_term, iteration)
                
                for j, kl_loss in enumerate(kl_terms):
                    writer.add_scalar('train/KL_loss' + str(j), kl_loss, iteration)
                del loss, recon_term, kl_loss, out
                

                gc.collect()
                torch.cuda.empty_cache()
            progress_bar.set_postfix({'Average ELBO': train_loss / len(trainloader)})
            progress_bar.close()
            
            # Log for the training set
            #import pdb;pdb.set_trace()
            '''
            with torch.no_grad():
                n = 10
                data_train = next(iter(trainloader))[0].to(torch.float32).to(self.device)
                data_train = data[:n].reshape(-1, *self.input_shape)
                recon_data_train = self.model(data_train)[0]
                writer.add_image('train/recon', make_grid(torch.cat([data_train, 
                             recon_data_train]).cpu(), nrow=n), global_step=epoch)
                samples = self.model.sample(n*n)    
                writer.add_image('samples/samples', make_grid(samples.cpu(), nrow=n), 
                                 global_step=epoch)
                del data_train, recon_data_train, samples
            '''
            if testloader:
                with torch.no_grad():
                    # Evaluate on test set (L1 log like)
                    self.model.eval()
                    test_loss, test_recon, test_kl = 0, 0, len(kl_terms)*[0]
                    for i, (data, _) in enumerate(testloader):
                        data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
                        out = self.model(data, 1, 1)    
                        loss, recon_term, kl_terms = vae_loss(data, *out, 1, 1, 
                                                              self.model.latent_dim, 
                                                              epoch, warmup, beta,
                                                              self.outputdensity)
                        test_loss += loss.item()
                        test_recon += recon_term.item()
                        test_kl = [l1+l2 for l1,l2 in zip(kl_terms, test_kl)]
            
                    writer.add_scalar('test/total_loss', test_loss, iteration)
                    writer.add_scalar('test/recon_loss', recon_term, iteration)
                    for j, kl_loss in enumerate(kl_terms):
                        writer.add_scalar('test/KL_loss' + str(j), kl_loss, iteration)
            
                    data_test = next(iter(testloader))[0].to(torch.float32).to(self.device)[:n]
                    data_test = data_test.reshape(-1, *self.input_shape)
                    recon_data_test = self.model(data_test)[0]
                    writer.add_image('test/recon', make_grid(torch.cat([data_test, 
                             recon_data_test]).cpu(), nrow=n), global_step=epoch)
                    if (epoch==n_epochs):
                        print('Final test loss', test_loss)
                    del data, out, loss, recon_term, kl_terms, data_test, recon_data_test

                    # Callback, if a model have something special to log
                    self.model.callback(writer, testloader, epoch)
                    
                    # If testset and we are at a eval epoch (or last epoch), 
                    # calculate L5000 (very expensive to do)
                    if (epoch % eval_epoch == 0) or (epoch==n_epochs):
                        progress_bar = tqdm(desc='Calculating log(p(x))', 
                                            total=len(testloader.dataset), unit='samples')
                        test_loss, test_recon, test_kl = 0, 0, self.model.latent_spaces*[0]
                        for i, (data, _) in enumerate(testloader):
                            data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
                            # We need to do this for each individual points, because
                            # iw_samples is high (running out of GPU memory)
                            for d in data:
                                out = self.model(d[None], 1, 1000)
                                loss, _, _ = vae_loss(d, *out, 1, 1000  ,
                                                      self.model.latent_dim, 
                                                      epoch, warmup, beta,
                                                      self.outputdensity)
                                test_loss += loss.item()
                                progress_bar.update()
                        progress_bar.close()
                        writer.add_scalar('test/L5000', test_loss, iteration)
                        
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
        
        
