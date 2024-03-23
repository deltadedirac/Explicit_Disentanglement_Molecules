from ..encoder_decoder import mlp_encoder,mlp_decoder
from ..LossFunctionsAlternatives import LossFunctionsAlternatives
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
from torch import nn
import numpy as np
import time, os, datetime, gc, pdb
import pickle
PI = torch.from_numpy(np.asarray(np.pi))

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

class DeepSequence(nn.Module):

    def __init__(self, input_shape, latent_dim, alphabet, device = 'cpu', beta=1):
        super(DeepSequence, self).__init__()
        # Constants

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.alphabet = alphabet
        self.outputnonlin = nn.Softmax(dim=-1)#nn.Softmax(dim=2)
        self.beta = beta
        #self.device = device
        self.device = (device,'cuda')[device =='gpu' or device =='cuda']

        # Define encoder and decoder
        self.encoder = mlp_encoder(input_shape, latent_dim, layer_ini = self.alphabet).to(self.device)
        self.decoder = mlp_decoder(input_shape, latent_dim, self.outputnonlin, layer_ini = self.alphabet).to(self.device)

    def KL(self, z, mu, log_var):
        log_z = log_standard_normal(z)
        log_qz = log_normal_diag(z, mu, log_var)
        return torch.nn.functional.kl_div( log_qz, log_z, reduction='none', log_target =True)

        #return ( log_z - log_qz ).mean()
    
    def KL_alternative(self, x):

        mu, var = self.encoder(x)
        z = self.reparameterize(mu, var, 1, 1)
        log_z = log_standard_normal(z)
        log_qz = log_normal_diag(z, mu, var)
        return ( log_z - log_qz )

    @staticmethod
    def load_BLAT(path):
        dataset = pickle.load(open(path,'rb'))

    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        batch_size, latent_dim = mu.shape
        eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
        return (mu[:,None,None,:] + var[:,None,None,:].sqrt() * eps).reshape(-1, latent_dim)
        
    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0):
        # Encode/decode transformer space
        '''-------------------------------------------------------------------------------------------------------'''
        # Encode/decode semantic space
        mu, var = self.encoder(x)
        z = self.reparameterize(mu, var, 1, 1)
        KLD = self.KL(z, mu, var) 
        x_mean, x_var = self.decoder(z)  
        '''-------------------------------------------------------------------------------------------------------'''
        x_var = switch*x_var + (1-switch)*0.02**2
        
        return x_mean.contiguous(), \
                x_var.contiguous(), \
                z, mu, var, KLD.mean()

    def sample(self, n, switch=1.0):
        device = self.device
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=device)
            x_mean, x_var = self.decoder(z)
            x_var = switch*x_var + (1-switch)*0.02**2
            return x_mean.contiguous(), \
                            x_var.contiguous(), \
                                z



    def training_representation( self, trainloader, loss_function, optimizer, n_epochs=10, warmup=1, logdir='', out_modelname='trained_model.pth',eq_samples=1, iw_samples=1, beta=1.0):

        # Dir to log results
        logdir = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M') if logdir is None else logdir
        if not os.path.exists(logdir): os.makedirs(logdir)

        # Summary writer
        writer = SummaryWriter(log_dir=logdir)


        # Main loop
        self.train()
        start = time.time()
        for epoch in range(1, n_epochs+1):
            progress_bar = tqdm(desc='Epoch ' + str(epoch) + '/' + str(n_epochs), 
                                total=len(trainloader.dataset), unit='samples')
            train_loss = 0
            # Training loop
            for i, data in enumerate(trainloader):
                # Zero gradient
                optimizer.zero_grad()
                #import pdb;pdb.set_trace()
                # Feed forward data
                #data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
                data = data.to(torch.float32).to(self.device)

                switch = 1.0 if epoch > warmup else 0.0
                out = self.forward(data, eq_samples, iw_samples, switch)
                
                # Calculat loss
                #loss = loss_function(method = 'CE', input = out[0].squeeze(1), target = data, forw_per=(0,2,1))
                loss = loss_function(method = 'CE', input = out[0], target = data, forw_per=(0,2,1)) + self.beta*out[5]

                
                # Backpropegate and optimize
                # We need to maximize the bound, so in this case we need to
                # minimize the negative bound
                #(-loss).backward()
                loss.backward()
                optimizer.step()
                
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
        torch.save(self.state_dict(), logdir + '/' + out_modelname)

        #import pdb;pdb.set_trace()

        # Close summary writer

        writer.close()