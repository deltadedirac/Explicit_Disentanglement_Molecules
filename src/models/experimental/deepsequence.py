from ..encoder_decoder import mlp_encoder,mlp_decoder
from ..LossFunctionsAlternatives import LossFunctionsAlternatives
import torch.distributions as D
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
from torch import nn
import numpy as np
import time, os, datetime, gc, pdb
import pickle
PI = torch.from_numpy(np.asarray(np.pi))
from torch.nn.utils import clip_grad_norm_, clip_grad_value_


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

    def __init__(self, input_shape, latent_dim, alphabet, device = 'cpu', 
                 outputnonlin=nn.Softmax(dim=-1),beta=1):
        super(DeepSequence, self).__init__()
        # Constants
        #import ipdb; ipdb.set_trace()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.alphabet = alphabet
        #self.outputnonlin = nn.Softmax(dim=-1)
        self.outputnonlin = outputnonlin

        self.beta = beta
        #self.device = device
        self.device = (device,'cuda')[device =='gpu' or device =='cuda']
        """self.prior = D.Independent(D.Normal(torch.zeros(latent_dim).to(self.device),
                                            torch.ones(latent_dim).to(self.device)), 1)
        """
        #self.prior = D.Normal(torch.zeros(latent_dim).to(self.device), torch.ones(latent_dim).to(self.device))
        #import ipdb; ipdb.set_trace()
        # Define encoder and decoder
        self.encoder = mlp_encoder(input_shape, latent_dim, layer_ini = self.alphabet, dropout=0.0).to(self.device)
        self.decoder = mlp_decoder(input_shape, latent_dim, self.outputnonlin, layer_ini = self.alphabet, dropout=0.0).to(self.device)

    def KL(self, z, mu, log_var):
        #import ipdb; ipdb.set_trace()
        # case of using the toy example, since torch.distributions works pretty bad
        # when the amount of samples is very small, e.g batch_size=4. In those case
        # the solution is to propagate the gradients calcating manually the log probabilities
        # over Normals, otherwise, we can use torch.distributions instead
        #import ipdb; ipdb.set_trace()
        if z.shape[0] <= 2:
            log_z = log_standard_normal(z)
            log_qz = log_normal_diag(z, mu, log_var)
            return torch.nn.functional.kl_div( log_qz, log_z, reduction='none', log_target =True)
        else:
            q_dist = D.Normal(mu, log_var)

            # shape of prior=[latent_size]
            #self.prior = D.Normal(torch.zeros(self.latent_dim).to(self.device), torch.ones(self.latent_dim).to(self.device))
            
            # shape of prior=[batch_size, latent_size]
            self.prior = D.Normal(torch.zeros_like(q_dist.mean).to(self.device), torch.ones_like(q_dist.variance).to(self.device))

            if self.prior.rsample().shape == q_dist.rsample().shape:
                kl = D.kl_divergence(q_dist, self.prior)
            else:
                kl = D.kl_divergence(q_dist, self.prior)
            return kl
    
    def KL_alternative(self, x):

        mu, var = self.encoder(x)
        z = self.reparameterize(mu, var, 1, 1)
        log_z = log_standard_normal(z)
        log_qz = log_normal_diag(z, mu, var)
        return torch.nn.functional.kl_div( log_qz, log_z, reduction='none', log_target =True)
        #return ( log_z - log_qz )

    @staticmethod
    def load_BLAT(path):
        dataset = pickle.load(open(path,'rb'))

    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        import ipdb; ipdb.set_trace()
        batch_size, latent_dim = mu.shape
        # case of using the toy example, since torch.distributions works pretty bad
        # when the amount of samples is very small, e.g batch_size=4. In those case
        # the solution is to propagate the gradients calcating manually the log probabilities
        # over Normals, otherwise, we can use torch.distributions instead
        if batch_size <=5:
            eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
            return (mu[:,None,None,:] + var[:,None,None,:] * eps).reshape(-1, latent_dim)
        else:    
            return D.Normal(mu,  var).rsample()


        
    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0):
        # Encode/decode transformer space
        '''-------------------------------------------------------------------------------------------------------'''
        # Encode/decode semantic space
        mu, var = self.encoder(x)
        import ipdb; ipdb.set_trace()
        z = self.reparameterize(mu, var, 1, 1)
        
        KLD = self.KL(z, mu, var) 
        x_mean, x_var = self.decoder(z)  
        '''-------------------------------------------------------------------------------------------------------'''
        x_var = switch*x_var + (1-switch)*0.02**2
        
        return x_mean.contiguous(), \
                x_var.contiguous(), \
                z, mu, var, KLD

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
        #writer = SummaryWriter(log_dir=logdir)


        # Main loop
        self.train()
        start = time.time()
        import ipdb; ipdb.set_trace()
        for epoch in range(1, n_epochs+1):
            progress_bar = tqdm(desc='Epoch ' + str(epoch) + '/' + str(n_epochs), 
                                total=len(trainloader.dataset), unit='samples')
            train_loss = 0
            # Training loop
            for i, data in enumerate(trainloader):
                # Zero gradient
                optimizer.zero_grad()

                # Feed forward data
                #data = data.reshape(-1, *self.input_shape).to(torch.float32).to(self.device)
                data = data.to(torch.float32).to(self.device)

                switch = 1.0 if epoch > warmup else 0.0
                out = self.forward(data, eq_samples, iw_samples, switch)
                
                # Calculat loss
                #loss = loss_function(method = 'CE', input = out[0].squeeze(1), target = data, forw_per=(0,2,1))
                #loss = loss_function(method = 'CE', input = out[0], target = data, forw_per=(0,2,1)) + self.beta*out[5]
                #/2222.4939
                # prior over global parameters as it states on DeepSequence paper
                #import ipdb; ipdb.set_trace()
                
                # For loss with priors and distributions that are not i.i.e
                
                loss = ( torch.nn.functional.cross_entropy(out[0].permute(0,2,1), 
                                                data.argmax(-1), reduction = "none")
                                                + self.beta*(out[5]).mean(-1).reshape(-1,1) ).mean()
                """
                
                
                 # For loss with priors and distributions that are i.i.e using D.independent
                loss = ( torch.nn.functional.cross_entropy(out[0].permute(0,2,1), 
                                                data.argmax(-1), reduction = "none")#.sum(-1).reshape(-1,1)
                                                + self.beta*(out[5]).reshape(-1,1) ).mean()
                """
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
                #writer.add_scalar('train/total_loss', loss, iteration)
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

        #writer.close()

    def get_elbo(self, data, out, reduction='none'):

        #import ipdb; ipdb.set_trace()
        #out_nogaps = out[0][:,:,2:].permute(0,2,1)
        #data_nogaps = data[:,:,2:]
        #ELBO = ( torch.nn.functional.cross_entropy(out_nogaps, 
        #                                        data_nogaps.argmax(-1), reduction = "none") + out[5].sum(-1).reshape(-1,1) )
        
        # This applies when not assuming distributions with i.i.e samples, or without D.distributions.independent
        """
        ELBO = ( torch.nn.functional.cross_entropy(out[0].permute(0,2,1), 
                                                data.argmax(-1), reduction = "none")
                                                + out[5].sum(-1).reshape(-1,1) )
        """
        

        #ELBO = ( torch.nn.functional.cross_entropy(out[0].permute(0,2,1), 
        #                                        data.argmax(-1), reduction = "none")#.sum(-1).reshape(-1,1)
        #                                        + self.beta*(out[5]).reshape(-1,1) )
        
        
        ELBO = ( torch.nn.functional.cross_entropy(out[0].permute(0,2,1), 
                                                data.argmax(-1), reduction = "none")
                                                + self.beta*(out[5]).mean(-1).reshape(-1,1) ).mean(-1)
        


        if reduction == 'mean':
            return ELBO.mean()
        else:
            return ELBO
        #torch.nn.functional.cross_entropy(out[0].permute(0,2,1),data.argmax(-1), reduction = "none").mean(-1).reshape(-1,1) + self.beta*(out[5]).mean(-1).reshape(-1,1)