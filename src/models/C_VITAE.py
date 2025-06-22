
import os
import torch
from torch import nn
import numpy as np

import torch.distributions as D
from .spatial_transformers_modes_tmp import get_transformer

PI = torch.from_numpy(np.asarray(np.pi))

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input):
        return input.view(input.size(0), -1)   


#%%
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x  

#%%

class C_VITAE(nn.Module):

    def __init__(self, input_shape, 
                 config, 
                 latent_dim, 
                 encoder, 
                 decoder, 
                 outputdensity, 
                 ST_type, **kwargs):
        super(C_VITAE, self).__init__()
        # Constants

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_spaces = latent_dim
        self.outputdensity = outputdensity
        self.alphabet = kwargs["alphabet_size"]
        ndim, device, gp_params = kwargs["trans_parameters"]

        self.device = (device,'cuda')[device =='gpu' or device =='cuda']

        # Spatial transformer
        
        self.ST_type = ST_type
        #import ipdb; ipdb.set_trace()

        self.stn = get_transformer(ST_type)(ndim, 
                                            config, 
                                            backend='pytorch', 
                                            device=device, 
                                            zero_boundary=False,
                                            volume_perservation=False, 
                                            override=False, 
                                            argparser_gpdata = gp_params)
        
        if 'posterior_variance' in kwargs:
            self.stn.st_gp_cpab.set_posterior_variance(kwargs['posterior_variance'])

        self.Trainprocess = True
        self.x_trans1 = torch.tensor([])

        # Define encoder and decoder
        if isinstance(encoder,tuple) and isinstance(decoder,tuple):
            self.encoder1 = encoder[0](input_shape, 
                                       latent_dim, 
                                       layer_ini = self.alphabet).to( self.device )
            
            self.decoder1 = decoder[0]((self.stn.dim(),), 
                                       latent_dim, Identity(), 
                                       layer_ini = self.alphabet).to( self.device )
            
        else:
            self.encoder1 = encoder(input_shape, 
                                    latent_dim, 
                                    layer_ini = self.alphabet).to( self.device )
            self.decoder1 = decoder((self.stn.dim(),), 
                                    latent_dim, Identity(), 
                                    layer_ini = self.alphabet).to( self.device )

    def KL(self, q_dist, prior):
        kl = D.kl_divergence(q_dist, prior)
        return kl

    def KL_manual(self, q_dist, prior, z):
        #import ipdb; ipdb.set_trace()
        # Compute log-probs
        log_qz = q_dist.log_prob(z) 
        log_pz = prior.log_prob(z)  
        # Manual KL estimate
        #kl_manual = log_qz - log_pz
        kl_manual = torch.nn.functional.kl_div( log_qz, 
                                                log_pz, 
                                                reduction='none', 
                                                log_target =True)
        return kl_manual
    

    def reparameterize(self, q_dist, n=1):
        if n == 1:
            return q_dist.rsample()
        
        return q_dist.rsample((n,)).mean(0)
        

    @torch.no_grad()
    def sample_from_deepsequence(self, x, DS, n=1):
        #import ipdb; ipdb.set_trace()
        DS = DS.to( self.device ); DS.eval()
        mu, var = DS.encoder(x)
        # Create a variational distribution from the
        # calculated mu and var in the encoder layer
        
        q_dist_ds, prior_ds = self.get_qz_and_pz( mu, var )
        
        z_samples = self.reparameterize( q_dist_ds, n )

        KLD = self.KL(q_dist_ds, prior_ds) 
        #KLD = self.KL_manual(q_dist_ds, prior_ds, z_samples)
        
        sample_mu, sample_var = DS.decoder(z_samples)
        
        #return sample_mu, sample_var, KLD
        return \
            torch.nn.functional.softmax(sample_mu, dim=-1), \
            torch.nn.functional.softmax(sample_var, dim=-1), \
            KLD

    def get_qz_and_pz(self, mu, var):    
        
        # Generate the Normal distributions with Diagonal Covariance Matrices
        q_dist = D.Independent( 
            D.Normal(mu, torch.exp(0.5 * var)), 
            #D.Normal(mu, torch.nn.functional.softplus( var) ), 
            1 )

        prior = D.Independent( 
                            D.Normal(
                                torch.zeros_like(q_dist.mean).to(self.device), 
                                torch.ones_like(q_dist.variance).to(self.device)
                                ), 1 )
        return q_dist, prior
        

    def forward(self, x, deepS, eq_samples=1, iw_samples=1, switch=1.0):
        # Encode/decode transformer space
        #import ipdb; ipdb.set_trace()
        mu1, var1 = self.encoder1(x)
        
        q_dist1, self.prior = self.get_qz_and_pz( mu1, var1 )


        z1 = self.reparameterize(q_dist1)
        theta_mean, theta_var = self.decoder1(z1)

        KLD = self.KL(q_dist1, self.prior) 
        #KLD = self.KL_manual(q_dist1, self.prior, z1)

        # Transform input
        self.stn.st_gp_cpab.interpolation_type = 'GP'
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1), 
                         theta_mean, self.Trainprocess, inverse=True)

        '''-------------------------------------------------------------------------------------------------------'''
        # Pretrained DeepSequence Output
        x_mean, x_var, KLds = self.sample_from_deepsequence(x_new, 
                                                            deepS, n=1)

        '''-------------------------------------------------------------------------------------------------------'''
        # "Detransform" output
        self.stn.st_gp_cpab.interpolation_type = 'GP' 
        x_mean = self.stn(x_mean, theta_mean,  self.Trainprocess, inverse=False)
        x_var = self.stn(x_var, theta_mean, self.Trainprocess, inverse=False)

        
        return  x_mean.contiguous(), \
                x_var.contiguous(), [z1, None], [mu1, None], \
                [var1, None], x_new, theta_mean, KLD, KLds


    def sample_only_trans(self, x):
        device = next(self.parameters()).device
        with torch.no_grad():
            #mu1, var1 = self.encoder1(x)
            z1 = torch.randn(x.shape[0], self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            self.stn.st_gp_cpab.interpolation_type = 'GP'
            x_new = self.stn(x.repeat(1, 1, 1), theta_mean, self.Trainprocess, inverse=True)
            return x_new, theta_mean
    
    
    def get_elbo(self, data, out, reduction='none', beta = 1):
        
        #import ipdb; ipdb.set_trace()
        ELBO = ( torch.nn.functional.cross_entropy(out[0].permute(0,2,1), 
                                               data.argmax(-1), reduction = "none") \
                                                + beta * out[7].reshape(-1,1)
                                                + beta * out[8].reshape(-1,1)
                                                #+ out[7].mean(-1).reshape(-1,1)
                                                #+ out[8].mean(-1).reshape(-1,1)
                                                ).mean(-1)


        if reduction == 'mean':
            return ELBO.mean()
        else:
            return ELBO