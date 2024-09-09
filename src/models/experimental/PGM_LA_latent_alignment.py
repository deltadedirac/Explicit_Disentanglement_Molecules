from ..vitae_ci_gp_tmp import *
from ..encoder_decoder import conv_attention
import math
import torch.nn as nn
import numpy as np
PI = torch.from_numpy(np.asarray(np.pi))


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input):
        return input.view(input.size(0), -1)
    



class LightAttention(nn.Module):
    # original implementation from https://github.com/HannesStark/protein-localization.git
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Sequential(
                                        nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2),
                                        nn.Conv1d(embeddings_dim, embeddings_dim, 5, stride=1,
                                             padding=5 // 2),
                                        nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1,
                                             padding=3 // 2) )
        
        self.attention_convolution = nn.Sequential(
                                        nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2),
                                        nn.Conv1d(embeddings_dim, embeddings_dim, 5, stride=1,
                                             padding=5 // 2),
                                        nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1,
                                             padding=3 // 2) )
        
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 320),
            nn.Dropout(dropout),
            nn.ReLU(),
            #nn.BatchNorm1d(320)
        )

        self.output = nn.Linear(320, output_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        if 'y' in kwargs:
            # in case of deciding something similar to cross attention
            y = kwargs['y']
            o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
            o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
            attention = self.attention_convolution(y)  # [batch_size, embeddings_dim, sequence_length]
        else:
            #import ipdb; ipdb.set_trace()
            o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
            o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
            attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        #attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o).unsqueeze(1)  # [batchsize, output_dim]
        #return torch.nn.functional.softmax(self.output(o).unsqueeze(1), dim=-1)  # [batchsize, output_dim]
    

class PGM_LA_latent_alignment(VITAE_CI):

    def __init__(self, input_shape, config, latent_dim, encoder, 
                 decoder, outputdensity, ST_type,**kwargs):
        super(VITAE_CI, self).__init__()
        # Constants
        #import ipdb; ipdb.set_trace()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_spaces = latent_dim
        self.outputdensity = outputdensity
        self.alphabet = kwargs["alphabet_size"]
        #ndim, dev, gp_params = kwargs["trans_parameters"]
        ndim, dev, gp_params = kwargs["trans_parameters"]
        #self.device = dev #(dev,'cuda')[dev =='gpu' or dev =='cuda']
        
        self.outputnonlin = torch.nn.Softmax(dim=2)#torch.nn.Softmax(dim=-1)
        # Spatial transformer
        ''' 
            Due to internal configuration in cpab, the spatial transformation take as 
            device "gpu", which is not true for device options inside pytorch. That's why
            it is necessary to make a casting of device in order to make it compatible with
            pytorch for the calculations. the cast_device method inside gp_cpab would make
            the job.
        '''
        
        self.ST_type = ST_type
        self.stn = get_transformer(ST_type)(ndim, config, backend='pytorch', device=dev, zero_boundary=False,
                                          volume_perservation=False, override=False, argparser_gpdata = gp_params)
        
        self.device = self.stn.st_gp_cpab.cast_device( dev )
        #self.Regularizer = Regularizer(1e-2, ndim[0]+1, self.device )


        self.Trainprocess = True
        self.diag_domain = kwargs['diagonal_att_regions']
        #self.attention =  conv_attention(channel_shape=abs(self.diag_domain[0]) + abs(self.diag_domain[1] + 1), 
        #                                shape_signal=self.input_shape, kernel=3)

        self.diagonal_comps = np.sum(np.absolute(self.diag_domain))+1
        self.attention = LightAttention(#embeddings_dim=np.sum(np.absolute(self.diag_domain))+1, #-16
                                        embeddings_dim=self.alphabet, #-16
                                        output_dim=self.diagonal_comps , dropout=0.03, kernel_size=9, conv_dropout = 0.03).to( self.device )
        
        #import ipdb; ipdb.set_trace()
        # Define encoder and decoder
        if isinstance(encoder,tuple) and isinstance(decoder,tuple):
            self.encoder1 = encoder[0](self.diagonal_comps, latent_dim, layer_ini = self.alphabet).to( self.device )
            self.decoder1 = decoder[0]((self.stn.dim(),), latent_dim, Identity(), layer_ini = self.alphabet).to( self.device )
            
        else:
            self.encoder1 = encoder([self.diagonal_comps], latent_dim, layer_ini = self.alphabet).to( self.device )
            self.decoder1 = decoder((self.stn.dim(),), latent_dim, Identity(), layer_ini = self.alphabet).to( self.device )



    def log_normal_diag(self, x, mu, log_var, reduction=None, dim=None):
        D = x.shape[1]
        log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p

    def log_standard_normal(self, x, reduction=None, dim=None):
        D = x.shape[1]
        log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * x**2.
        if reduction == 'avg':
            return torch.mean(log_p, dim)
        elif reduction == 'sum':
            return torch.sum(log_p, dim)
        else:
            return log_p

    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        
        batch_size, latent_dim = mu.shape
        eps = torch.randn(batch_size, eq_samples, iw_samples, latent_dim, device=var.device)
        return (mu[:,None,None,:] + var[:,None,None,:] * eps).reshape(-1, latent_dim)
        
        #return D.Normal(mu,  var.sqrt()).rsample()    
    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        zz = self.reparameterize(mu_e, log_var_e)
        return self.log_normal_diag(zz, mu_e, log_var_e)

    def KL(self, z, mu, log_var):
        log_z = self.log_standard_normal(z)
        log_qz = self.log_normal_diag(z, mu, log_var)
        #log_qz = self.log_prob(mu_e=mu, log_var_e=log_var, z=z)
        return torch.nn.functional.kl_div( log_qz, log_z, reduction='none', log_target =True)
        #return (log_z - log_qz ).mean() #        return ( log_z - log_qz ).sum(dim=-1).mean()
    
    def agnostic_KL(self, sampled_attent):
        #import ipdb; ipdb.set_trace()
        log_z_attent= self.log_standard_normal(torch.randn_like(sampled_attent))
        #log_q_sampled = torch.nn.functional.gumbel_softmax(sampled_attent, tau=1, hard=False, dim=-1)


        #q_dist = torch.distributions.Gumbel(sampled_attent, 1)
        q_dist=torch.distributions.Normal(sampled_attent, 1e-5)
        log_q_sampled = q_dist.log_prob(q_dist.rsample())
        return torch.nn.functional.kl_div(log_q_sampled, log_z_attent,  reduction='none', log_target =True)
        #return ( log_z_attent - log_q_sampled ).mean()
            
    @torch.no_grad()
    def get_deepsequence_nograd(self, x, DS):
        #x_copy = torch.tensor(x, requires_grad=False)
        with torch.no_grad():
            DS.eval()
            x_mean_no_grad, x_var_no_grad,_,__,____,KLds = DS(x) #_copy)
        #return x_mean_no_grad, x_var_no_grad, KLds
        return torch.nn.functional.softmax(x_mean_no_grad, dim=-1),\
               torch.nn.functional.softmax(x_var_no_grad,dim=-1), KLds


    @torch.no_grad()
    def MC_sampling_DeepSequence(self, DS, iters=100):
        DS.eval()
        set_of_samples = DS.sample(iters)[0] #[ DS.sample(1)[0] for i in range(0,iters)]
        MC_sample = torch.mean(set_of_samples, dim=0)
        return MC_sample

        
    def forward(self, x, deepS, eq_samples=1, iw_samples=1, switch=1.0):

        # This is just in case of wanting to do the task using somethign similar to
        # to cross attention
        attention_repr = self.attention(x=x.permute(0,2,1),
                                        y=self.MC_sampling_DeepSequence( deepS, 100)\
                                            .unsqueeze(0).permute(0,2,1))
        

        mu1, var1 = self.encoder1(attention_repr)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1) 

        KLD = self.KL(z1, mu1, var1)
        KLD_attent = self.agnostic_KL( attention_repr)

        # Transform input
        self.stn.st_gp_cpab.interpolation_type = 'GP'
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1), theta_mean, self.Trainprocess, inverse=True).to(torch.device(self.device))

        # In case of using the log space in the prior for avoid local minima 

        '''-------------------------------------------------------------------------------------------------------'''
        # Pretrained DeepSequence Output
        x_mean, x_var,  KLds = self.get_deepsequence_nograd(x_new,deepS)

        '''-------------------------------------------------------------------------------------------------------'''
        # "Detransform" output
        self.stn.st_gp_cpab.interpolation_type = 'GP' 
        x_mean = self.stn(x_mean, theta_mean,  self.Trainprocess, inverse=False).to(torch.device(self.device))

        
        return x_mean, \
                x_var, \
                    [z1, None], [mu1, None], [var1, None], x_new, theta_mean, KLD,  KLD_attent, KLds



    def sample_cross_attention(self, x, DS):
        with torch.no_grad():
            return self.attention(x=x.permute(0,2,1),
                                        y=DS.sample(1)[0]\
                                        .unsqueeze(0).permute(0,2,1))
        
    def sample_only_trans(self, x):
        device = next(self.parameters()).device
        with torch.no_grad():
            mu1, var1 = self.encoder1(x)
            z1 = torch.randn(x.shape[0], self.latent_dim, device=device)
            theta_mean, _ = self.decoder1(z1)
            self.stn.st_gp_cpab.interpolation_type = 'GP'
            x_new = self.stn(x.repeat(1, 1, 1), theta_mean, self.Trainprocess, inverse=True)
            return x_new, theta_mean
        
    def get_elbo(self,data, out, reduction='none'):

        ELBO  = ( torch.nn.functional.cross_entropy(out[0].permute(0,2,1), 
                            data.argmax(-1), reduction = "none") 
                                                + out[7].mean(-1).reshape(-1,1)
                                                + out[8].mean(-1).reshape(-1,1) 
                                                + out[9].mean(-1).reshape(-1,1) ).mean(-1)
        
        if reduction == 'mean':
            return ELBO.mean()
        else:
            return ELBO
        