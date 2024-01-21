from ..vitae_ci_gp_tmp import *
from ..encoder_decoder import conv_attention
import numpy as np
PI = torch.from_numpy(np.asarray(np.pi))
import math


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Regularizer(nn.Module):
    def __init__(self, default_val = 1e-7, n_comps=60, device='cpu'):
        super(Regularizer, self).__init__()
        self.device = (device,'cuda')[device =='gpu' or device =='cuda']
        self.Regularizer = torch.nn.Parameter( data = torch.tensor([default_val] ,requires_grad=True, device = torch.device(self.device)),  requires_grad=True)
        #self.Regularizer = torch.nn.Parameter( data = torch.tensor([default_val] ,requires_grad=True, device=torch.device(device) ),  requires_grad=True)
    
    def forward(self, input):
        return input*self.Regularizer   
    

import torch.nn as nn


class LightAttention(nn.Module):
    # original implementation from https://github.com/HannesStark/protein-localization.git
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Sequential(
                                        nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2),
                                        nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1,
                                             padding=3 // 2) )
        self.attention_convolution = nn.Sequential(
                                        nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2),
                                        nn.Conv1d(embeddings_dim, embeddings_dim, 3, stride=1,
                                               padding=3 // 2) ) # initially worked with kernel_size=3 for this second blocks of convolutions

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            #nn.Linear(2 * embeddings_dim, 32),
            nn.Linear(2 * embeddings_dim, 320),
            nn.Dropout(dropout),
            nn.ReLU(),
            #nn.BatchNorm1d(32)
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

    def __init__(self, input_shape, config, latent_dim, encoder, decoder, outputdensity, ST_type, **kwargs):
        super(VITAE_CI, self).__init__()
        # Constants

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
                                        embeddings_dim=22, #-16
                                        output_dim=self.diagonal_comps , dropout=0.25, kernel_size=9, conv_dropout = 0.25).to( self.device )
        
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
        
    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        zz = self.reparameterize(mu_e, log_var_e)
        return self.log_normal_diag(zz, mu_e, log_var_e)

    def KL(self, z, mu, log_var):
        log_z = self.log_standard_normal(z)
        #log_qz = self.log_normal_diag(z, mu, log_var)
        log_qz = self.log_prob(mu_e=mu, log_var_e=log_var, z=z)
        return ( log_z - log_qz ).mean() #        return ( log_z - log_qz ).sum(dim=-1).mean()
    
    def agnostic_KL(self, sampled_attent):
        #import ipdb; ipdb.set_trace()
        log_z_attent= self.log_standard_normal(torch.randn_like(sampled_attent))
        #log_q_sampled = torch.nn.functional.softmax(sampled_attent, dim=-1)
        log_q_sampled = torch.nn.functional.gumbel_softmax(sampled_attent, tau=1, hard=False, dim=-1)
        return ( log_z_attent - log_q_sampled ).mean()
            
    @torch.no_grad()
    def get_deepsequence_nograd(self, x, DS):
        #x_copy = torch.tensor(x, requires_grad=False)
        with torch.no_grad():
            DS.eval()
            x_mean_no_grad, x_var_no_grad,_,__,____,_____ = DS(x)
        return x_mean_no_grad, x_var_no_grad

    @torch.no_grad()
    def MC_sampling_DeepSequence(self, DS, iters=100):
        DS.eval()
        set_of_samples = DS.sample(iters)[0] #[ DS.sample(1)[0] for i in range(0,iters)]
        MC_sample = torch.mean(set_of_samples, dim=0)
        return MC_sample

    def get_diagonal_attention(self, Matrix, comp, min_r, max_r, list_attention = []):
    # Works assuming that we are dealing with square matrixes

        if comp <= max_r and comp >= min_r:
            prod_diag = torch.diagonal(Matrix, comp)
            min_val = 1e-9 #Matrix.min().item()              # to specify the gaps
            #min_val = prod_diag.min().item()
            if comp > 0 and len(prod_diag) < len(Matrix):

                list_attention.append( torch.cat( 
                    ( prod_diag, torch.tensor([min_val]* ( len(Matrix) - len(prod_diag)) ).to(torch.device(self.device))  )
                    )   )
            elif comp < 0 and len(prod_diag) < len(Matrix):
                list_attention.append( torch.cat( 
                    ( torch.tensor([min_val]* (len(Matrix) - len(prod_diag) )).to(torch.device(self.device)) , prod_diag ) # change 0 for prod_diag.min()
                    )   )
            else:
                list_attention.append( torch.diagonal(Matrix, comp) )
            
            self.get_diagonal_attention(Matrix, comp-1, min_r, max_r, list_attention)

    def get_batch_diagonal_attention(self, Matrix, comp, min_r, max_r):
        list_batch = []
        list_attention = []

        for m in Matrix:
            if max_r == None or min_r == None:
                list_batch.append( m.T )
            else:
                self.get_diagonal_attention(m, comp, min_r, max_r, list_attention )
                list_batch.append(   torch.fliplr(torch.column_stack(list_attention) )  )
            list_attention.clear()

        batch_diag_attention = torch.stack(list_batch)
        return batch_diag_attention



    @torch.no_grad()
    def get_attention_matrix(self, raw_seqs, DS, iters=100):

        #import ipdb;ipdb.set_trace()
        MC_DS_protein = self.MC_sampling_DeepSequence( DS, iters)
        list_of_attentions = []

        seqs = torch.matmul(raw_seqs, MC_DS_protein.T)
        #seqs = torch.nn.Softmax(dim=-1)(seqs)
        list_of_attentions= list(seqs)
        return list_of_attentions




        
    def forward(self, x, deepS, eq_samples=1, iw_samples=1, switch=1.0):
        # Encode/decode transformer space
        #import ipdb; ipdb.set_trace()

        '''
        # extract the attention mechanisms between the
        l_attention = self.get_attention_matrix(x,deepS, iters=1000)
        batch_diagonal_regions =  self.get_batch_diagonal_attention(l_attention, 
                                                                    self.diag_domain[1], 
                                                                    self.diag_domain[0], 
                                                                    self.diag_domain[1])
        attention_repr = self.attention(batch_diagonal_regions.permute(0,2,1))
        '''
        #import ipdb; ipdb.set_trace()
        # This is just in case of wanting to do the task using somethign similar to
        # to cross attention
        attention_repr = self.attention(x=x.permute(0,2,1),
                                        y=self.MC_sampling_DeepSequence( deepS, 10000)\
                                            .unsqueeze(0).permute(0,2,1))
        
        KLD_attent = self.agnostic_KL( attention_repr)

        mu1, var1 = self.encoder1(attention_repr)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1) # JUST TO SEE IF THIS SCALING WORKS
        #theta_mean = self.Regularizer(theta_mean)

        KLD = self.KL(z1, mu1, var1)

        # Transform input
        self.stn.st_gp_cpab.interpolation_type = 'GP'
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1), theta_mean, self.Trainprocess, inverse=True).to(torch.device(self.device))

        # In case of using the log space in the prior for avoid local minima 
        #if self.prior_space=='log':
        #    x_new = self.outputnonlin(x_new)
        '''-------------------------------------------------------------------------------------------------------'''
        # Pretrained DeepSequence Output
        x_mean_no_grad, x_var_no_grad = self.get_deepsequence_nograd(x_new,deepS)

        x_mean = torch.tensor(x_mean_no_grad, requires_grad=True)
        x_var = torch.tensor(x_var_no_grad, requires_grad=True)
        '''-------------------------------------------------------------------------------------------------------'''
        # "Detransform" output
        self.stn.st_gp_cpab.interpolation_type = 'GP' 
        x_mean = self.stn(x_mean, theta_mean,  self.Trainprocess, inverse=False).to(torch.device(self.device))
        #x_var = self.stn(x_var, theta_mean, self.Trainprocess, inverse=False).to(torch.device(self.device))
        #x_var = switch*x_var + (1-switch)*0.02**2

        # In case of using the log space in the prior for avoid local minima 
        #if self.prior_space=='log':
        #    x_mean = self.outputnonlin(x_mean)
        #    x_var = self.outputnonlin(x_var)
        
        return x_mean, \
                x_var, \
                    [z1, None], [mu1, None], [var1, None], x_new, theta_mean, KLD - KLD_attent



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