
import os, math, torch
import torch.nn as nn
import numpy as np
import torch.distributions as D

from .spatial_transformers_modes_tmp import get_transformer
from .encoder_decoder import conv_attention
from .C_VITAE import Flatten, Identity, C_VITAE


PI = torch.from_numpy(np.asarray(np.pi))

class LightAttention(nn.Module):
    # original implementation from
    # https://github.com/HannesStark/protein-localization.git
    def __init__(self, 
                 embeddings_dim=1024, 
                 output_dim=11, 
                 dropout=0.25, 
                 kernel_size=9, 
                 conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Sequential(
                                        nn.Conv1d(embeddings_dim, 
                                                  embeddings_dim, 
                                                  kernel_size, stride=1,
                                                  padding=kernel_size // 2),
                                        nn.Conv1d(embeddings_dim, 
                                                  embeddings_dim, 5, 
                                                  stride=1,
                                                  padding=5 // 2),
                                        nn.Conv1d(embeddings_dim, 
                                                  embeddings_dim, 3, 
                                                  stride=1,
                                                  padding=3 // 2) 
                                                  )
        
        self.attention_convolution = nn.Sequential(
                                        nn.Conv1d(embeddings_dim, 
                                                  embeddings_dim, 
                                                  kernel_size, stride=1,
                                                  padding=kernel_size // 2),
                                        nn.Conv1d(embeddings_dim, 
                                                  embeddings_dim, 5, 
                                                  stride=1,
                                                  padding=5 // 2),
                                        nn.Conv1d(embeddings_dim, 
                                                  embeddings_dim, 3, 
                                                  stride=1,
                                                  padding=3 // 2) 
                                                  )
        
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
    
class C_LA_VITAE(C_VITAE):

    def __init__(self, input_shape, 
                 config, 
                 latent_dim, 
                 encoder, 
                 decoder, 
                 outputdensity, 
                 ST_type,**kwargs):
        super(C_VITAE, self).__init__()

        # Constants

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_spaces = latent_dim
        self.outputdensity = outputdensity
        self.alphabet = kwargs["alphabet_size"]

        ndim, dev, gp_params = kwargs["trans_parameters"]
        #self.device = dev #(dev,'cuda')[dev =='gpu' or dev =='cuda']
        #self.device = (device,'cuda')[device =='gpu' or device =='cuda']
        
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
        self.stn = get_transformer(ST_type)(ndim, 
                                            config, 
                                            backend='pytorch', 
                                            device=dev, 
                                            zero_boundary=False,
                                            volume_perservation=False, 
                                            override=False, 
                                            argparser_gpdata = gp_params)
        
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



    @torch.no_grad()
    def MC_sampling_DeepSequence(self, DS, iters=100):
        DS.eval()
        set_of_samples = DS.sample(iters)[0]
        MC_sample = torch.mean(set_of_samples, dim=0)
        
        MC_sample = torch.nn.functional.softmax(MC_sample, dim=-1)
        return MC_sample

    def agnostic_KL(self, sampled_attent, tol=1e-2):

        q_dist_psi = D.Independent(
                        D.Normal(sampled_attent,
                                 torch.ones_like(sampled_attent) ),#tol),
                        1 )
        
        prior_psi = D.Independent( 
                        D.Normal(
                            torch.zeros_like(q_dist_psi.mean).to(self.device), 
                            torch.ones_like(q_dist_psi.variance).to(self.device)
                            ), 1 )
        
        return self.KL(q_dist_psi, prior_psi)
    

    def forward(self, x, deepS, eq_samples=1, iw_samples=1, switch=1.0):

        # This is just in case of wanting to do the task using somethign similar to
        # to cross attention
        #import ipdb; ipdb.set_trace()

        # Light Attention featurizer (Psi) to get
        # stochastic latent variable and KL constrain 100
        attention_repr = self.attention(x = x.permute(0,2,1),
                                        y = self.MC_sampling_DeepSequence( deepS, 100)\
                                            .unsqueeze(0).permute(0,2,1)
                                        )
        KLD_psi = self.agnostic_KL(attention_repr)


        # calculation of Zp latent variable component
        mu_zp, var_zp = self.encoder1(attention_repr)
        q_dist_zp, prior_zp = self.get_qz_and_pz( mu_zp, var_zp )

        zp_samples= self.reparameterize( q_dist_zp, n=1 )
        KLD_zp =  self.KL(q_dist_zp, prior_zp) 
        theta_mean, theta_var = self.decoder1(zp_samples) 

        # Transform input
        self.stn.st_gp_cpab.interpolation_type = 'GP'
        x_new = self.stn(x.repeat(eq_samples*iw_samples, 1, 1), 
                         theta_mean, 
                         self.Trainprocess, 
                         inverse=True).to(torch.device(self.device))

        '''-------------------------------------------------------------------------------------------------------'''
        #x_mean, x_var,  KLds = self.get_deepsequence_nograd( x_new ,deepS)
        x_mean, x_var, KL_ds = self.sample_from_deepsequence(x_new, 
                                                            deepS, n=1)#1
        '''-------------------------------------------------------------------------------------------------------'''
        # "Detransform" output
        self.stn.st_gp_cpab.interpolation_type = 'GP' 
        x_mean = self.stn(x_mean, theta_mean,  self.Trainprocess, inverse=False).to(torch.device(self.device))

        
        return x_mean, \
                x_var, \
                    [zp_samples, None], [mu_zp, None], [var_zp, None],\
                     x_new, theta_mean, KLD_zp,  KLD_psi, KL_ds



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
        
    def get_elbo(self,data, out, reduction='none', beta = 1):

        #import ipdb; ipdb.set_trace()
        ELBO  = ( torch.nn.functional.cross_entropy(out[0].permute(0,2,1), 
                            data.argmax(-1), reduction = "none") 
                                                + beta * out[7].reshape(-1,1)
                                                + beta * out[8].reshape(-1,1) 
                                                + beta * out[9].reshape(-1,1) 
                        ).mean(-1)
        
        if reduction == 'mean':
            return ELBO.mean()
        else:
            return ELBO
        