#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:44:23 2018

@author: nsde
"""

#%%
from torch import nn
import numpy as np
from src.unsuper.unsuper.helper.encoder_decoder import mlp_encoder,mlp_decoder
from src.unsuper.unsuper.helper.utility import CenterCrop, Flatten, BatchReshape
import pdb

#%%

def get_encoder(encoder_name):
    models = {'mlp': mlp_encoder,
              'conv': conv_encoder}
    assert (encoder_name in models), 'Encoder not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[encoder_name]

#%%
def get_decoder(decoder_name):
    models = {'mlp': mlp_decoder,
              'conv': conv_decoder}
    assert (decoder_name in models), 'Decoder not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[decoder_name]

#%%
def get_list_encoders(encoder_name):
    models = {'mlp': mlp_encoder,
              'conv': conv_encoder}
    list_models = ()
    for i in encoder_name:
        assert (i in models), 'Encoder not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
        list_models +=( models[i] ,)
    return list_models

#%
def get_list_decoders(decoder_name):
    models = {'mlp': mlp_decoder,
              'conv': conv_decoder}
    list_models = ()
    for i in decoder_name:
        assert (i in models), 'Encoder not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
        list_models +=( models[i] ,)
    return list_models

def calculate_features_cat_size(l , c_pad = 1, dil = 1, k_size = 3, c_stride =1):
  #import pdb; pdb.set_trace()
  l_out = int( 1 + ( l + 2*c_pad - dil*( k_size - 1) - 1 )/c_stride )
  return l_out
#%%
class Resizing(nn.Module):
    def __init__(self, input_shape):
        super(Resizing, self).__init__()
        self.resize = input_shape

    def forward(self, x):
        return x.reshape(-1, *self.resize)
#%%
#best setup 512,256,latent_dim -> latent_dim,256,512
class mlp_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, **kwargs):
        super(mlp_encoder, self).__init__()
        self.flat_dim = np.prod(input_shape)
        self.encoder_mu = nn.Sequential(
            #nn.BatchNorm1d(input_shape[0]),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(256, latent_dim),
            #nn.LeakyReLU(0.1)
        )
        self.encoder_var = nn.Sequential(
            #nn.BatchNorm1d(input_shape[0]),
            nn.Linear(self.flat_dim, 512),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Softplus(),
        )
        self.encoder_mu.apply(self._init_weights)
        self.encoder_var.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            #nn.init.constant_(module.bias, 0)
            #nn.init.zeros_(module.weight)
            #nn.init.normal_(module.weight,mean=0, std=1e-8 ) # works on 1e-6, DONT FORGET


        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        #x = x.flatten()
        z_mu = self.encoder_mu(x)
        z_var = self.encoder_var(x)
        return z_mu, z_var

#%%
class mlp_decoder(nn.Module):
    def __init__(self, output_shape, latent_dim, outputnonlin, **kwargs):
        super(mlp_decoder, self).__init__()
        self.flat_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.outputnonlin = outputnonlin

        
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(512, self.flat_dim),
            Resizing(self.output_shape),
            self.outputnonlin
            #nn.LeakyReLU(0.1) #nn.ReLU(),
        )
        
        self.decoder_var = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(512, self.flat_dim),
            Resizing(self.output_shape),
            nn.Softplus()
            #nn.LeakyReLU(0.1) #nn.ReLU(),
        )
        
        self.decoder_mu.apply(self._init_weights)
        self.decoder_var.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            #nn.init.zeros_(module.weight)
            nn.init.kaiming_normal_(module.weight)
            #nn.init.normal_(module.weight,mean=0, std=1e-8 ) # works on 1e-6, DONT FORGET
            #nn.init.constant_(module.bias, 0)
        
    def forward(self, z):
        #pdb.set_trace()
        '''
        x_mu = self.decoder_mu(z).reshape(-1, *self.output_shape)
        x_mu = self.outputnonlin(x_mu)

        x_var = self.decoder_var(z).reshape(-1, *self.output_shape)
        x_var = nn.Softplus()(x_var)
        '''
        x_mu = self.decoder_mu(z)
        x_var = self.decoder_var(z)
        return x_mu, x_var

#%%    
import torch
class conv_attention(nn.Module):
    
    def __init__(self, channel_shape, input_shape, shape_signal, kernel):
        super(conv_attention, self).__init__()
        self.channel_dim = channel_shape
        self.input_shape = input_shape # pos0 = #channels, pos1 = #diagonal comps, or viseversa
        len_diagonal_comp = shape_signal #e.g. [39,22], if I choose 

        self.encoder = nn.Sequential(
            nn.Conv1d(self.channel_dim, self.channel_dim, kernel_size=kernel, stride=1, padding=kernel//2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.channel_dim, self.channel_dim, kernel_size=kernel, stride=1, padding=kernel//2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.channel_dim, self.channel_dim, kernel_size=kernel, stride=1, padding=kernel//2),
        )
        
        self.encoder.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            #nn.init.zeros_(module.weight)
            nn.init.kaiming_normal_(module.weight)
            #nn.init.constant_(module.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        return z #z.permute(0,2,1) 

#%%

class conv_attention2(nn.Module):
    
    def __init__(self, channel_shape, input_shape, shape_signal, kernel, deconv_flat_out):
        super(conv_attention2, self).__init__()
        self.channel_dim = channel_shape
        self.input_shape = input_shape # pos0 = #channels, pos1 = #diagonal comps, or viseversa
        len_diagonal_comp = shape_signal #e.g. [39,22], if I choose 
        self.deconv_out = deconv_flat_out

        channel_size_convs=[10,5]
        channel_size_deconvs=[15,28]
        kernel_cnn = kernel
        kernel_decnn = 30 #15


        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.channel_dim),
            nn.Conv1d(self.channel_dim, self.channel_dim, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2), # the default was 10
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.channel_dim,self.channel_dim, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.channel_dim,self.channel_dim, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),
            nn.LeakyReLU(0.1),
            #nn.Dropout(0.1),
            #nn.AvgPool1d(kernel_size = 3, stride=1, padding=1),
        )

        self.upsampling = nn.Sequential(
            #nn.ConvTranspose1d(10, 10, kernel_size=kernel_cnn, stride=2, padding=kernel_cnn//2, output_padding=1),
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.channel_dim, self.channel_dim, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),   #(15, 28)
            nn.LeakyReLU(0.1), #0.5
            nn.Conv1d(self.channel_dim, 1, kernel_size=1, stride=1),   #(15, 28)
            #Flatten()
        )
                    
    def forward(self, x):
        z = self.encoder(x)
        z = self.upsampling(z).squeeze(1)
        return z
#%%    

class conv_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, **kwargs):
        super(conv_encoder, self).__init__()
        self.latent_dim = latent_dim
        self.alphabet_layer_ini = kwargs["layer_ini"]
        self.len_seqs, self.channel = input_shape
        kernel_cnn=5


        self.encoder_mu = nn.Sequential(
            #nn.Conv1d(self.alphabet_layer_ini, 64, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),
            nn.Conv1d(self.channel, self.channel, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.channel, self.channel, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.channel, self.channel, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),
            nn.LeakyReLU(0.1),
            Flatten(),
            nn.Linear(self.channel*self.len_seqs, self.latent_dim)
        )
        self.encoder_var = nn.Sequential(
            nn.Conv1d(self.channel, self.channel, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.channel, self.channel, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),   #64
            nn.LeakyReLU(0.1),
            nn.Conv1d(self.channel, self.channel, kernel_size=kernel_cnn, stride=1, padding=kernel_cnn//2),
            nn.LeakyReLU(0.1),
            Flatten(),
            nn.Linear(self.channel*self.len_seqs, self.latent_dim),
            nn.Softplus()
        )
        self.encoder_mu.apply(self._init_weights)
        self.encoder_var.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.zeros_(module.weight)
            #nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        #pdb.set_trace()
        xx = x.permute(0,2,1)
        z_mu = self.encoder_mu(xx)
        z_var = self.encoder_var(xx)
        return z_mu, z_var
    
#%%
class conv_decoder(nn.Module):
    def __init__(self, output_shape, latent_dim, outputnonlin, **kwargs):
        super(conv_decoder, self).__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        self.outputnonlin = outputnonlin
        self.alphabet_layer_ini = kwargs["layer_ini"]


        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, self.alphabet_layer_ini*self.alphabet_layer_ini*1),
            BatchReshape((self.alphabet_layer_ini, self.alphabet_layer_ini)),
            nn.Conv1d(self.alphabet_layer_ini, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(0.1),
            Flatten(),
            #nn.Linear(64*25*25, 1*28*28),
            #outputnonlin
        )
        
        self.decoder_var = nn.Sequential(
            nn.Linear(latent_dim, self.alphabet_layer_ini*self.alphabet_layer_ini*1),
            BatchReshape(( self.alphabet_layer_ini, self.alphabet_layer_ini)),
            nn.Conv1d(self.alphabet_layer_ini, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(0.1),
            Flatten(),
            #nn.Linear(64*25*25, 1*28*28),
            #nn.Softplus()
        )
    
        #self.decoder_mu.apply(self._init_weights)
        #self.decoder_var.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            #nn.init.constant_(module.bias, 0)

    def forward(self, z):
        #pdb.set_trace()

        #x_mu = nn.Linear(self.latent_dim, self.llayer_from_enc[1])(z)
        #x_mu = BatchReshape((20, self.llayer_from_enc[1]/20))(x_mu)  
        x_mu = self.decoder_mu(z)
        x_mu = nn.Linear(x_mu.shape[1], np.prod(self.output_shape))(x_mu)
        x_mu = self.outputnonlin(x_mu.reshape(-1, *self.output_shape))

        #x_var = nn.Linear(self.latent_dim, self.llayer_from_enc)(z)
        #x_var = BatchReshape((20, self.llayer_from_enc/20))(x_var)   
        x_var = self.decoder_var(z)
        x_var = nn.Linear(x_var.shape[1], np.prod(self.output_shape))(x_var)
        x_var = nn.Softplus()(x_var).reshape(-1, *self.output_shape)
        return x_mu, x_var