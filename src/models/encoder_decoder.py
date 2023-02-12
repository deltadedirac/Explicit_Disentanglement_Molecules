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
        #self.encoder_mu.apply(self._init_weights)
        #self.encoder_var.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        
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
            #nn.LeakyReLU(0.1) #nn.ReLU(),
        )
        self.decoder_var = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1), #nn.ReLU(),
            nn.Linear(512, self.flat_dim),
            #nn.LeakyReLU(0.1) #nn.ReLU(),
        )
        #self.decoder_mu.apply(self._init_weights)
        #self.decoder_var.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        
    def forward(self, z):
        #pdb.set_trace()
        x_mu = self.decoder_mu(z).reshape(-1, *self.output_shape)
        x_mu = self.outputnonlin(x_mu)

        x_var = self.decoder_var(z).reshape(-1, *self.output_shape)
        x_var = nn.Softplus()(x_var)
        return x_mu, x_var

#%%    
import torch
class conv_attention(nn.Module):
    
    def __init__(self, channel_shape, shape_signal, kernel):
        super(conv_attention, self).__init__()
        self.channel_dim = channel_shape
        self.input_shape = shape_signal
        self.final_out_cnn = self.calc_conv_length_out( self.input_shape[0] , 3, c_pad = 1, dil = 1, k_size = kernel, c_stride =1)

        self.encoder = nn.Sequential(
            nn.Conv1d(self.channel_dim, 12, kernel_size=kernel, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(12, 12, kernel_size=kernel, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(12, 12, kernel_size=kernel, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            Flatten(),
            # To be fixable and automatic when the new molecule domains comes by
            nn.Linear(12*self.final_out_cnn, np.prod(self.input_shape))
            #nn.Linear(self.final_out_cnn, self.input_shape[0])
        )

    def calc_conv_length_out(self, l , cont, c_pad = 1, dil = 1, k_size = 3, c_stride =1):
        #import pdb; pdb.set_trace()
        if cont>0:
            l = min( l, self.calc_conv_length_out( int( 1 + ( l + 2*c_pad - dil*( k_size - 1) - 1 )/c_stride ), 
                                             cont-1, c_pad = c_pad, 
                                             dil = dil, 
                                             k_size = k_size, 
                                             c_stride =c_stride))

        return l
        
                
    def forward(self, x):
        z = self.encoder(x)
        z= z.reshape(-1, *(self.input_shape))
        return torch.nn.Softmax(dim=1)(z)

#%%    

class conv_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, **kwargs):
        super(conv_encoder, self).__init__()
        self.latent_dim = latent_dim
        self.alphabet_layer_ini = kwargs["layer_ini"]
        #import pdb;pdb.set_trace()
        num_convs = 3
        #pdb.set_trace()
        self.conv1_out_lenght = input_shape[0]

        for i in range(0,num_convs):
            self.conv1_out_lenght = calculate_features_cat_size(self.conv1_out_lenght, c_pad = 1, dil = 1, k_size = 3, c_stride =1)

        self.encoder_mu = nn.Sequential(
            #nn.BatchNorm1d(input_shape[0]),
            nn.Conv1d(self.alphabet_layer_ini, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
            #nn.Linear(64*1*self.alphabet_layer_ini, 128),
            #nn.Linear(384, 128),
            nn.Linear(64*1*self.conv1_out_lenght, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)

        )
        self.encoder_var = nn.Sequential(
            #nn.BatchNorm1d(input_shape[0]),
            nn.Conv1d(self.alphabet_layer_ini, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),   #64
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
            #nn.Linear(64*1*self.alphabet_layer_ini, 128),
            #nn.Linear(384, 128),
            nn.Linear(64*1*self.conv1_out_lenght, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.Softplus()
        )
        #self.encoder_mu.apply(self._init_weights)
        #self.encoder_var.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode = 'fan_in', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        #pdb.set_trace()
        ''' TO DON'T ADD THE LINEAR LAYERS INTO THE NETWORK IS A HUUGEEEE MISTAKE, THOSE ARE NOT GOING TO BE INCLUDED INTO THE OPT IF I CONTINUE
        TO DO IT LIKE THAT, FIX!!!!!!!!!!!!!!!!!!!!'''
        xx = x.permute(0,2,1)
        z_mu = self.encoder_mu(xx)
        #z_mu = nn.Linear(z_mu.shape[1],self.latent_dim)(z_mu)

        z_var = self.encoder_var(xx)
        #z_var = nn.Softplus()(nn.Linear(z_var.shape[1],self.latent_dim)(z_var))
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