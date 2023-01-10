#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:48:18 2018

@author: nsde
"""

#from .vitae_ci_gp import VITAE_CI
#from src.unsuper.unsuper.models.vae import VAE
#from src.unsuper.unsuper.models.vitae_ui import VITAE_UI
from src.models.vitae_ci_gp_tmp import VITAE_CI

#%%
def get_model(model_name):
    models = {#'vae': VAE,
              'vitae_ci': VITAE_CI#,
              #'vitae_ui': VITAE_UI,
              }
    assert (model_name in models), 'Model not found, choose between: ' \
            + ', '.join([k for k in models.keys()])
    return models[model_name]