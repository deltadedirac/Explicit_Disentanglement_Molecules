#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:31:08 2019

@author: nsde
"""

#%%
import argparse, os, sys

#%%
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=0, help='experiment to run')
    args = parser.parse_args()
    return args

#%%
experiments = [
    
    ## Stability experiments    
    # Affine stn    
    "python main.py --model vitae_ci --n_epochs 500 --warmup 250 --batch_size 256 --dataset mnist --logdir a4 --stn_type affine --lr 1e-4"
    ]

if __name__ == '__main__':
    args = argparser()
    command = experiments[args.n]
    try:
        os.system(command)
    except Exception as e:
        print("Incountered error in command", args.n)
        print(e)
        sys.exit()

