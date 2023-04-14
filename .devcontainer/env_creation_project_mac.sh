#!/bin/bash

#conda create -y -n latent_msaprot_env python=3.8 && conda activate latent_msaprot_env

conda install pytorch torchvision torchaudio -c pytorch -y 
#conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch -y


conda install gpytorch -c gpytorch -y 
conda install -c conda-forge mlflow -y 
pip install tensorboard tensorboardx torch-tb-profiler pyfiglet
conda install -c conda-forge dpath -y 
conda install -c bioconda logomaker -y 
conda install -c conda-forge biopython -y 
conda install pandas numpy tqdm confuse seaborn -y 
conda install -n latent_msaprot_env ipykernel --update-deps --force-reinstall -y 
conda install -c conda-forge nbconvert -y 
pip install papermill[all]
