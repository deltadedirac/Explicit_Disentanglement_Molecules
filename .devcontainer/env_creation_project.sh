#!/bin/bash

#conda env create -f conda_env.yaml 
#conda init && conda activate latent_msaprot_env
conda install pytorch=1.8.1 torchvision=0.9.1 pytorch-cuda=11.6 -c pytorch -c nvidia

conda install gpytorch -c gpytorch
conda install -c conda-forge mlflow
pip install tensorboard tensorboardx torch-tb-profiler pyfiglet
conda install -c conda-forge dpath
conda install -c bioconda logomaker
conda install -c conda-forge biopython
conda install pandas numpy tqdm confuse seaborn
conda install -n latent_msaprot_env ipykernel --update-deps --force-reinstall
conda install -c conda-forge nbconvert
pip install papermill[all]
