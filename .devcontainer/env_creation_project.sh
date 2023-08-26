#!/bin/bash

#conda env create -f conda_env.yaml 
#conda init && conda activate latent_msaprot_env
#conda install pytorch=1.8.1 torchvision=0.9.1 pytorch-cuda=11.6 -c pytorch -c nvidia
#conda install gpytorch -y -c gpytorch

#conda env create -f conda_env.yaml && conda activate PGM_latent_alignment2
#conda install pytorch==1.8.1 torchvision==0.9.1 pytorch-cuda=11.0 -c pytorch-nightly -c nvidia


#pip3 install torch==1.8.1 torchvision==0.9.1 --extra-index-url https://download.pytorch.org/whl/nightly/cu110
conda install pytorch=1.8.1 torchvision=0.9.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge gpytorch
conda install -c conda-forge pandas numpy scipy
#pip3 install  gpytorch==1.5.1
#conda install -y -c conda-forge mlflow

conda install -c "conda-forge/label/cf202003" tensorboard
conda install -c conda-forge tensorboardx
pip3 install pyfiglet ipdb
#pip3 install tensorboard tensorboardx torch-tb-profiler pyfiglet #scikit-learn-intelex
conda install -y -c conda-forge dpath
conda install -y -c bioconda logomaker
conda install -y -c conda-forge biopython
conda install -y tqdm confuse seaborn
conda install -y ipykernel
conda install -c conda-forge matplotlib
pip3 install papermill[all]
#pip3 uninstall kiwisolver
#pip3 install -U kiwisolver