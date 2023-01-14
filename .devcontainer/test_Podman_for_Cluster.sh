#!/bin/bash

echo "The script for creating the container system must be executed inside .devcontainer folder present in the repo"
cd ../../ && pwd
#git clone --recursive https://github.com/deltadedirac/Explicit_Disentanglement_Molecules.git
$1 build -f ./Explicit_Disentanglement_Molecules/.devcontainer/Podmanfile -t podman_disentanglement_proteins .
$1 run --name gp_cpab_vitae_podman_test -it -v $(pwd)/Explicit_Disentanglement_Molecules/:/workspace/ podman_disentanglement_proteins /bin/bash
#jupyter nbconvert --to markdown *.ipynb #-> For getting all the images in my notebook in a way of files
#docker exec gp_cpab_vitae_podman_test \
#         papermill /workspace/tests/WW_test.ipynb /workspace/tests/myout.ipynb