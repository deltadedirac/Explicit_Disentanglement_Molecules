#!/bin/bash

podman build -t explicit_disentanglement_proteins -f Podmanfile
podman run --name gp_cpab_vitae_podman_test -it -v         explicit_disentanglement_proteins /bin/bash