#!/bin/bash

$1 exec gp_cpab_vitae_podman_test bash -c 'cd /workspace/tests && papermill WW_test.ipynb myoutend1.ipynb'
$1 exec gp_cpab_vitae_podman_test bash -c 'cd /workspace/tests && jupyter nbconvert --to markdown myoutend1.ipynb'