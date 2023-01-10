# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0.201.4-3

# [Choice] Node.js version: none, lts/*, 16, 14, 12, 10
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment to install a different version of Python than the default
# RUN conda install -y python=3.6 \
#     && pip install --no-cache-dir pipx \
#     && pipx reinstall-all

# It is necessary to specify version 0.91 in torchvision cause otherwise it generates 
# and error related to the absense of a library called torch.ao, which influence the calculations
RUN /opt/conda/bin/conda install pytorch torchvision=0.9.1 cpuonly -c pytorch-lts
RUN /opt/conda/bin/conda install gpytorch -c gpytorch
RUN /opt/conda/bin/conda install -c conda-forge mlflow
RUN /opt/conda/bin/pip install tensorboard tensorboardx torch-tb-profiler pyfiglet
RUN /opt/conda/bin/conda install -c conda-forge dpath
RUN /opt/conda/bin/conda install -c bioconda logomaker
RUN /opt/conda/bin/conda install -c conda-forge biopython
RUN /opt/conda/bin/conda install pandas numpy tqdm confuse seaborn
RUN /opt/conda/bin/conda install -n base ipykernel --update-deps --force-reinstall
RUN /opt/conda/bin/conda install -c conda-forge nbconvert
RUN pip install papermill[all]

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
