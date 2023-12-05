FROM mambaorg/micromamba:1.5.3

COPY --chown=$MAMBA_USER:$MAMBA_USER docs/environment.yml /tmp/env.yaml

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

USER root
RUN apt-get update && apt-get install -y build-essential git libglib2.0-0 libgl1 ffmpeg libsm6 libxext6

USER $MAMBA_USER
WORKDIR /code
COPY timm-0.5.4.tar /code/
RUN pip install /code/timm-0.5.4.tar
COPY ctranspath_model.pt /code/

ARG BRANCH=single-slide-cli
ADD https://api.github.com/repos/msk-mind/CLAM/git/refs/heads/$BRANCH version.json
RUN git clone -b $BRANCH https://github.com/msk-mind/CLAM /code/CLAM
RUN git clone https://github.com/Xiyue-Wang/TransPath.git /code/TransPath
COPY docker/extract_features_config.yaml /code/CLAM/

ENV PATH="$PATH:/code/CLAM"
