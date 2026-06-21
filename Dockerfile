FROM mambaorg/micromamba:1.5.10

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/environment.yml /tmp/environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt /tmp/requirements-visual.txt

RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba run -n base pip install --no-cache-dir -r /tmp/requirements-visual.txt \
    && micromamba clean --all --yes

WORKDIR /workspace/fine-tune-visual

ENV PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    FACE_LANDMARKER_MODEL=/workspace/fine-tune-visual/pretrained/mediapipe/face_landmarker.task

ENTRYPOINT ["micromamba", "run", "-n", "base", "--"]
CMD ["python", "--version"]
