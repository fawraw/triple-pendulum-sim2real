# Triple Pendulum — RunPod / generic GPU training image.
#
# Base: PyTorch CUDA 12.1 with cuDNN, on Ubuntu 22.04.
# Adds: MuJoCo (via pip), osmesa for headless rendering, ffmpeg, sb3-contrib.
#
# Build:
#   docker build -t fawraw/triple-pendulum:cuda121 -f runpod/Dockerfile .
#   docker push fawraw/triple-pendulum:cuda121
#
# Or use directly via RunPod template — see docs/runbook.md "Cloud GPU".
#
# Runtime entrypoint: scripts/runpod_bootstrap.sh
#   - clones / pulls the latest main from GitHub
#   - reads training stage from env (TP_STAGE_MODULE, TP_STAGE_CONFIG)
#   - runs training, pushes results JSON via pipeline_notifier (n8n webhook)
#   - tag the pod for auto-shutdown on completion

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=osmesa \
    PIP_NO_CACHE_DIR=1

# OS deps for headless mujoco rendering, git for clone, tmux for parity with CT 1018
RUN apt-get update -qq && apt-get install -y -qq \
        git tmux ffmpeg \
        libgl1 libglu1-mesa libosmesa6 libegl1 libglfw3 \
        ca-certificates curl jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps in a layer so code changes don't bust the cache.
# The repo's requirements.txt already pins torch>=2.2 — but we use the CUDA
# torch from the base image, so we install requirements with --no-deps for torch.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# We do NOT copy the repo code into the image — it's pulled fresh at pod
# start via runpod_bootstrap.sh, so the same image works for any commit.

CMD ["/bin/bash"]
