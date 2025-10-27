# Isaac Sim Installation:

check GLIBC>= 2.35 by `ldd --version`

set up python env:
`uv venv --python 3.11 env_isaac`
`source env_isaac/bin/activate`

upgrade pip
`uv pip install --upgrade pip`

Install pytorch for CUDA 12.8:
`uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128`

Install isaacsim via pip:
`uv pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com`

# Isaac Lab Installation

after cloning Isaaclab

`./isaaclab.sh --install rl_games`