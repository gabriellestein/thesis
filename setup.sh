#!/bin/bash

# python3 -m venv .venv

# source .venv/bin/activate

# python3 -m pip install -r requirements.txt
# pip install flash-attn --no-build-isolation

# pip freeze > requirement.txt

# nohup /usr/bin/python3 /home/steinga20/data/thesis/code/controller.py &

# install mini conda
# install cudatoolkit

# conda activate thesis
# install pytorch in conda
# python code/controller.py > output.log 2>&1
# bash run_controller.sh


# Create conda environment with PyTorch
conda create -n thesis
conda activate thesis
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install requirements from requirements.txt
path_to_file="./requirements.txt"
if [ -f "$path_to_file" ]; then
    echo "Installing requirements from $path_to_file"
    pip install -r "$path_to_file"
else
    echo "Error: requirements.txt file not found at $path_to_file"
    exit 1
fi

echo "Setup complete. Activate the environment using 'conda activate my_env'"

