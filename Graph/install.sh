#!/bin/bash

# Update and upgrade the package lists
apt-get update
apt-get -y upgrade

# Install packages to create and render graphs
pip3 install graphviz
apt-get -y install graphviz
apt-get install feh

# Install the xdg-utils package
apt-get -y install xdg-utils

pip3 install ipython

# Install PyTorch
pip install torch torchvision torchaudio

