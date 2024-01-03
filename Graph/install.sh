#!/bin/bash

# Update and upgrade the package lists
apt-get update
apt-get -y upgrade

# Install Python package 'graphviz'
pip3 install graphviz

# Install the graphviz package
apt-get -y install graphviz

# Install the xdg-utils package
apt-get -y install xdg-utils

pip3 install ipython
