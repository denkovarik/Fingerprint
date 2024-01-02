#!/bin/bash

# Update and upgrade the package lists
sudo apt-get update
sudo apt-get -y upgrade

# Install Python package 'graphviz'
sudo pip install graphviz

# Install the graphviz package
sudo apt-get -y install graphviz

# Install the xdg-utils package
sudo apt-get -y install xdg-utils

