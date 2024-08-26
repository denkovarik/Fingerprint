#!/bin/bash

# The following script should download and install the CUDA Toolkit 12.6 for 
# the Linux OS x86_64 running Ubuntu 22.04. Be sure to run this script as root,
# and you may need to restart the terminal or your machine when installation 
# is complete.
#
#   Operating System:   Linux
#   Architecture:       x86_64
#   Distribution:       Ubuntu
#   Version:            22.04
#   Installer Type:     deb (local)
#
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-12-6

# To install the open kernel module flavor:
apt-get install -y nvidia-open

