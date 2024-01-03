#!/bin/bash

apt update
apt install python3-pip
pip3 install progress

# Run the install script
./Graph/install.sh
