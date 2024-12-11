#!/bin/bash

# Ensure the script is run as root
if [ "$(id -u)" != "0" ]; then
    echo "This script must be run as root" 1>&2
    exit 1
fi

# Update and install required packages
echo "Installing required packages..."
apt-get update

echo -e '\e]8;;https://docs.modular.com/mojo/manual/get-started/\aClick here to visit the Mojo Getting Started guide\e]8;;\a'

curl -ssL https://magic.modular.com/de6fc06e-facc-4d45-a112-f73b53875782 | bash

# CITTIZINS
apt-get install -y gcc g++ zlib1g-dev libtinfo-dev
apt-get install -y intel-mkl

# Optionally, source the bashrc or similar file to refresh environment variables
echo "Sourcing .bashrc to update environment settings..."
source $HOME/.bashrc  # or source $HOME/.profile or any other relevant shell initialization file

echo "If you experience any issues, please restart your terminal manually."

