#!/bin/bash
# This script dynamically sets the LD_LIBRARY_PATH to include the MKL library.

# Define the base directory for the search.
BASE_DIR="$HOME/.cache"

# Find the MKL library directory.
MKL_PATH=$(find $BASE_DIR -type f -name "libmkl_intel_lp64.so.2" -print -quit)

if [ -z "$MKL_PATH" ]; then
    echo "MKL library not found in $BASE_DIR"
    exit 1
fi

# Extract the directory path from the full file path.
MKL_DIR=$(dirname "$MKL_PATH")

# Update the LD_LIBRARY_PATH.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MKL_DIR

# To check path from export command
# echo $LD_LIBRARY_PATH

echo "Updated LD_LIBRARY_PATH with MKL library path: $MKL_DIR"

magic shell
