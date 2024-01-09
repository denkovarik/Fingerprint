#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to that directory
cd "$SCRIPT_DIR"

# Run Python scripts
echo -e "Running All Tests:\n"

echo "Running Node Tests"
python3 ./nodeTests.py

echo -e "\nRunning Graph Tests"
python3 ./graphTests.py

# Optionally, return to the original directory
# cd -

echo -e "\nAll Tests Completed"

