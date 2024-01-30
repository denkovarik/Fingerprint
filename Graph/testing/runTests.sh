#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to that directory
cd "$SCRIPT_DIR"

# Run Python scripts
echo -e "Running All Tests:\n"

echo "Running General Unit Tests"
python3 ./UnitTests/generalTests.py

echo "Running Node Unit Tests"
python3 ./UnitTests/nodeTests.py

echo "running Graph Unit Tests"
python3 ./UnitTests/graphTests.py

echo -e "\nRunning Graph Functional Tests"
python3 ./FunctionalTests/graphTests.py


echo -e "\nAll Tests Completed"

