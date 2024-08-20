#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to that directory
cd "$SCRIPT_DIR"

# Run Python scripts
echo -e "Running All Tests:\n"

echo -e "Running Unit Tests\n"

echo "Running General Unit Tests"
python3 ./UnitTests/generalTests.py

echo "Running Node Unit Tests"
python3 ./UnitTests/nodeTests.py

echo "Running SharedConv2d Unit Tests"
python3 ./UnitTests/sharedConv2dTests.py

echo "Running SharedLinear Unit Tests"
python3 ./UnitTests/sharedLinearTests.py

echo "Running Graph Unit Tests"
python3 ./UnitTests/graphTests.py

echo -e "Running Functional Tests\n"

echo -e "Running Graph Functional Tests"
python3 ./FunctionalTests/graphTests.py

echo -e "Running ENAS Functional Tests"
python3 ./FunctionalTests/enasTests.py

echo -e "Running Visual Tests\n"

echo -e "Running test for Sample Architecture Human"
python3 ./VisualValidationTests/testSampleArchitectureHuman.py


echo -e "\nAll Tests Completed"

