#!/bin/bash

# Runs all tests
# This is breaking for some reason
#mojo test -I src testing

# Run Unit tests
mojo test -I src testing/UnitTests

# Run Functional tests
mojo test -I src testing/FunctionalTests
