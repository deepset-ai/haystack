#!/usr/bin/env bash

# Purpose : Release a new docs version
# Input: New docs version

# Create folder for new docs veresion
mkdir "$1"

cp -ar make.bat Makefile _src static templates "$1"