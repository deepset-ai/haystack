#!/usr/bin/env bash

# Purpose : Release a new docs version
# Input: New docs version

# Create folder for new docs veresion
mkdir "$1"

cp -a make.bat Makefile _src "$1"
