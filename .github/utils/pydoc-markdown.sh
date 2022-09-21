#!/bin/bash

set -e   # Fails on any error in the following loop
export PYTHONPATH=$PWD/docs/pydoc # Make the renderers available to pydoc
cd docs/_src/api/api/
mkdir temp
cd temp
for file in ../../pydoc/* ; do
    pydoc-markdown "$file"
done
