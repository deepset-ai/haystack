#!/bin/bash

set -e   # Fails on any error in the following loop
export PYTHONPATH=$PWD/docs/pydoc # Make the renderers available to pydoc
cd docs/pydoc
rm -rf temp && mkdir temp
cd temp
for file in ../config/* ; do
    pydoc-markdown "$file"
done
