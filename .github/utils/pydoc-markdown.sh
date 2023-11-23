#!/bin/bash

set -e   # Fails on any error in the following loop
export PYTHONPATH=$PWD/docs/pydoc # Make the renderers available to pydoc
cd docs/pydoc
rm -rf temp && mkdir temp
cd temp
for file in ../config/* ; do
    echo "Converting $file..."
    pydoc-markdown "$file"
done
# render preview markdown docs
cd ..
rm -rf temp-preview && mkdir temp-preview
cd temp-preview
for file in ../config-preview/* ; do
    echo "Converting $file..."
    pydoc-markdown "$file"
done
