#!/bin/bash

set -e   # Fails on any error in the following loop
cd docs/pydoc
rm -rf temp && mkdir temp
cd temp
for file in ../config/* ; do
    echo "Converting $file..."
    pydoc-markdown "$file"
done
