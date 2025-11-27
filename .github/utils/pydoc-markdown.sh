#!/bin/bash

# Generate API documentation from pydoc-markdown config files.
# Run from repo root using hatch run docs

set -e

cd pydoc
rm -rf temp && mkdir temp
cd temp

for file in ../*.yml ; do
    echo "Converting $file..."
    pydoc-markdown "$file"
done
