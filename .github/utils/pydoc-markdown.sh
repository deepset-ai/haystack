#!/bin/bash

set -e   # Fails on any error in the following loop
cd docs/_src/api/api/
for file in ../pydoc/* ; do
    pydoc-markdown "$file"
done
