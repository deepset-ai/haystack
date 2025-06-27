#!/bin/bash

# Usage: ./pydoc-markdown.sh [CONFIG_PATH]
#
# Generate documentation from pydoc-markdown config files.
#
# Examples:
#   ./pydoc-markdown.sh                    # Uses default path: ../config/*
#   ./pydoc-markdown.sh ../config/api/*    # Uses custom path
#   ./pydoc-markdown.sh /path/to/configs/* # Uses absolute path

set -e   # Fails on any error in the following loop

# Set default config path or use provided parameter
CONFIG_PATH="${1:-../config/*}"

cd docs/pydoc
rm -rf temp && mkdir temp
cd temp
echo "Processing config files in $CONFIG_PATH"
for file in $CONFIG_PATH ; do
    echo "Converting $file..."
    pydoc-markdown "$file"
done
