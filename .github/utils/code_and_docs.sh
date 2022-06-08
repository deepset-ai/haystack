#!/bin/bash

echo "========== Apply Black ========== "
black .
echo

echo "========== Convert tutorial notebooks into webpages ========== "
python .github/utilsconvert_notebooks_into_webpages.py
echo

echo "========== Generate OpenAPI docs ========== "
python .github/utils/generate_openapi_specs.py
echo

echo "==========  Generate JSON schema ========== "
python .github/utils/generate_json_schema.py
echo

echo "==========  Generate the API documentation ========== "
set -e   # Fails on any error in the following loop
cd docs/_src/api/api/
for file in ../pydoc/* ; do
    echo "Processing" $file
    pydoc-markdown "$file"
done
echo 
