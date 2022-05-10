# Apply Black
black .

# Generate the API documentation
set -e   # Fails on any error in the following loop
cd docs/_src/api/api/
for file in ../pydoc/* ; do
    echo "Processing" $file
    pydoc-markdown "$file"
done
cd ../../../../

# Convert tutorial notebooks into webpages
python convert_notebooks_into_webpages.py

# Generate OpenAPI docs
python .github/utils/generate_openapi_specs.py

# Generate JSON schema
python .github/utils/generate_json_schema.py
