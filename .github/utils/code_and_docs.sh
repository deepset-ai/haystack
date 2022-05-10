# Apply Black
black .

# Convert tutorial notebooks into webpages
cd docs/_src/tutorials/tutorials/
python3 convert_ipynb.py

# Generate the API documentation
set -e   # Fails on any error in the following loop
cd docs/_src/api/api/
for file in ../pydoc/* ; do
    echo "Processing" $file
    pydoc-markdown "$file"
done

# Generate OpenAPI docs
python .github/utils/generate_openapi_specs.py

# Generate JSON schema
python .github/utils/generate_json_schema.py
