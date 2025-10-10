#!/bin/bash
# Setup local development environment for Haystack docs testing
# This script generates requirements.txt and installs dependencies

set -e  # Exit on any error

echo "🚀 Setting up local development environment for Haystack docs testing..."

# Check if we're in the right directory
if [ ! -f "scripts/generate_requirements.py" ]; then
    echo "❌ Error: Please run this script from the docs-website directory in the haystack repository"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if base dependencies are installed
echo "📦 Checking base dependencies..."
python -c "import requests, toml" 2>/dev/null || {
    echo "⚠️  Installing base dependencies (requests, toml)..."
    pip install requests toml
}

# Get Haystack version (default to main)
VERSION=${1:-main}
echo "🔍 Generating requirements.txt for Haystack version: $VERSION"

# Generate requirements.txt
python scripts/generate_requirements.py --version "$VERSION"

# Check if requirements.txt was generated
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt was not generated"
    exit 1
fi

echo "📋 Generated requirements.txt with $(wc -l < requirements.txt) dependencies"

# Install dependencies
echo "⚙️  Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete! You can now run:"
echo "   python scripts/test_python_snippets.py --verbose"
echo ""
echo "📝 Note: requirements.txt is auto-generated and should not be committed to git"
