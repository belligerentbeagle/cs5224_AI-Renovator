#!/bin/bash
set -e

echo "🚀 Building Lambda package using Docker with uv..."

# Clean previous builds
rm -rf build lambda_function.zip

# Determine Python version from your environment (or hardcode it)
PYTHON_VERSION="3.12"

# Use AWS Lambda's official Python base image
docker run --rm \
  --entrypoint /bin/bash \
  -v "$(pwd)":/workspace \
  -w /workspace \
  public.ecr.aws/lambda/python:${PYTHON_VERSION} \
  -c "
    set -e

    echo '📥 Installing build tools...'
    microdnf install -y tar gzip zip

    echo '📥 Installing uv...'
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\"\$HOME/.local/bin:\$PATH\"

    echo '📦 Extracting and installing dependencies from pyproject.toml...'
    mkdir -p build

    # Extract only the dependencies (not the package itself) and install them
    # First, export dependencies to a temporary requirements file
    uv export --no-dev --no-hashes > /tmp/requirements.txt

    # Install the dependencies
    uv pip install -r /tmp/requirements.txt --target build/ --no-cache --python /var/lang/bin/python${PYTHON_VERSION}

    echo '📂 Copying application code...'
    cp -r app build/

    echo '🧹 Cleaning up build artifacts...'
    find build -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find build -type d -name '*.dist-info' -exec rm -rf {} + 2>/dev/null || true
    find build -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
    find build -type f -name '*.pyc' -delete 2>/dev/null || true
    find build -type f -name '*.pyo' -delete 2>/dev/null || true

    echo '📦 Creating zip package...'
    cd build
    zip -r -q ../lambda_function.zip .
    cd ..

    echo '✅ Build complete!'
  "

# Make the zip owned by you, not root
if [ "$(uname)" = "Darwin" ]; then
  sudo chown $(whoami) lambda_function.zip 2>/dev/null || chown $(id -u):$(id -g) lambda_function.zip
fi

echo "✅ Package created: $(du -h lambda_function.zip | cut -f1)"
echo "📤 Upload lambda_function.zip to your Lambda function"