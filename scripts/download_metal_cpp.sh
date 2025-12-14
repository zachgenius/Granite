#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="$SCRIPT_DIR/../third_party/metal-cpp"

echo "Downloading metal-cpp from Apple..."

# Create temp directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download metal-cpp (check Apple's current URL)
curl -L -o metal-cpp.zip "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip"

# Extract
unzip -q metal-cpp.zip

# Find the extracted directory (name varies by version)
EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "metal-cpp*" | head -1)

if [ -z "$EXTRACTED_DIR" ]; then
    echo "Error: Could not find extracted metal-cpp directory"
    exit 1
fi

# Copy headers to third_party
mkdir -p "$THIRD_PARTY_DIR"
cp -r "$EXTRACTED_DIR"/* "$THIRD_PARTY_DIR/"

# Cleanup
cd -
rm -rf "$TEMP_DIR"

echo "metal-cpp installed to $THIRD_PARTY_DIR"
echo "Contents:"
ls -la "$THIRD_PARTY_DIR"
