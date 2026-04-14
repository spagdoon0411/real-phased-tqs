#!/usr/bin/env bash
set -euo pipefail

# Install Node.js and npm if not present
if ! command -v npm &>/dev/null; then
    read -r -p "npm not found. Install Node.js and npm via apt? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    sudo apt-get update -qq
    sudo apt-get install -y nodejs npm
fi

# Install prettier globally
npm install -g prettier

PRETTIER="$(npm prefix -g)/bin/prettier"
echo "Prettier $("$PRETTIER" --version) installed at $PRETTIER"
