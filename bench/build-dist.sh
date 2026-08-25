#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bench/dist
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o bench/dist/kiwi-linux-amd64 ./cmd/kiwi
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -o bench/dist/kiwi-linux-arm64 ./cmd/kiwi
