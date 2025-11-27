#!/usr/bin/env bash

# Fuse operations and bufferize
mlir-opt $1 \
  --convert-elementwise-to-linalg \
  --linalg-fuse-elementwise-ops
