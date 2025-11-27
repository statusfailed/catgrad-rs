#!/usr/bin/env bash

# Lower all the way to LLVM
mlir-opt $1 \
  --convert-elementwise-to-linalg \
  --linalg-fuse-elementwise-ops \
  --one-shot-bufferize=bufferize-function-boundaries \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --expand-strided-metadata \
  --lower-affine \
  --finalize-memref-to-llvm \
  --convert-math-to-llvm \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts
