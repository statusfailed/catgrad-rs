#!/usr/bin/env bash
set -euxo

# compile helper
gcc -o main main.c

../scripts/llvm.sh main.mlir > lowered.mlir
mlir-translate lowered.mlir --mlir-to-llvmir -o main.ll
llc -filetype=obj main.ll -o main.o
clang -shared main.o -o main.so -lm
