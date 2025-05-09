#!/bin/bash

# Setup
export NCCL_TESTS_DIR=$SCRATCH/nccl-testing
ml nccl

# Build
set -x
cd $NCCL_TESTS_DIR
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make -j $N MPI=1 CC=cc CXX=CC
