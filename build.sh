#!/bin/bash

# Select software environment
ENV_TYPE="${ENV_TYPE:-module}"  # or "container"
ENV_VERSION="${ENV_VERSION:-default}"

# Optional alt_read toggle
USE_ALT_READ="${USE_ALT_READ:-false}"  # true or false to enable alt_read settings

# Set up output
export OUTDIR="$SCRATCH/nccl-benchmarking/logs/$SLURM_JOBID"
mkdir -p "$OUTDIR"

# Load environment
# source ./env/load_env.sh "$ENV_TYPE" "$ENV_VERSION"
if [ "$ENV_TYPE" == "module" ]; then    
    echo "Loading module environment: nccl/$ENV_VERSION"
    module load nccl/"$ENV_VERSION"
    LAUNCH_CMD=""
elif [ "$ENV_TYPE" == "container" ]; then
    echo "Loading container environment: $ENV_VERSION"
    # Setup container launch syntax here
    LAUNCH_CMD="shifter --image=$ENV_VERSION --module=gpu,nccl-plugin"
else
    echo "Unknown load type: $ENV_TYPE"
    exit 1
fi

# module load cudatoolkit/12.4
# module unload craype-accel-nvidia80

export MPI_HOME=$CRAY_MPICH_DIR
export NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
export NPROC=10
export MPICC=CC
# export CC=gcc #cc
# export CXX=g++ #CC

# Apply environment settings
# Default settings that are always applied
export MPICH_GPU_SUPPORT_ENABLED=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216

# Apply alt_read settings if enabled
if [ "$USE_ALT_READ" == "true" ]; then
    export FI_CXI_RDZV_PROTO=alt_read
    export FI_CXI_RDZV_GET_MIN=0
    echo "Alt_read settings enabled"
fi

echo "Environment settings:"
env | grep -E '^FI_|^NCCL_'

# Build NCCL tests if needed
NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-$SCRATCH/nccl-benchmarking/builds/$ENV_VERSION/nccl-tests}
# Force clean
rm -rf $NCCL_TESTS_DIR
if [ ! -d "$NCCL_TESTS_DIR" ]; then
    echo "NCCL tests directory not found. Cloning..."
    mkdir -p "$NCCL_TESTS_DIR"
    git clone https://github.com/NVIDIA/nccl-tests.git $NCCL_TESTS_DIR
fi
if [ ! -f "$NCCL_TESTS_DIR/build/$BENCHMARK_EXE" ]; then
    echo "NCCL tests executable not found. Building..."
    cd "$NCCL_TESTS_DIR"
    make -j $NPROC MPI=1 CC=cc CXX=CC
    cd -
fi

# # Setup
# export NCCL_TESTS_DIR=$SCRATCH/nccl-testing
# ml nccl

# # Build
# set -x
# cd $NCCL_TESTS_DIR
# git clone https://github.com/NVIDIA/nccl-tests.git
# cd nccl-tests
# make -j $N MPI=1 CC=cc CXX=CC
