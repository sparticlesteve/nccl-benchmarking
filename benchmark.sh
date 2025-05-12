#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH -C gpu
#SBATCH --account=nstaff
#SBATCH -q regular
#SBATCH --nodes=256
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32

set -euo pipefail

echo "JobID: $SLURM_JOBID"
date

# Select software environment
ENV_TYPE="${ENV_TYPE:-module}"  # or "container"
ENV_VERSION="${ENV_VERSION:-default}"

# Benchmark configs
BENCHMARK_EXE="${BENCHMARK_EXE:-all_reduce_perf}"  # or allgather, reducescatter
NODE_COUNTS="${NODE_COUNTS:-2 4}" # 8 16 32 64 128 256}"

# Optional alt_read toggle
USE_ALT_READ="${USE_ALT_READ:-false}"  # true or false to enable alt_read settings

# Set up output
export OUTDIR="$SCRATCH/nccl-testing/nccl-benchmarking/$SLURM_JOBID"
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
NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-$SCRATCH/nccl-testing/nccl-tests}
if [ ! -d "$NCCL_TESTS_DIR" ]; then
    echo "NCCL tests directory not found. Cloning..."
    mkdir -p "$NCCL_TESTS_DIR"
    git clone https://github.com/NVIDIA/nccl-tests.git $NCCL_TESTS_DIR
fi
if [ ! -f "$NCCL_TESTS_DIR/build/$BENCHMARK_EXE" ]; then
    echo "NCCL tests executable not found. Building..."
    cd "$NCCL_TESTS_DIR"
    make -j 16 MPI=1 CC=cc CXX=CC
    cd -
fi

# Run benchmark
exe="$NCCL_TESTS_DIR/build/$BENCHMARK_EXE"

common_args="-b 32K -e 4G -d float -G 1 -f 2 -g 1"

for nn in $NODE_COUNTS; do
    echo "Running $BENCHMARK_EXE on $nn nodes"
    logfile="$OUTDIR/${BENCHMARK_EXE%_perf}_nodes_${nn}_out.log"
    srun -u --cpu-bind=none --nodes=$nn --ntasks-per-node=4 "$LAUNCH_CMD" "$exe" $common_args 2>&1 | tee "$logfile"
done