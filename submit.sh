#!/bin/bash

# Using this script to capture my primary experiment launch settings

# Current experiments
OUTPUT_BASE="logs/jun4-3"
export SBATCH_OUTPUT="$OUTPUT_BASE/slurm-%j.out"
VERSIONS=("2.19.4" "2.21.5" "2.24.3")
NCCL_ALGOS=("Tree" "Ring")
if [ "$NERSC_HOST" = "muller" ]; then
    NUM_NODES=(2 4 8 16)
elif [ "$NERSC_HOST" = "perlmutter" ]; then
    NUM_NODES=(32 128 512)
else
    NUM_NODES=(2)
fi

# Loop over different NCCL_ALGO values
for algo in "${NCCL_ALGOS[@]}"; do
    export NCCL_ALGO=$algo
    echo "Submitting jobs for NCCL_ALGO=$NCCL_ALGO"
    
    # Loop over different ENV_VERSION values
    for version in "${VERSIONS[@]}"; do
        export ENV_VERSION=$version
        echo "Submitting jobs for ENV_VERSION=$ENV_VERSION"

        # Loop over different node counts
        for nodes in "${NUM_NODES[@]}"; do
            echo "Submitting jobs for $nodes nodes"

            # Submit regular job
            sbatch -N $nodes -t 10 benchmark.sh
        
            # Submit job with alt_read enabled
            USE_ALT_READ=true sbatch -N $nodes -t 10 benchmark.sh
        done
    done
done
