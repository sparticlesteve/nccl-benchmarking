#!/bin/bash

# Using this script to capture my primary experiment launch settings

# Initial small scale experiments
export SBATCH_OUTPUT="slurm-Jun4-%j.out"
VERSIONS=("2.19.4" "2.21.5" "2.24.3")
NUM_NODES=(2 4 8)

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
