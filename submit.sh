#!/bin/bash

# Using this script to capture my primary experiment launch settings

# Initial small scale experiments
export SBATCH_OUTPUT="slurm-May16-%j.out"
VERSIONS=("2.19.4" "2.21.5" "2.24.3")

# Loop over different ENV_VERSION values
for version in "${VERSIONS[@]}"; do
    export ENV_VERSION=$version
    echo "Submitting jobs for ENV_VERSION=$ENV_VERSION"
    
    # Submit regular job
    sbatch -N 2 -t 10 benchmark.sh
    
    # Submit job with alt_read enabled
    USE_ALT_READ=true sbatch -N 2 -t 10 benchmark.sh
done
