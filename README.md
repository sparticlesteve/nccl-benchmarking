# nccl-benchmarking

Simple benchmarking scripts to sweep over various sizes of NCCL collectives.

 1) Submit the runs with `benchmark.sh` sbatch script (modify as needed to change number of nodes, runs per trial, etc).
    This drops results into an output directory in `$SCRATCH`

 2) Analyze the results with `analyze_benchmark.ipynb`. This notebook will look in the above scratch directory 
    (specified by job ID) and parse the results for analysis/plotting. It supports fitting parametric curves
    (e.g. sigmoids) to the data which can be saved to json for use in performance modeling

