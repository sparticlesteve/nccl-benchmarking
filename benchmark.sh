#!/bin/bash -l
#SBATCH --time=2:00:00
#SBATCH -C gpu
#SBATCH --account=nstaff
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -q regular

echo jobID: $SLURM_JOBID
date
module load nccl
reduce="nccl-tests/build/all_reduce_perf -b 32K -e 8G -d float -G 1 -f 2 -g 1"
gather="nccl-tests/build/all_gather_perf -b 32K -e 8G -d float -G 1 -f 2 -g 1"
reducescatter="nccl-tests/build/reduce_scatter_perf -b 32K -e 8G -d float -G 1 -f 2 -g 1"

outdir=$SCRATCH/benchmark_nccl/$SLURM_JOBID
mkdir -p $outdir

# Settings
export MPICH_GPU_SUPPORT_ENABLED=0
export FI_MR_CACHE_MONITOR=userfaultfd
#export FI_CXI_RX_MATCH_MODE=software
#export FI_CXI_RDZV_PROTO=alt_read
#export NCCL_ALGO=Tree
#export NCCL_ALGO=Ring

echo "FI and NCCL env var settings:"
env | grep NCCL
env | grep FI_

for nn in 1 2 4 8 16 32 64 128
do
    for gpusper in 4
    do
        if [ "$nn" -eq 1 ] && [ "$gpusper" -eq 1 ]; then
            continue
        fi
        echo Running on $nn nodes with $gpusper GPUs per node
        for run in 1 2 3 4 5
        do
            logfile="$outdir/allreduce_nodes_${nn}_gpuspernode_${gpusper}_run${run}_out.log"
            srun -u --cpu-bind=none --nodes=$nn --ntasks-per-node=$gpusper $reduce 2>&1 | tee $logfile
            #logfile="$outdir/allgather_nodes_${nn}_gpuspernode_${gpusper}_out.log"
            #srun -u --cpu-bind=none --nodes=$nn --ntasks-per-node=$gpusper $gather 2>&1 | tee $logfile
            #logfile="$outdir/reducescatter_nodes_${nn}_gpuspernode_${gpusper}_out.log"
            #srun -u --cpu-bind=none --nodes=$nn --ntasks-per-node=$gpusper $reducescatter 2>&1 | tee $logfile
        done
    done
done
