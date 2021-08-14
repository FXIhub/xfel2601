#!/bin/bash

#SBATCH --array=196
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH -J integrate
#SBATCH -o .%j.out
#SBATCH -e .%j.out
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_002601

# Change the runs to process using the --array option on line 3
# DONT FORGET TO SET THE RIGHT DARK RUN

source /etc/profile.d/modules.sh
module purge
module load anaconda3

dark_run=195
run=`printf %.4d "${SLURM_ARRAY_TASK_ID}"`
#cell_max=`h5ls -d /gpfs/exfel/exp/SPB/202101/p002827/raw/r${run}/RAW-R${run}-DA01-S00000.h5/RUN/SPB_XTD9_XGM/XGM/DOOCS/pulseEnergy/numberOfSa1BunchesActual/value|tail -n 1 |awk '{print $2+2}'`
cell_max=120
cells=0,${cell_max},1

mpirun -mca btl_tcp_if_include ib0 python ../integrate.py $run $dark_run -c $cells --num_cells=200
