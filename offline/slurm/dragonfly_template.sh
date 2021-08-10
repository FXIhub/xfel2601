#!/bin/sh

#SBATCH -p upex
#SBATCH -t 12:00:00 
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36

#SBATCH -J <set_job_name>
#SBATCH -o .%j.out
#SBATCH -e .%j.out
#SBATCH --constraint=Gold-6240

source /etc/profile.d/modules.sh
export MODULEPATH=/home/ayyerkar/.local/modules:$MODULEPATH
module purge
module load dragonfly_slurm

export OMP_NUM_THREADS=`nproc`

mpirun --bind-to none emc -c config.ini 10
