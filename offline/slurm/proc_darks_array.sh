#!/bin/bash

#SBATCH --array=7
#SBATCH --time=04:00:00
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_002601
#SBATCH --export=ALL
#SBATCH -J dark
#SBATCH -o .%j.out
#SBATCH -e .%j.out

# Change the runs to process using the --array option on line 3

PREFIX=/gpfs/exfel/exp/SQS/202102/p002601

source /etc/profile.d/modules.sh
source ${PREFIX}/scratch/user/ayyerkar/source_this_at_euxfel

run=`printf %.4d "${SLURM_ARRAY_TASK_ID}"`
python ../proc_darks.py $run -o ${PREFIX}/scratch/dark/r${run}_dark.h5
