#!/bin/bash

#SBATCH --array=44-58
#SBATCH --time=24:00:00
#SBATCH --partition=upex-beamtime
#SBATCH --reservation=upex_002601
#SBATCH --export=ALL
#SBATCH -J crop
#SBATCH -o .%j.out
#SBATCH -e .%j.out

# Change the runs to process using the --array option on line 3

PREFIX=/gpfs/exfel/exp/SQS/202102/p002601

source /etc/profile.d/modules.sh
source ${PREFIX}/scratch/user/ayyerkar/source_this_at_euxfel

python ../crop_dragonfly.py ${SLURM_ARRAY_TASK_ID}

