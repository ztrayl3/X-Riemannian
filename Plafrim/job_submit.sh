#!/usr/bin/env bash
#SBATCH -J context_HPC # name of job
#SBATCH --nodelist=sirocco07 # Ask for sirocco nodes (if less tasks than nodes then slurm adjusts list automatically)
#SBATCH -t 2-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -o ./logs/slurm/sirocco07.out # standard output message
#SBATCH -e ./logs/slurm/sirocco07.err # output error message

# Load modules
module purge
module load language/python/3.9.6

echo "=====my job informations ===="
echo "Node List: " $SLURM_NODELIST
echo "my jobID: " $SLURM_JOB_ID
echo "Partition: " $SLURM_JOB_PARTITION
echo "submit directory:" $SLURM_SUBMIT_DIR
echo "submit host:" $SLURM_SUBMIT_HOST
echo "In the directory: `pwd`"
echo "As the user: `whoami`"

srun -N1 -n1 -c1 --exclusive python3 plafrim_launcher.py --path ./jobs/main_00016638510971749672960/00016638515133901285376