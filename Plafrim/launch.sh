#!/bin/bash
#SBATCH --job-name=$2_$1                                           # Job name
#SBATCH --ntasks=1                                                 # Run on a single CPU
#SBATCH --mem=32gb                                                 # Job memory request
#SBATCH --time=03:00:00                                            # Time limit hrs:min:sec
#SBATCH --output=/beegfs/ztraylor/X-Riemannian/Output/log_%j.log   # Standard output and error log

# Load modules
module purge  # unload all modules
module load language/python/3.9.6  # load python 3.9.6

# activate Venv
source /beegfs/ztraylor/X-Riemannian/venv/bin/activate  # activate virtual environment

python3 Analyses.py -s "$1" -a "$2"  # run script
