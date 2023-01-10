#!/bin/bash
#SBATCH --job-name=X_Riemann                                       # Job name
#SBATCH --ntasks=1                                                 # Run on a single CPU
#SBATCH --mem=32gb                                                 # Job memory request
#SBATCH --time=10:00:00                                            # Time limit hrs:min:sec
#SBATCH --output=/beegfs/ztraylor/X-Riemannian/Output/%j.log       # Standard output and error log

# Load modules
module purge  # unload all modules
module load language/python/3.9.6  # load python 3.9.6
module load compiler/cuda/11.6  # load cuda for DL
module load dnn/cudnn/9.0-v7.1  # load cuDNN

# activate Venv
cd /beegfs/ztraylor/X-Riemannian
source venv/bin/activate  # activate virtual environment

# for time keeping PRE
date

# first argument is subject ID second is analysis name
python3 Analyses.py -s "$1" -a "$2"  # run script

# for time keeping POST
date
