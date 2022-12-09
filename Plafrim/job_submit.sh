#!/bin/bash
#SBATCH --job-name=PythonTest         # Job name
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=00:05:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log

# Load modules
module purge  # unload all modules
module load language/python/3.9.6  # load python 3.9.6
module load slurm/14.03.0

# activate Venv
source ~/X-Riemannian/venv/bin/activate  # activate virtual environment

python3 myscript.py  # run script
