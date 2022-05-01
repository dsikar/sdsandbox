#!/bin/bash
#SBATCH --job-name=nvidia2		# Job name
#SBATCH --mail-type=ALL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=daniel.sikar@city.ac.uk         # Where to send mail	
#SBATCH --nodes=4                                # Run on 4 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=48
#SBATCH --exclusive                              # Exclusive use of nodes
#SBATCH --mem=0                                  # Expected memory usage (0 means use all available memory)
#SBATCH --time=24:00:00                          # Time limit hrs:min:sec
#SBATCH -e outputs/runpy%j.e
#SBATCH -o outputs/runpy%j.o           # Standard output and error log [%j is replaced with the jobid]

#enable modules
source /opt/flight/etc/setup.sh 
# deactivate any previously activated environments
flight env deactivate
# activate gridware to get python3 
flight env activate gridware
# load required modules
module add libs/numpy_python39

#remove any unwanted modules 
#module purge

srun hostname -s > outputs/hosts.$SLURM_JOB_ID

#Command line to run task

python3 /users/aczd097/git/sdsandbox/src/utils/utils.py --filepath='/users/aczd097/localscratch/dataset/unity/log_sample/logs_Mon_Jul_13_08_29_01_2020/' --model='nvidia2' --mask='*.jpg'
