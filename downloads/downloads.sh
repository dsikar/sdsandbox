#! /bin/bash

# Bash script to run batch job on Camber server (@city) - see master branch for full notes
# To run on camber
# $ sbatch traincnn.sh
# To check
# $ squeue
# $ sinfo
# To log into node:
# ssh <node> # e.g. ssh africa, etc
#SBATCH --job-name="NVIDIA x Udacity data model"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.sikar@city.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output job%J.out
#SBATCH --error job%J.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1

# module load cuda/10.0
# download overcast driving data 124G compressed
wget http://bit.ly/udacity-dataset-2-2
