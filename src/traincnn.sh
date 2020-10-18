#! /bin/bash

# Script source: https://moodle.city.ac.uk/pluginfile.php/1651287/mod_forum/attachment/307378/Cluster%20access%20documentation.pdf?forcedownload=1
# Bash script to run batch job on Camber server (@city)
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

module load cuda/10.0

echo
echo started Ud1 job: $(date "+%y%m%d.%H%M%S.%3N")
echo

python3 train.py --model=../outputs/ud1_nvidia.h5

echo
echo finished Ud1 job: $(date "+%y%m%d.%H%M%S.%3N")
echo

