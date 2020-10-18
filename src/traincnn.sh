#! /bin/bash

# Bash script to run batch job on Camber server (@city)
# To run on camber
# $ sbatch traincnn.sh
# To check
# $ watch -n 1 squeue
# $ sinfo
# $ tail -f job<jobid>.out
# To log into node:
# $ ssh <node> # e.g. ssh africa, etc
# To check memory usage on node:
# $ ps -o pid,user,%mem,command ax | sort -b -k3 -r > procs.txt
# $ head procs.txt

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
echo started job: $(date "+%y%m%d.%H%M%S.%3N")
echo

python3 train.py --model=../outputs/unity5_nvidia_camber.h5

echo
echo finished job: $(date "+%y%m%d.%H%M%S.%3N")
echo
