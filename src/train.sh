# running batch jobs on devcloud
# queuing
# $ qsub -l nodes=1:gpu:ppn=2 -train.sh -l walltime=23:59:59
# checking
# $ ps -o pid,user,%mem,command ax | sort -b -k3 -r > procs.txt
# $ head procs.txt
# $ watch -n 1 qstat -n 1
# unity data
python train.py --model=../outputs/unity2_intel.h5
# udacity data
# python train.py --model=../outputs/udacity2_nvidia_inteldevcloud.h5 # --inputs=../dataset/udacity/Ch2_001/center/*.jpg
