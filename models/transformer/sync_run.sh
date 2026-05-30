#!/bin/bash
set -e
REMOTE=/leonardo_scratch/large/usertrain/$HPC_USER/mlenv
rsync -az baseline.py train.py train_job.sh predict.py predict_job.sh leonardo:$REMOTE/
JID=$(ssh leonardo "cd $REMOTE && sbatch --parsable train_job.sh")
echo "submitted job $JID"
while ssh leonardo "squeue -h -j $JID" | grep -q .; do sleep 8; done
echo "---- result ----"
ssh leonardo "cat $REMOTE/slurm-$JID.out"
