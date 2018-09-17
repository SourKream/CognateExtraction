#!/bin/bash
#PBS -N CogD
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=12:00:00

## SPECIFY JOB NOW
JOBNAME=CogKeras
CURTIME=$(date +%Y%m%d%H%M%S)

LOGNUM=AUSTRO
CODEDIR=/home/ee/btech/ee1130798/BTP/Code/SiameseConvNet
PYTHON=/home/ee/btech/ee1130798/anaconda/bin/python
cd $CODEDIR

$PYTHON $CODEDIR/CoAttModel.py data/abvd2-part2.tsv.asjp > $CODEDIR/Logs/log$LOGNUM.txt 2> $CODEDIR/Logs/err$LOGNUM.txt

