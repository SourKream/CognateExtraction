#!/bin/bash
#PBS -N CogDCF
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=16:00:00

## SPECIFY JOB NOW
JOBNAME=CogKerasCF
CURTIME=$(date +%Y%m%d%H%M%S)

LOGNUM=CF_Austro_CFeat
CODEDIR=/home/ee/btech/ee1130798/BTP/Code/SiameseConvNet
PYTHON=/home/ee/btech/ee1130798/anaconda/bin/python
cd $CODEDIR

THEANO_FLAGS='lib.cnmem=0.5' $PYTHON $CODEDIR/CoAtt.py data/Austro_CF_DF.pkl -conceptfeat True -lstm 40 -embd 10 -l2 0.02 -epochs 50 > $CODEDIR/CFRerunLogs/log$LOGNUM.txt 2> $CODEDIR/CFRerunLogs/err$LOGNUM.txt


