#!/bin/bash
#PBS -N CogD
#PBS -P ee
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=12:00:00

## SPECIFY JOB NOW
JOBNAME=CogKeras
CURTIME=$(date +%Y%m%d%H%M%S)

PRETAM=0.3
LOGNUM=LF_Mayan_PretAustro_TarakaEmbed_$PRETAM
CODEDIR=/home/ee/btech/ee1130798/BTP/Code/SiameseConvNet
PYTHON=/home/ee/btech/ee1130798/anaconda/bin/python
cd $CODEDIR

THEANO_FLAGS='lib.cnmem=0.5' $PYTHON $CODEDIR/PretCoAtt.py data/Mayan_DF1.pkl data/Austro_DF1.pkl -ptamount $PRETAM -init_taraka True -lstm 50 -embd 10 -l2 0.02 -epochs 20 > $CODEDIR/RerunLogs/log$LOGNUM.txt 2> $CODEDIR/RerunLogs/err$LOGNUM.txt

