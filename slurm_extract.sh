#!/bin/bash
#SBATCH --job-name VGGS32
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --cpus-per-task=9
#SBATCH --mem 96GB
#SBATCH --constraint=[v100]

echo `hostname`
# conda activate refineloc
# module load anaconda3
source activate torch1.3

DIR=$HOME/VGGSoundFeatures
cd $DIR
echo `pwd`

LOG_DIR=./logs/
DEVICE=cuda:0

mkdir -p $LOG_DIR
WINDOW_SIZE=32
FEAT_TYPE='audio_stride_4_'$WINDOW_SIZE
python test.py --batch_size 1024 --window_size $WINDOW_SIZE
python npy2hdf5.py --feat_type $FEAT_TYPE