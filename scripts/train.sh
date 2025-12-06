#!/bin/bash

TIME=$(date "+%Y%m%d%H%M%S")

# cuda available setting
export CUDA_VISIBLE_DEVICES=0

DATASET="all"
MODLE_TYPE="Deep3DS_alpha"
LOSS_FUNC="mse"
RELA_LAYERS=3
N_HEADS=8
DROPOUT=0.2
EXP_NAME="train_NoDataNorm_${MODLE_TYPE}_${N_HEADS}_${DROPOUT}_${RELA_LAYERS}_${DATASET}_${TIME}_${LOSS_FUNC}"

python train_deep3ds.py \
    --base_dir logs \
    --exp_name $EXP_NAME \
    --category $DATASET \
    --data_dir path/to/train_dataset \
    --data_dir_test path/to/test_dataset \
    --num_workers 8 \
    --device cuda:0 \
    --epochs 3000 \
    --batch_size 32 \
    --epoch_save 25 \
    --val_every_epochs 1 \
    --model_type $MODLE_TYPE \
    --rela_layers $RELA_LAYERS \
    --n_heads $N_HEADS \
    --dropout $DROPOUT \
    --render_pose_type "qua_artifact" \
    --sel_thre 0.75
