#!/bin/bash
TIME=$(date "+%Y%m%d%H%M%S")

DATASET="artifact"
SAMPLER="PC_origin"
MODEL_TYPE="Deep3DS_alpha"
SEL_THRE=0.75
N_HEADS=4
DROPOUT=0.5
EXP_NAME="eval_${MODEL_TYPE}_${DATASET}_${TIME}_${SAMPLER}_${SEL_THRE}"

python eval_deep3ds.py --base_dir logs \
    --exp_name $EXP_NAME \
    --category $DATASET \
    --data_dir path/to/dataset \
    --num_workers 8 \
    --device cuda:0 \
    --model_path ./pretrained/... \
    --sel_sampler $SAMPLER \
    --max_sampling 150 \
    --model_type $MODEL_TYPE \
    --sel_thre $SEL_THRE \
    --n_heads $N_HEADS \
    --dropout $DROPOUT \
    --render_pose_type "qua_artifact" \
    --val_batch_size 4
