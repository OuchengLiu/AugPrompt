#!/bin/bash

# custom config
DATA=../DATA
TRAINER=CoCoOp

DATASET=$1
SEED=$2

CFG=vit_b16_c16_ep10_batch1
SHOTS=16

DIR=output/all/test/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/all/train/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch 10 \
    --eval-only
fi
