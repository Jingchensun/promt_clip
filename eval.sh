#!/bin/bash

# custom config
DATA=data/
TRAINER=CoOp
SHOTS=16
NCTX=16
CSC=False
CTP=end
SEED=1
CFG=rn50

DATASET1=ucf101 
for DATASET2 in imagenet caltech101 oxford_pets oxford_flowers stanford_cars food101 fgvc_aircraft sun397 ucf101 dtd eurosat
do
    CUDA_VISIBLE_DEVICES=2 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET2}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir output/evaluation/${DATASET1}/${DATASET2} \
    --model-dir output/$DATASET1/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED} \
    --load-epoch 200 \
    --no-train \
    --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
done