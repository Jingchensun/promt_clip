CUDA_VISIBLE_DEVICES=3 python train.py \
        --root data \
        --seed 1 \
        --trainer CoOp \
        --dataset-config-file configs/datasets/caltech101.yaml \
        --config-file configs/trainers/CoOp/rn50.yaml \
        --output-dir output/caltech101/CoOp/rn50_16shots_ori/nctx16_cscFalse_ctpend/seed1 \
        TRAINER.COOP.N_CTX 16 \
        TRAINER.COOP.CSC False \
        TRAINER.COOP.CLASS_TOKEN_POSITION end \
        DATASET.NUM_SHOTS 16
 CUDA_VISIBLE_DEVICES=3 python train.py \
        --root data \
        --seed 2 \
        --trainer CoOp \
        --dataset-config-file configs/datasets/caltech101.yaml \
        --config-file configs/trainers/CoOp/rn50.yaml \
        --output-dir output/caltech101/CoOp/rn50_16shots_ori/nctx16_cscFalse_ctpend/seed2 \
        TRAINER.COOP.N_CTX 16 \
        TRAINER.COOP.CSC False \
        TRAINER.COOP.CLASS_TOKEN_POSITION end \
        DATASET.NUM_SHOTS 16
 CUDA_VISIBLE_DEVICES=3 python train.py \
        --root data \
        --seed 3 \
        --trainer CoOp \
        --dataset-config-file configs/datasets/caltech101.yaml \
        --config-file configs/trainers/CoOp/rn50.yaml \
        --output-dir output/caltech101/CoOp/rn50_16shots_ori/nctx16_cscFalse_ctpend/seed3 \
        TRAINER.COOP.N_CTX 16 \
        TRAINER.COOP.CSC False \
        TRAINER.COOP.CLASS_TOKEN_POSITION end \
        DATASET.NUM_SHOTS 16