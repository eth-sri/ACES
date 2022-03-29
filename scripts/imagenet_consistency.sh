IMAGENET_DIR=/mnt/sdb/imagenet CUDA_VISIBLE_DEVICES=5 \
    taskset -c 0-11 \
    python certify_ace.py \
    imagenet \
    ./pretrained-models/imagenet/consistency/noise_1.00/checkpoint.pth.tar \
    1.00 \
    ./data/imagenet/consistency/noise_1.00/imagenet_consistency_100 \
    --skip 100 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 500;
IMAGENET_DIR=/mnt/sdb/imagenet CUDA_VISIBLE_DEVICES=5 \
    taskset -c 0-11 \
    python certify_ace.py \
    imagenet \
    ./pretrained-models/imagenet/consistency/noise_0.50/checkpoint.pth.tar \
    0.50 \
    ./data/imagenet/consistency/noise_0.50/imagenet_consistency_050 \
    --skip 100 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 500;
IMAGENET_DIR=/mnt/sdb/imagenet CUDA_VISIBLE_DEVICES=5 \
    taskset -c 0-11 \
    python certify_ace.py \
    imagenet \
    ./pretrained-models/imagenet/consistency/noise_0.25/checkpoint.pth.tar \
    0.25 \
    ./data/imagenet/consistency/noise_0.25/imagenet_consistency_025 \
    --skip 100 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 500;
