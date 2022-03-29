IMAGENET_DIR=/mnt/sdb/imagenet CUDA_VISIBLE_DEVICES=4 \
    taskset -c 12-23 \
    python certify_ace.py \
    imagenet \
    ./pretrained-models/imagenet/smoothadv/checkpoint-PGD1step-eps512-100.pth.tar \
    1.00 \
    ./data/imagenet/smoothadv/noise_1.00/imagenet_smoothadv_100 \
    --skip 100 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 500 \
    --center_layer 1;
IMAGENET_DIR=/mnt/sdb/imagenet CUDA_VISIBLE_DEVICES=4 \
    taskset -c 12-23 \
    python certify_ace.py \
    imagenet \
    ./pretrained-models/imagenet/smoothadv/checkpoint-PGD1-255-050.pth.tar \
    0.50 \
    ./data/imagenet/smoothadv/noise_0.50/imagenet_smoothadv_050 \
    --skip 100 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 500 \
    --center_layer 1;
IMAGENET_DIR=/mnt/sdb/imagenet CUDA_VISIBLE_DEVICES=4 \
    taskset -c 12-23 \
    python certify_ace.py \
    imagenet \
    ./pretrained-models/imagenet/smoothadv/checkpoint-DDN2-512-025.pth.tar \
    0.25 \
    ./data/imagenet/smoothadv/noise_0.25/imagenet_smoothadv_025 \
    --skip 100 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 500 \
    --center_layer 1;
