CUDA_VISIBLE_DEVICES=7 \
    taskset -c 32-47 \
    python certify_ace.py \
    cifar10 \
    ./pretrained-models/cifar10/gaussian/noise_1.00/checkpoint.pth.tar \
    1.00 \
    ./data/cifar10/gaussian/noise_1.00/cifar10_gaussian_100 \
    --skip 20 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 1000; \
CUDA_VISIBLE_DEVICES=7 \
    taskset -c 32-47 \
    python certify_ace.py \
    cifar10 \
    ./pretrained-models/cifar10/gaussian/noise_0.50/checkpoint.pth.tar \
    0.50 \
    ./data/cifar10/gaussian/noise_0.50/cifar10_gaussian_050 \
    --skip 20 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 1000; \
CUDA_VISIBLE_DEVICES=7 \
    taskset -c 32-47 \
    python certify_ace.py \
    cifar10 \
    ./pretrained-models/cifar10/gaussian/noise_0.25/checkpoint.pth.tar \
    0.25 \
    ./data/cifar10/gaussian/noise_0.25/cifar10_gaussian_025 \
    --skip 20 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 1000;
