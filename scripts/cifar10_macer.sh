CUDA_VISIBLE_DEVICES=2 \
    taskset -c 32-47 \
    python certify_ace.py \
    cifar10 \
    ./pretrained-models/cifar10/macer/noise_1.00/1.00.pth \
    1.00 \
    ./data/cifar10/macer/noise_1.00/cifar10_macer_100 \
    --skip 20 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --use_macer 1 \
    --batch 1000; \
CUDA_VISIBLE_DEVICES=2 \
    taskset -c 32-47 \
    python certify_ace.py \
    cifar10 \
    ./pretrained-models/cifar10/macer/noise_0.50/0.50.pth \
    0.50 \
    ./data/cifar10/macer/noise_0.50/cifar10_macer_050 \
    --skip 20 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --use_macer 1 \
    --batch 1000; \
CUDA_VISIBLE_DEVICES=2 \
    taskset -c 32-47 \
    python certify_ace.py \
    cifar10 \
    ./pretrained-models/cifar10/macer/noise_0.25/0.25.pth \
    0.25 \
    ./data/cifar10/macer/noise_0.25/cifar10_macer_025 \
    --skip 20 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --use_macer 1 \
    --batch 1000;
