CUDA_VISIBLE_DEVICES=4 \
    taskset -c 32-47 \
    python certify_ace.py \
    cifar10 \
    ./pretrained-models/cifar10/smoothadv/checkpoint-cifar10-PGD_10steps_multiNoiseSamples-2-multitrain-eps_512-cifar10-resnet110-noise_100.pth.tar \
    1.00 \
    ./data/cifar10/smoothadv/noise_1.00/cifar10_smoothadv_100 \
    --skip 20 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 1000 \
    --center_layer 1; \
CUDA_VISIBLE_DEVICES=4 \
    taskset -c 32-47 \
    python certify_ace.py \
    cifar10 \
    ./pretrained-models/cifar10/smoothadv/checkpoint-cifar10-PGD_2steps_multiNoiseSamples-8-multitrain-eps_512-cifar10-resnet110-noise_050.pth.tar \
    0.50 \
    ./data/cifar10/smoothadv/noise_0.50/cifar10_smoothadv_050 \
    --skip 20 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 1000 \
    --center_layer 1; \
CUDA_VISIBLE_DEVICES=4 \
    taskset -c 32-47 \
    python certify_ace.py \
    cifar10 \
    ./pretrained-models/cifar10/smoothadv/checkpoint-cifar10-PGD_10steps_multiNoiseSamples-8-multitrain-eps_255-cifar10-resnet110-noise_025.pth.tar \
    0.25 \
    ./data/cifar10/smoothadv/noise_0.25/cifar10_smoothadv_025 \
    --skip 20 \
    --N0 100 \
    --N1 10000 \
    --N 100000 \
    --alpha 0.001 \
    --batch 1000 \
    --center_layer 1; \