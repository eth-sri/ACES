CUDA_VISIBLE_DEVICES=0 \
    python predict_core.py \
    cifar10 \
    ./data/cifar10/core/resnet110 \
    --skip 20 \
    --core_classifier ../pretrained-models-new/gaussian/noise_0.00/checkpoint.pth.tar
