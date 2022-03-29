IMAGENET_DIR=/mnt/sdb/imagenet CUDA_VISIBLE_DEVICES=7 \
    python predict_core.py \
    imagenet \
    ./data/imagenet/core/resnet50 \
    --skip 100 \
    --core_classifier ./pretrained-models/imagenet/core-models/resnet50/checkpoint.pth.tar
