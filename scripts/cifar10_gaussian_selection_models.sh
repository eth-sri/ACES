CUDA_VISIBLE_DEVICES=2 \
    taskset -c 16-19 \
    python certify_selection.py \
    cifar10 \
    ./pretrained-models/cifar10/gaussian/noise_0.25_selection_model/selection_model_gaussian_025_binary_050/checkpoint.pth.tar 0.25 \
    ./data/cifar10/gaussian/noise_0.25_selection_model/selection_model_gaussian_025_binary_050/selection_model_gaussian_025_binary_050 \
    --skip 20 \
    --batch 1000 \
    --use_binary_classifier 1; \
CUDA_VISIBLE_DEVICES=2 \
    taskset -c 16-19 \
    python certify_selection.py \
    cifar10 \
    ./pretrained-models/cifar10/gaussian/noise_0.25_selection_model/selection_model_gaussian_025_binary_095/checkpoint.pth.tar 0.25 \
    ./data/cifar10/gaussian/noise_0.25_selection_model/selection_model_gaussian_025_binary_095/selection_model_gaussian_025_binary_095 \
    --skip 20 \
    --batch 1000 \
    --use_binary_classifier 1;
