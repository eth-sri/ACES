python analyze_utils.py \
    data/imagenet/gaussian/noise_0.25/imagenet_gaussian_025 \
    data/imagenet/core/efficientnet-b7 \
    analysis/tables/imagenet/gaussian/noise_025/ace_smoothing_table_025_enb7_gaussian \
    analysis/tables/imagenet/gaussian/noise_025/ace_smoothing_table_selection_025_enb7_gaussian; \
python analyze_utils.py \
    data/imagenet/gaussian/noise_0.50/imagenet_gaussian_050 \
    data/imagenet/core/efficientnet-b7 \
    analysis/tables/imagenet/gaussian/noise_050/ace_smoothing_table_050_enb7_gaussian \
    analysis/tables/imagenet/gaussian/noise_050/ace_smoothing_table_selection_050_enb7_gaussian; \
python analyze_utils.py \
    data/imagenet/gaussian/noise_1.00/imagenet_gaussian_100 \
    data/imagenet/core/efficientnet-b7 \
    analysis/tables/imagenet/gaussian/noise_100/ace_smoothing_table_100_enb7_gaussian \
    analysis/tables/imagenet/gaussian/noise_100/ace_smoothing_table_selection_100_enb7_gaussian \
    --radii_mode 1; \
python analyze_utils.py \
    data/imagenet/smoothadv/noise_0.25/imagenet_smoothadv_025 \
    data/imagenet/core/efficientnet-b7 \
    analysis/tables/imagenet/smoothadv/noise_025/ace_smoothing_table_025_enb7_smoothadv \
    analysis/tables/imagenet/smoothadv/noise_025/ace_smoothing_table_selection_025_enb7_smoothadv; \
python analyze_utils.py \
    data/imagenet/smoothadv/noise_0.50/imagenet_smoothadv_050 \
    data/imagenet/core/efficientnet-b7 \
    analysis/tables/imagenet/smoothadv/noise_050/ace_smoothing_table_050_enb7_smoothadv \
    analysis/tables/imagenet/smoothadv/noise_050/ace_smoothing_table_selection_050_enb7_smoothadv; \
python analyze_utils.py \
    data/imagenet/smoothadv/noise_1.00/imagenet_smoothadv_100 \
    data/imagenet/core/efficientnet-b7 \
    analysis/tables/imagenet/smoothadv/noise_100/ace_smoothing_table_100_enb7_smoothadv \
    analysis/tables/imagenet/smoothadv/noise_100/ace_smoothing_table_selection_100_enb7_smoothadv \
    --radii_mode 1; \
python analyze_utils.py \
    data/imagenet/consistency/noise_0.25/imagenet_consistency_025 \
    data/imagenet/core/efficientnet-b7 \
    analysis/tables/imagenet/consistency/noise_025/ace_smoothing_table_025_enb7_consistency \
    analysis/tables/imagenet/consistency/noise_025/ace_smoothing_table_selection_025_enb7_consistency; \
python analyze_utils.py \
    data/imagenet/consistency/noise_0.50/imagenet_consistency_050 \
    data/imagenet/core/efficientnet-b7 \
    analysis/tables/imagenet/consistency/noise_050/ace_smoothing_table_050_enb7_consistency \
    analysis/tables/imagenet/consistency/noise_050/ace_smoothing_table_selection_050_enb7_consistency; \
python analyze_utils.py \
    data/imagenet/consistency/noise_1.00/imagenet_consistency_100 \
    data/imagenet/core/efficientnet-b7 \
    analysis/tables/imagenet/consistency/noise_100/ace_smoothing_table_100_enb7_consistency \
    analysis/tables/imagenet/consistency/noise_100/ace_smoothing_table_selection_100_enb7_consistency \
    --radii_mode 1;
