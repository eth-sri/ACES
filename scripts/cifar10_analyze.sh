python analyze_utils.py \
    data/cifar10/gaussian/noise_0.25/cifar10_gaussian_025 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/gaussian/noise_025/ace_smoothing_table_025_lanet_gaussian \
    analysis/tables/cifar10/gaussian/noise_025/ace_smoothing_table_selection_025_lanet_gaussian; \
python analyze_utils.py \
    data/cifar10/gaussian/noise_0.50/cifar10_gaussian_050 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/gaussian/noise_050/ace_smoothing_table_050_lanet_gaussian \
    analysis/tables/cifar10/gaussian/noise_050/ace_smoothing_table_selection_050_lanet_gaussian; \
python analyze_utils.py \
    data/cifar10/gaussian/noise_1.00/cifar10_gaussian_100 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/gaussian/noise_100/ace_smoothing_table_100_lanet_gaussian \
    analysis/tables/cifar10/gaussian/noise_100/ace_smoothing_table_selection_100_lanet_gaussian \
    --radii_mode 1; \
python analyze_utils.py \
    data/cifar10/smoothadv/noise_0.25/cifar10_smoothadv_025 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/smoothadv/noise_025/ace_smoothing_table_025_lanet_smoothadv \
    analysis/tables/cifar10/smoothadv/noise_025/ace_smoothing_table_selection_025_lanet_smoothadv; \
python analyze_utils.py \
    data/cifar10/smoothadv/noise_0.50/cifar10_smoothadv_050 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/smoothadv/noise_050/ace_smoothing_table_050_lanet_smoothadv \
    analysis/tables/cifar10/smoothadv/noise_050/ace_smoothing_table_selection_050_lanet_smoothadv; \
python analyze_utils.py \
    data/cifar10/smoothadv/noise_1.00/cifar10_smoothadv_100 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/smoothadv/noise_100/ace_smoothing_table_100_lanet_smoothadv \
    analysis/tables/cifar10/smoothadv/noise_100/ace_smoothing_table_selection_100_lanet_smoothadv \
    --radii_mode 1; \
python analyze_utils.py \
    data/cifar10/macer/noise_0.25/cifar10_macer_025 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/macer/noise_025/ace_smoothing_table_025_lanet_macer \
    analysis/tables/cifar10/macer/noise_025/ace_smoothing_table_selection_025_lanet_macer; \
python analyze_utils.py \
    data/cifar10/macer/noise_0.50/cifar10_macer_050 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/macer/noise_050/ace_smoothing_table_050_lanet_macer \
    analysis/tables/cifar10/macer/noise_050/ace_smoothing_table_selection_050_lanet_macer; \
python analyze_utils.py \
    data/cifar10/macer/noise_1.00/cifar10_macer_100 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/macer/noise_100/ace_smoothing_table_100_lanet_macer \
    analysis/tables/cifar10/macer/noise_100/ace_smoothing_table_selection_100_lanet_macer \
    --radii_mode 1; \
python analyze_utils.py \
    data/cifar10/consistency/noise_0.25/cifar10_consistency_025 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/consistency/noise_025/ace_smoothing_table_025_lanet_consistency \
    analysis/tables/cifar10/consistency/noise_025/ace_smoothing_table_selection_025_lanet_consistency; \
python analyze_utils.py \
    data/cifar10/consistency/noise_0.50/cifar10_consistency_050 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/consistency/noise_050/ace_smoothing_table_050_lanet_consistency \
    analysis/tables/cifar10/consistency/noise_050/ace_smoothing_table_selection_050_lanet_consistency; \
python analyze_utils.py \
    data/cifar10/consistency/noise_1.00/cifar10_consistency_100 \
    data/cifar10/core/lanet \
    analysis/tables/cifar10/consistency/noise_100/ace_smoothing_table_100_lanet_consistency \
    analysis/tables/cifar10/consistency/noise_100/ace_smoothing_table_selection_100_lanet_consistency \
    --radii_mode 1;
