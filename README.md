# Robust and Accurate - Compositional Architectures for Randomized Smoothing <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

This repository is the official implementation of `ACES` for the paper [Robust and Accurate - Compositional Architectures for Randomized Smoothing (ICLR 2022 Workshop on Socially Responsible Machine Learning)](https://www.sri.inf.ethz.ch/publications/horvath2022robust).

## Overview

The key idea of `ACES` is to utilize a smoothed selection-model to certifiable decide on a per-sample basis whether to use a robust smoothed certification-model with guarantees or a highly accurate core-model without guarantees. By doing so, it achieves significantly more attractive trade-offs between robustness and accuracy than any current method on `ImageNet` and `CIFAR10`.

This codebase implements `ACES` with entropy-based selection (`smooth_ace.py` and `certify_ace.py`), as well as an arbitrary binary selection mechanism (`smooth_selection.py` and `certify_selection.py`). In addition to the code, the `scripts` folder contains scripts that were used to create data for the paper. The resulting data is stored in the `data` folder. Finally, this data was processed by `analyze_utils.py` and the resulting `LaTex` tables are stored in the `analysis` folder.

## Getting Started

First, please install the required environment as follows:

```
conda create --name aces_env python=3.8
conda activate aces_env
pip install -r requirements.txt
```

To install pytorch 1.8.0 and torchvision 0.9.0 you can use the following command (depending on your installed CUDA version (can be checked e.g., by running nvidia-smi)):

```
# CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

If you are not using the most recent GPU drivers, please see [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for a compatible cudatoolkit version.

We provide trained models on `CIFAR10` and `ImageNet` we use [here (2.3 GB)](https://files.sri.inf.ethz.ch/aces/aces-models.tar.gz). Please move the folder `aces-models` into the root directory of this codebase.

## `ACES` Example

We provide an example of the whole `ACES` pipeline with an entropy-based selection model on `CIFAR10` to show how to use our codebase. For this, first, please create a folder `output_dir` where the outputs of the example will be saved, e.g. via `mkdir output_dir`.

### 1. Certifying the Robust Certification-Model and the Corresponding Entropy-Based Selection-Model

We can certify the certification and corresponding entropy-based selection model with the `certify_ace.py` code as follows:

```
python certify_ace.py \
    cifar10 \
    ./aces-models/cifar10/consistency/noise_0.25/checkpoint.pth.tar \
    0.25 \
    ./output_dir/output_file_smoothed
```

The first argument is the dataset (`cifar10` or `imagenet`), the second is the path to the (pre-trained) classifier, followed by the certification noise and the path where the results should be saved.

### 2. Computing Results for the Accurate Core-Model

We can compute results for the core model via `predict_core.py` as follows:

```
python predict_core.py \
    cifar10 \
    ./output_dir/output_file_core \
    --core_classifier ./aces-models/cifar10/gaussian/noise_0.00/checkpoint.pth.tar
```

The first argument is the dataset (either `cifar10` or `imagenet`), the second one is the path where the output should be saved. The path to the classifier can be added with `--core_classifier`.

We remark, that for some core models, we used their some publicly available code bases and just adapted them such that they created the outputs in a similar format as `predict_core.py`:
- [`LaNet`](https://github.com/facebookresearch/LaMCTS/tree/master/LaNAS/LaNet) for `CIFAR10`:
- [`EfficientNet`](https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/examples/imagenet) for `ImageNet`: 

### 3. Combining Results to `ACES` Outputs

Finally, to compute the `ACES` outputs, we combine the results of the constituting models via `analyze_utils.py`:

```
python analyze_utils.py \
    output_dir/output_file_smoothed \
    output_dir/output_file_core \
    output_dir/aces_table \
    output_dir/aces_table_selection;
```

The `output_dir/output_file_smoothed` is the path to the outputs of the selection and certification model from from step 1, the `output_dir/output_file_core` is the path to the output from step 2. The final `LaTex` table with the certified radii is then stored in `output_dir/aces_table` while the corresponding certified selection radii is stored in `output_dir/aces_table_selection`.

## Contributors

- Miklós Z. Horváth
- [Mark Niklas Müller](https://www.sri.inf.ethz.ch/people/mark)
- [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc)
- [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)

## Citation

If you find this work useful for your research, please cite it as:

```
@misc{
    horvath2022robust,
    title={Robust and Accurate - Compositional Architectures for Randomized Smoothing},
    author={Mikl{\'o}s Z. Horv{\'a}th and Mark Niklas M{\"u}ller and Marc Fischer and Martin Vechev},
    booktitle={ICLR 2022 Workshop on Socially Responsible Machine Learning},
    year={2022},
    url={https://www.sri.inf.ethz.ch/publications/horvath2022robust}
}
```
