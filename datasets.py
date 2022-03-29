'''
- this file is necessary to get the right datasets (CIFAR10 and ImageNet)
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/datasets.py written by Jeremy Cohen and on https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/datasets.py written by Hadi Salman
'''

from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "cifar10", "cifar10_selection"]


def get_dataset(dataset: str, split: str, selection_labels=None, use_binary=False) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "cifar10_selection":
        return _cifar10_selection(split, selection_labels, use_binary)

# efficientnet transforms the data slighlty differently
# https://github.com/lukemelas/EfficientNet-PyTorch
def get_dataset_efficientnet():
    dir = os.environ[IMAGENET_LOC_ENV]
    subdir = os.path.join(dir, "val")
    # transforms.Scale(256),
    #        transforms.CenterCrop(224), # centerCrop seems to be necessary
    transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    return datasets.ImageFolder(subdir, transform)

def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar10_selection":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "cifar10_selection":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)

# added because some SmoothAdv models use this
def get_input_center_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's Input Centering layer"""
    if dataset == "imagenet":
        return InputCenterLayer(_IMAGENET_MEAN)
    elif dataset == "cifar10":
        return InputCenterLayer(_CIFAR10_MEAN)
    elif dataset == "cifar10_selection":
        return InputCenterLayer(_CIFAR10_MEAN)

# added because some SmoothAdv models use this 
class InputCenterLayer(torch.nn.Module):
    """Centers the channels of a batch of images by subtracting the dataset mean.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(InputCenterLayer, self).__init__()
        self.means = torch.tensor(means).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return input - means


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())
    elif split == "train_without_augmentation":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.ToTensor())


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


# dataset for selection regression    
class CIFAR10SELECTION(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, selection_labels=None, use_binary=False):
        super().__init__(root, train, transform, target_transform, download) 
        if use_binary:
            self.selection_labels = torch.from_numpy(selection_labels).type(torch.LongTensor)
        else:
            self.selection_labels = torch.from_numpy(selection_labels).type(torch.FloatTensor)
        
        
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        #print('types: target', type(target))
        target = self.selection_labels[index]
        #print('types: target, img', type(img), type(target))
        return img, target
        
    def __len__(self):
        return len(self.selection_labels)
    
def _cifar10_selection(split, selection_labels, use_binary=False) -> Dataset:
    if split == "train":
        return CIFAR10SELECTION("./dataset_cache", train=True, download=True, selection_labels=selection_labels, use_binary=use_binary, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return CIFAR10SELECTION("./dataset_cache", train=False, download=True, selection_labels=selection_labels, use_binary=use_binary, transform=transforms.ToTensor())
    elif split == "train_without_augmentation":
        return CIFAR10SELECTION("./dataset_cache", train=True, download=True, selection_labels=selection_labels, use_binary=use_binary, transform=transforms.ToTensor())
