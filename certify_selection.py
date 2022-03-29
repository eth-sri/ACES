'''
- this is the file which does certification for the SmoothSelection class (smooth_selection.py)
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/certify.py written by Jeremy Cohen
'''

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from smooth_selection import SmoothSelection
from time import time
import torch
import datetime
from architectures import get_architecture, get_architecture_center_layer
from architectures_macer import resnet110
import numpy as np
import torch
from torchvision import transforms
from torch import nn as nn

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--use_binary_classifier", type=int, default=0)
parser.add_argument("--N1", type=int, default=10000, help="sampling size for prediction")
args = parser.parse_args()



if __name__ == "__main__":
    
    # load the base_classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture('cifar_resnet110_selection', 'cifar10')
    if args.use_binary_classifier == 1:
        base_classifier = get_architecture('cifar_resnet110_binary', 'cifar10')
    print(base_classifier)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier = base_classifier.to('cuda')

    # boundaries to consider for selection network
    # relevant boundaries depend a bit on dataset and on base classifier
    boundaries = np.arange(0.0, 1+0.01, 0.01)
    
    # log files for selection models (one for each boundary)
    output_files_selection = []
    output_dir = os.path.dirname(args.outfile)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for boundary in boundaries:
        f = open(args.outfile+"_selection_"+"{:.2f}".format(boundary), 'w')
        print("idx\tpredict\tradius\tunperturbed predict\ttime", file=f, flush=True)
        output_files_selection.append(f)
    
    # log file for outputs statistics
    outputs_file = open(args.outfile+"_outputs", 'w')
    print("idx\tmean\tstd\tmin\t25 percentile\t50 percentile\t75 percentile\tmax",
          file=outputs_file, flush=True)
    
    # smoothed classifier
    smoothed_classifier = SmoothSelection(base_classifier, get_num_classes(args.dataset), args.sigma, boundaries)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        # compute things for selection model
        before_time = time()
        (x, label) = dataset[i]
        x = x.cuda()
        certified_radii_s, outputs_statistics = smoothed_classifier.certify(x, args.N0, args.N, args.alpha / 2, args.batch)
        predictions_s = smoothed_classifier.predict(x, args.N1, args.alpha / 2, args.batch)
        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            
        # processing results for selection networks
        # "idx\tpredict\tradius\tunperturbed predict\ttime"
        for j, output_file in enumerate(output_files_selection):
            predict = certified_radii_s[j][0]
            radius = certified_radii_s[j][1]
            unperturbed_predict = predictions_s[j]
            print("{}\t{}\t{:.3}\t{}\t{}".format(
                i, predict, radius, unperturbed_predict, time_elapsed), file=output_file, flush=True)
        
        # processing entropy statistics
        # "idx\tmean\tstd\tmin\t25 percentile\t50 percentile\t75 percentile\tmax"
        print("{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}".format(
            i, outputs_statistics[0], outputs_statistics[1], outputs_statistics[2],
            outputs_statistics[3], outputs_statistics[4], outputs_statistics[5],
            outputs_statistics[6]), file=outputs_file, flush=True)
