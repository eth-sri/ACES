'''
- this is the file which computes natural accuracy of the core model on unperturbed samples
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/certify.py written by Jeremy Cohen
'''

import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes, get_dataset_efficientnet
from smooth_ace import SmoothACE
from time import time
import torch
import datetime
from architectures import get_architecture
import numpy as np
import torch
from torchvision import transforms


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--skip", type=int, default=20, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--core_classifier", type=str, help="path to saved core classifier") # path to accurate core-classifier
args = parser.parse_args()



if __name__ == "__main__":
    
    # load core model
    checkpoint = torch.load(args.core_classifier)
    core_classifier = get_architecture(checkpoint["arch"], args.dataset)
    core_classifier.load_state_dict(checkpoint['state_dict'])
    core_classifier.eval()
    core_classifier = core_classifier.cuda()
        
    # log output
    output_file = open(args.outfile+"_core_prediction", 'w')
    print("idx\tlabel\tpredict\tcorrect\ttime",
          file=output_file, flush=True)

    # obtain dataset
    dataset = get_dataset(args.dataset, args.split)
        
    corrects, all_num = 0, 0
    for i in range(len(dataset)):

        # only consider every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        all_num += 1

        before_time = time()
        x = x.cuda()
        
        # computing prediction
        to_add = []
        with torch.no_grad():
            clean_output = core_classifier(x.repeat((1, 1, 1, 1)).cuda())
        print(clean_output.argmax().item(), label)
        correct = 0
        predict = clean_output.argmax().item()
        if predict == label:
            corrects += 1
            correct = 1
        
        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        
        # "idx\tlabel\tpredict\tcorrect\ttime"
        print("{}\t{}\t{}\t{}\t{}".format(
            i, label, predict, correct, time_elapsed), file=output_file, flush=True)
        
        
    # some logging data
    print('corrects (absolute and relative): ', corrects, corrects / all_num)
    print('count: ', all_num)
