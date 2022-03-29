'''
- this file contains various functions to transform log files of constituting models into data for analysis
- it is based on the publicly available code https://github.com/locuslab/smoothing/blob/master/code/analyze.py written by Jeremy Cohen
'''

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Process data into tables')
parser.add_argument("aces_file_path", type=str, help="path to the path of the selection and certification model's output")
parser.add_argument("core_file_path", type=str, help="path to the core model's output")
parser.add_argument("output_file", type=str, help="path to the certification results' output file")
parser.add_argument("output_file_selection", type=str, help="path to the selection tables' output file")
parser.add_argument("--radii_mode", default=0, type=int, help="encoding for radii to consider")
args = parser.parse_args()



# transforms log file into numpy arrays
def get_data(file_path, data_type):
    
    nums = {'certification': 7, 'core_prediction': 4, 'selection': 4, 'entropies': 8}
    num = nums[data_type]

    f = open(file_path, 'r')
    lines = f.readlines()
    data = np.array([line.split('\t')[:num] for line in lines[1:]]).astype(float)
    
    return data

# computes clean natural accuracy (accuracy on unperturbed sample)
def get_natural_accuracy(selection_data, certification_data, core_prediction_data):
    accuracy = 0
    for i in range(len(selection_data)):
        if selection_data[i][3] == 1 and certification_data[i][6]: # certification network is chosen and is correct
            accuracy += 1
        elif selection_data[i][3] == 0 and core_prediction_data[i][3]: # core network is chosen and is correct
            accuracy += 1
        elif selection_data[i][3] == -1 and certification_data[i][6] and core_prediction_data[i][3]: # not clear what chosen but both are correct anyway
            accuracy += 1
    return accuracy / (len(selection_data) / 100)

# computes ACR (average certified radius)
def get_acr(selection_data, certification_data):
    acr = 0
    for i in range(len(selection_data)):
        # if certification network gets chosen and is correct
        if selection_data[i][1] and certification_data[i][4]: 
            acr += min(selection_data[i][2], certification_data[i][3])
    acr /= len(selection_data)
    return acr

# computes certified accuracy at a given radius > 0
def get_robust_accuracy_positive(selection_data, certification_data, radius):
    certified_accuracy = 0
    for i in range(len(selection_data)):
        # if certification network gets chosen and is correct
        if selection_data[i][1] and certification_data[i][4]:
            if min(selection_data[i][2], certification_data[i][3]) >= radius:
                certified_accuracy += 1
    return certified_accuracy / (len(selection_data) / 100)

# computes certified accuracy at radius 0
def get_robust_accuracy_zero(selection_data, certification_data, core_prediction_data):
    certified_accuracy = 0
    for i in range(len(selection_data)):
        # certification network gets chosen and is correct
        if selection_data[i][1] and certification_data[i][4]:
            certified_accuracy += 1
        # core network is chosen and is correct
        elif selection_data[i][1] == 0 and core_prediction_data[i][3]:
            certified_accuracy += 1
        # not clear what selection network chooses but both are correct anyway
        elif core_prediction_data[i][3] and certification_data[i][4]:
            certified_accuracy += 1
    return certified_accuracy / (len(selection_data) / 100)

# computes certified accuracy at a given radius
def get_robust_accuracy(selection_data, certification_data, core_prediction_data, radius):
    if radius > 0:
        return get_robust_accuracy_positive(selection_data, certification_data, radius)
    if radius == 0:
        return get_robust_accuracy_zero(selection_data, certification_data, core_prediction_data)
    raise Exception('radius has to be non-negative')

# gets selection rate of certification model at given radius
def get_selection_at_radius(selection_data, radius):
    certification_selected = 0
    for i in range(len(selection_data)):
        if selection_data[i][1] and selection_data[i][2] >= radius:
            certification_selected += 1
    return certification_selected / (len(selection_data) / 100)

# gets selection rate of certification model for unperturbed samples
def get_selection_unperturbed(selection_data):
    certification_selected = 0
    for i in range(len(selection_data)):
        if selection_data[i][3]:
            certification_selected += 1
    return certification_selected / (len(selection_data) / 100)

# gets summarised data for selection model
def get_selection_data(file_path, file_path_core, radii = np.arange(0, 4, 0.25), boundaries = np.arange(0, 1.01, 0.01)):
    unperturbed_selections = []
    selections = []
    for i in boundaries:
        selection = get_data(file_path + '_selection_{:.2f}'.format(i), 'selection')
        unperturbed_selections.append(get_selection_unperturbed(selection))
        selections.append([get_selection_at_radius(selection, radius) for radius in radii])
    return selections, unperturbed_selections

# gets data about natural accuracies, robust accuracies (at a given radius) and acr
def get_basic_data(file_path, file_path_core, radii = np.arange(0, 4, 0.25), boundaries = np.arange(0, 1.01, 0.01)):
    natural_accuracies, acrs, robust_accuracies = [], [], []
    certification = get_data(file_path + '_certification', 'certification')
    core_prediction = get_data(file_path_core + '_core_prediction', 'core_prediction')
    for i in boundaries:
        selection = get_data(file_path + '_selection_{:.2f}'.format(i), 'selection')
        natural_accuracies.append(get_natural_accuracy(selection, certification, core_prediction))
        acrs.append(get_acr(selection, certification))
        robust_accuracies.append([get_robust_accuracy(selection, certification, core_prediction, radius) for radius in radii])
    return natural_accuracies, acrs, robust_accuracies

# creates a latex table from the data; each row contains values for a threshold; the first two columns contain natural accuracy respectively acr; the remaining ones robust accuracy at given radii
def get_latex_table(output_path, file_path, file_path_core, radii = np.arange(0, 4, 0.25), boundaries = np.arange(0, 1.01, 0.01)):
    natural_accuracies, acrs, robust_accuracies = get_basic_data(file_path, file_path_core, radii, boundaries)
    
    # create file and write header
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f = open(output_path, 'w')
    f.write("& Threshold & NAC & ACR")
    for radius in radii:
        f.write(" & {:.3}".format(radius))
    f.write("\\\\\n\midrule\n")
    
    # add all rows to file
    for i in range(len(natural_accuracies)):
        f.write("& {:.2f}".format(boundaries[i]))
        f.write(" & {:.1f}".format(natural_accuracies[i]))
        f.write(" & {:.3f}".format(acrs[i]))
        for j in range(len(robust_accuracies[0])):
            f.write(" & {:.1f}".format(robust_accuracies[i][j]))
        f.write("\\\\\n")
    f.close()
    
# creates latex table for the selection model's data
def get_latex_table_selection(output_path, file_path, file_path_core, radii = np.arange(0, 4, 0.25), boundaries = np.arange(0, 1.01, 0.01)):
    natural_accuracies, acrs, robust_accuracies = get_basic_data(file_path, file_path_core, radii, boundaries)
    selections, unperturbed_selections = get_selection_data(file_path, file_path_core, radii, boundaries)
    
    # create file and write header
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    f = open(output_path, 'w')
    f.write("& Threshold & NAC & ACR")
    for radius in radii:
        f.write(" & {:.3}".format(radius))
    f.write("\\\\\n\midrule\n")
    
    # add all rows to file
    for i in range(len(natural_accuracies)):
        f.write("& {:.2f}".format(boundaries[i]))
        f.write(" & {:.1f}".format(natural_accuracies[i]))
        f.write(" & {:.3f}".format(acrs[i]))
        for j in range(len(selections[0])):
            f.write(" & {:.1f}".format(selections[i][j]))
        f.write("\\\\\n")
    f.close()
    
# main function
if __name__ == "__main__":
    thresholds = np.arange(0.0, 1.01, 0.1)
    if args.radii_mode == 0:
        radii = np.arange(0, 2.25, 0.25)
    elif args.radii_mode == 1:
        radii = np.arange(0, 4.5, 0.5)
    get_latex_table(args.output_file, args.aces_file_path, args.core_file_path, radii, thresholds)
    get_latex_table_selection(args.output_file_selection, args.aces_file_path, args.core_file_path, radii, thresholds)
           