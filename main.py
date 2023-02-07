# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
# Visualization
import seaborn as sns
import visdom
from PIL import Image

import os
from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums,\
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device, get_train
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from models import get_model, train, test, save_model

import argparse

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='Houston')
parser.add_argument('--model', type=str, default='MSTViT')
parser.add_argument('--folder', type=str, help="Folder where to store the ", default="./Datasets/")
parser.add_argument('--cuda', type=int, default=0, help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=0.05,
                    help="Percentage of samples to use for training (default: 10%)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling or disjoint, default: random)",
                    default='random')
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, default=200, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int, default=15,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',  default=True,
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')

group_da.add_argument('--flip_augmentation', action='store_true', default = True,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default = True,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default = True,
                    help="Random mixes between spectra")
parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")

args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)
SAMPLE_PERCENTAGE = args.training_sample
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
TEST_STRIDE = args.test_stride

if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + ' ' + MODEL)
if not viz.check_connection:
    print("Wisdom is not connected. Did you run 'python -m visdom.server' ?")
repeat_term = 5
mean_oa = 0
mean_aa = 0
mean_kappa = 0
for dataset in ['IndianPines', "PaviaU", "Houston", "KSC"]:
    DATASET = dataset
    for j in range(repeat_term):
        hyperparams = vars(args)
        hyperparams['dataset'] = DATASET
        # Load the dataset
        img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(dataset_name=DATASET,
                                                                                target_folder=FOLDER,
                                                                                patch_size=PATCH_SIZE)
        # Number of classes
        # N_CLASSES = len(LABEL_VALUES) -  len(IGNORED_LABELS)
        N_CLASSES = len(LABEL_VALUES)
        # Number of bands (last dimension of the image tensor)
        N_BANDS = img.shape[-1]

        # Parameters for the SVM grid search
        SVM_GRID_PARAMS = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3], 'C': [1, 10, 100, 1000]},
                           {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
                           {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]

        if palette is None:
            palette = {0: (0, 0, 0)}
            for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
                palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
        invert_palette = {v: k for k, v in palette.items()}


        def convert_to_color(x):
            return convert_to_color_(x, palette=palette)


        def convert_from_color(x):
            return convert_from_color_(x, palette=invert_palette)


        # Instantiate the experiment based on predefined networks
        hyperparams.update(
            {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

        # Show the image and the ground truth
        color_gt = convert_to_color(gt)
        if DATAVIZ:
            # Data exploration : compute and show the mean spectrums
            mean_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz,
                                               ignored_labels=IGNORED_LABELS)
            plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')
        results = []
        # run the experiment several times
        for run in range(N_RUNS):
            train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, ignore=IGNORED_LABELS, mode=SAMPLING_MODE, )
            print("{} training samples & {} testing samples selected (over {})".format(np.count_nonzero(train_gt),
                                                                                       np.count_nonzero(test_gt),
                                                                                       np.count_nonzero(gt)))
            print("Running an experiment with the {} model".format(MODEL),
                  "run {}/{}".format(run + 1, N_RUNS))

            display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")  # 展示数据集和测试集
            display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")
            if CLASS_BALANCING:
                weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
                weights = torch.from_numpy(weights)
                weights = weights.cuda()
                hyperparams['weights'] = weights.float()
            # Neural network
            model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
            train_dataset = HyperX(img, train_gt, **hyperparams)
            train_loader = data.DataLoader(train_dataset,
                                           batch_size=hyperparams['batch_size'],
                                           shuffle=True)
            if CHECKPOINT is not None:
                model.load_state_dict(torch.load(CHECKPOINT))
            try:
                train(model, optimizer, loss, train_loader, hyperparams['epoch'],
                      scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                      supervision=hyperparams['supervision'], display=viz)
            except KeyboardInterrupt:
                pass

            probabilities = test(model, img, hyperparams)
            prediction = np.argmax(probabilities, axis=-1)
            prediction = prediction[PATCH_SIZE // 2:-(PATCH_SIZE // 2), PATCH_SIZE // 2: -(PATCH_SIZE // 2)]
        run_results = metrics(prediction, test_gt, ignored_labels=IGNORED_LABELS, n_classes=N_CLASSES)
        mask = np.zeros(gt.shape, dtype='bool')
        for l in IGNORED_LABELS:
            mask[gt == l] = True
        prediction[mask] = 0
        color_prediction = convert_to_color(prediction)
        display_predictions(color_prediction, viz, gt=convert_to_color(test_gt),
                            caption="Prediction vs. test ground truth")
        results.append(run_results)
        show_results(run_results, viz, label_values=LABEL_VALUES)
    if N_RUNS > 1:
        show_results(results, viz, label_values=LABEL_VALUES, agregated=True)
