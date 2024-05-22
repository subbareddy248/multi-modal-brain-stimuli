#!/usr/bin/env python
# coding: utf-8

from matplotlib.pyplot import figure, cm
import numpy as np
import logging
import argparse
import os
from npp import zscore
import h5py
from sklearn.kernel_ridge import KernelRidge

def load_brain_data(story, subj):
    if story=='bourn':
        brain_data = zscore(np.load('sub'+str(subj)+'-bourne-fsaverage6.npy')[:,:4024])
    if story=='wolf':
        brain_data = zscore(np.load('sub'+str(subj)+'-wolf-fsaverage6.npy')[:,:6989])
    if story=='all':
        temp = []
        brain_data = np.load('sub'+str(subj)+'-bourne-fsaverage6.npy')[:,:4024]
        temp.append(zscore(brain_data))
        brain_data = np.load('sub'+str(subj)+'-wolf-fsaverage6.npy')[:,:6989]
        temp.append(zscore(brain_data))
        brain_data = np.hstack(temp)
    return brain_data


def load_test_brain_data(subj):
    brain_data = zscore(np.load('sub'+str(subj)+'-life-fsaverage6.npy'))[:,:2008]
    return brain_data

def pearcorr(actual, predicted):
    corr = []
    for i in range(0, len(actual)):
        corr.append(np.corrcoef(actual[i],predicted[i])[0][1])
    return corr

def kernel_ridge(xtrain, xtest, ytrain, ytest):
    krr = KernelRidge()
    krr.fit(np.nan_to_num(xtrain), np.nan_to_num(ytrain))
    ypred = krr.predict(np.nan_to_num(xtest))
    corr = pearcorr(np.nan_to_num(ytest.T),np.nan_to_num(ypred.T))
    return corr

subs = ['1','5','3']

save_dir = 'movie10_predictions_results_wolf/noise_ceiling/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = int)
    parser.add_argument("story", help="Choose story", type = str)
    args = parser.parse_args()
    target_subject = str(args.subjectNum)
    source_subjects = [i for i in subs if i != target_subject]
    for source_subject in source_subjects:
        # Load responses
        # Load training data for subject 1, reading dataset 
        sourcedata_train, sourcedata_test = load_brain_data(args.story, source_subject).T, load_test_brain_data(source_subject).T
        targetdata_train, targetdata_test = load_brain_data(args.story, target_subject).T, load_test_brain_data(target_subject).T
        corrs_t = kernel_ridge(sourcedata_train, sourcedata_test, targetdata_train, targetdata_test)
        np.save(os.path.join(save_dir, "predict_{}_with_{}_{}pcs.npy".format(target_subject, source_subject, sourcedata_train.shape[1])),corrs_t)