#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib.pyplot import figure, cm
import numpy as np
import logging
import argparse
import os
from npp import zscore
import h5py
import pickle


def load_brain_data(story, subj):
    if story=='bourn':
        brain_data = np.load('sub'+str(subj)+'-bourne-fsaverage6.npy')[:,:4024]
    if story=='wolf':
        brain_data = np.load('sub'+str(subj)+'-wolf-fsaverage6.npy')[:,:6989]
    if story=='all':
        temp = []
        brain_data = np.load('sub'+str(subj)+'-bourne-fsaverage6.npy')[:,:4024]
        temp.append(zscore(brain_data))
        brain_data = np.load('sub'+str(subj)+'-wolf-fsaverage6.npy')[:,:6989]
        temp.append(zscore(brain_data))
        brain_data = np.hstack(temp)
    return brain_data


# In[5]:


def load_test_brain_data(subj):
    brain_data = zscore(np.load('sub'+str(subj)+'-life-fsaverage6.npy'))[:,:2008]
    return brain_data


def load_stim_data(Rstim, Pstim):
    print(np.array(Rstim).shape, np.array(Pstim).shape)

    # Delay stimuli
    from util import make_delayed
    ndelays = 5
    delays = range(1, ndelays+1)

    print ("FIR model delays: ", delays)

    delRstim = []
    for eachlayer in np.arange(1):
        delRstim.append(make_delayed(np.array(Rstim), delays))
        
    delPstim = []
    for eachlayer in np.arange(1):
        delPstim.append(make_delayed(np.array(Pstim), delays))
    return delRstim, delPstim


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = int)
    parser.add_argument("modality", help="Choose modality", type = str)
    parser.add_argument("story", help="Choose story", type = str)
    parser.add_argument("outputdir", help="Output directory", type = str)
    parser.add_argument("layer", help="Choose layer", type = int)
    args = parser.parse_args()

    # Load responses
    # Load training data for subject 1, reading dataset 
    zRresp, zPresp = load_brain_data(args.story, args.subjectNum).T, load_test_brain_data(args.subjectNum).T
    # Print matrix shapes
    print ("zRresp shape (num time points, num voxels): ", zRresp.shape)
    print ("zPresp shape (num time points, num voxels): ", zPresp.shape)

    train_data = np.load('meta-video-audio-train.npy', allow_pickle=True)
    test_data = np.load('meta-video-audio-test.npy', allow_pickle=True)

    # train_data = np.load('tvlt-joint-videomae-train.npy', allow_pickle=True)
    # test_data = np.load('tvlt-joint-videomae-test.npy', allow_pickle=True)
   
    
    # Run regression
    from ridge_utils.ridge import bootstrap_ridge

    nboots = 1 # Number of cross-validation runs.
    chunklen = 40 # 
    nchunks = 20
    save_dir = args.outputdir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    subdir = str(args.subjectNum)
    if not os.path.exists(save_dir+'/'+subdir):
        os.mkdir(save_dir+'/'+subdir)
    for eachlayer in np.arange(args.layer, 12):
        Rstim, Pstim = train_data.item()[eachlayer], test_data.item()[eachlayer]
        delRstim, delPstim = load_stim_data(Rstim, Pstim)
        # Print the sizes of these matrices
        print ("delRstim shape: ", delRstim[0].shape)
        print ("delPstim shape: ", delPstim[0].shape)
        
        alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
        all_corrs = []
        if not os.path.exists(save_dir+'/'+subdir+'/'+args.modality+'/'+args.story+'/'+str(eachlayer)):
            os.makedirs(save_dir+'/'+subdir+'/'+args.modality+'/'+args.story+'/'+str(eachlayer))
        wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(np.nan_to_num(delRstim[0]), np.nan_to_num(zRresp), np.nan_to_num(delPstim[0]), np.nan_to_num(zPresp),
                                                             alphas, nboots, chunklen, nchunks,
                                                             singcutoff=1e-10, single_alpha=True)
        pred = np.dot(delPstim[0], wt)

        print ("pred has shape: ", pred.shape)
        voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
        for vi in range(zPresp.shape[1]):
            voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]
        print (voxcorrs)

        #np.save(os.path.join(main_dir+'/'+save_dir, "layer_"+str(eachlayer)),voxcorrs)
        np.save(os.path.join(save_dir+'/'+subdir+'/'+args.modality+'/'+args.story+'/'+str(eachlayer), "layer_"+str(eachlayer)),voxcorrs)


# In[ ]:




