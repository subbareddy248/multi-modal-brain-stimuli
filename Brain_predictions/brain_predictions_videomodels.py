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


def load_data(modality, story):
    train_data = []
    if story=='bourn':
        temp_data = []
        for i in np.arange(10):
            if i+1<10:
                sess_data = np.load('features_representations/bourn_'+modality+'_0'+str(i+1)+'.npy', allow_pickle=True)
                
            else:
                sess_data = np.load('features_representations/bourn_'+modality+'_'+str(i+1)+'.npy', allow_pickle=True)
            #print(sess_data[:,0,:].shape)
            temp_data.append([sess_data[:,0,:]])
        train_data.append(np.hstack(temp_data))
    elif story=='wolf':
        temp_data = []
        for i in np.arange(17):
            if i+1<10:
                sess_data = np.load('features_representations/wolf_'+modality+'_0'+str(i+1)+'.npy', allow_pickle=True)
                
            else:
                sess_data = np.load('features_representations/wolf_'+modality+'_'+str(i+1)+'.npy', allow_pickle=True)
            #print(sess_data[:,0,:].shape)
            temp_data.append([sess_data[:,0,:]])
        train_data.append(np.hstack(temp_data))
    elif story=='all':
        temp_data = []
        for i in np.arange(10):
            if i+1<10:
                sess_data = np.load('features_representations/bourn_'+modality+'_0'+str(i+1)+'.npy', allow_pickle=True)
                
            else:
                sess_data = np.load('features_representations/bourn_'+modality+'_'+str(i+1)+'.npy', allow_pickle=True)
            #print(sess_data[:,0,:].shape)
            temp_data.append([sess_data[:,0,:]])
        train_data.append(np.hstack(temp_data))
        temp_data = []
        for i in np.arange(17):
            if i+1<10:
                sess_data = np.load('features_representations/wolf_'+modality+'_0'+str(i+1)+'.npy', allow_pickle=True)
                
            else:
                sess_data = np.load('features_representations/wolf_'+modality+'_'+str(i+1)+'.npy', allow_pickle=True)
            #print(sess_data[:,0,:].shape)
            temp_data.append([sess_data[:,0,:]])
        train_data.append(np.hstack(temp_data))
    train_data = np.hstack(train_data)
    return train_data[0]


def load_test_data(modality):
    test_data = []
    for i in np.arange(5):
        sess_data = np.load('features_representations/life_'+modality+'_0'+str(i+1)+'.npy', allow_pickle=True)
        test_data.append([sess_data[:,0,:]])
    test_data = np.hstack(test_data)[0][:2008,:]
    return test_data


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


def load_test_brain_data(subj):
    brain_data = zscore(np.load('sub'+str(subj)+'-life-fsaverage6.npy'))[:,:2008]
    return brain_data


def load_tvlt(story, eachlayer):
    train_data = []
    if story=='bourn':
        temp_data = []
        for i in np.arange(10):
            if i+1<10:
                with open('features_representations/joint_embeddings_bourne0'+str(i+1)+'.pkl', 'rb') as f:
                    sess_data = pickle.load(f)
                temp_data.append([np.array(sess_data['bourne0'+str(i+1)])[:,eachlayer,:]])
            else:
                with open('features_representations/joint_embeddings_bourne'+str(i+1)+'.pkl', 'rb') as f:
                    sess_data = pickle.load(f)
                temp_data.append([np.array(sess_data['bourne'+str(i+1)])[:379,eachlayer,:]])
        train_data.append(np.hstack(temp_data))
    elif story=='wolf':
        temp_data = []
        for i in np.arange(17):
            if i+1<10:
                with open('features_representations/joint_embeddings_wolf0'+str(i+1)+'.pkl', 'rb') as f:
                    sess_data = pickle.load(f)
                temp_data.append([np.array(sess_data['wolf0'+str(i+1)])[:,eachlayer,:]])
            else:
                with open('features_representations/joint_embeddings_wolf'+str(i+1)+'.pkl', 'rb') as f:
                    sess_data = pickle.load(f)
                temp_data.append([np.array(sess_data['wolf'+str(i+1)])[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data)[:,:6989])
    elif story=='all':
        temp_data = []
        for i in np.arange(10):
            if i+1<10:
                with open('features_representations/joint_embeddings_bourne0'+str(i+1)+'.pkl', 'rb') as f:
                    sess_data = pickle.load(f)
                temp_data.append([np.array(sess_data['bourne0'+str(i+1)])[:,eachlayer,:]])
            else:
                with open('features_representations/joint_embeddings_bourne'+str(i+1)+'.pkl', 'rb') as f:
                    sess_data = pickle.load(f)
                temp_data.append([np.array(sess_data['bourne'+str(i+1)])[:379,eachlayer,:]])
        train_data.append(np.hstack(temp_data))
        temp_data = []
        for i in np.arange(17):
            if i+1<10:
                with open('features_representations/joint_embeddings_wolf0'+str(i+1)+'.pkl', 'rb') as f:
                    sess_data = pickle.load(f)
                temp_data.append([np.array(sess_data['wolf0'+str(i+1)])[:,eachlayer,:]])
            else:
                with open('features_representations/joint_embeddings_wolf'+str(i+1)+'.pkl', 'rb') as f:
                    sess_data = pickle.load(f)
                temp_data.append([np.array(sess_data['wolf'+str(i+1)])[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data)[:,:6989])
    train_data = np.hstack(train_data)
    return train_data[0]

def load_test_tvlt(eachlayer):
    test_data = []
    temp_data = []
    for i in np.arange(5):
        with open('features_representations/joint_embeddings_life0'+str(i+1)+'.pkl', 'rb') as f:
            sess_data = pickle.load(f)
        temp_data.append([np.array(sess_data['life0'+str(i+1)])[:,eachlayer,:]])
    test_data.append(np.hstack(temp_data)[0,:2008,:])
    return test_data[0]


def load_videomae(modality, story, eachlayer):
    train_data = []
    if story=='bourn':
        temp_data = []
        sess_data = np.load('features_representations/bourne_videomae.npy', allow_pickle=True)
        for i in np.arange(10):
            temp_data.append([np.array(sess_data[i])[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data))
    elif story=='wolf':
        temp_data = []
        sess_data = np.load('features_representations/wolf_videomae.npy', allow_pickle=True)
        for i in np.arange(8):
            temp_data.append([np.array(sess_data[i])[:,eachlayer,:]])
        sess_data = np.load('features_representations/wolf_videomae_8_14.npy', allow_pickle=True)
        for i in np.arange(6):
            temp_data.append([np.array(sess_data[i])[:,eachlayer,:]])
        sess_data = np.load('features_representations/wolf_videomae_14_17.npy', allow_pickle=True)
        for i in np.arange(3):
            temp_data.append([np.array(sess_data[i])[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data)[:,:6989])
    elif story=='all':
        temp_data = []
        sess_data = np.load('features_representations/bourne_videomae.npy', allow_pickle=True)
        for i in np.arange(10):
            temp_data.append([np.array(sess_data[i])[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data))
        temp_data = []
        sess_data = np.load('features_representations/wolf_videomae.npy', allow_pickle=True)
        for i in np.arange(8):
            temp_data.append([np.array(sess_data[i])[:,eachlayer,:]])
        sess_data = np.load('features_representations/wolf_videomae_8_14.npy', allow_pickle=True)
        for i in np.arange(6):
            temp_data.append([np.array(sess_data[i])[:,eachlayer,:]])
        sess_data = np.load('features_representations/wolf_videomae_14_17.npy', allow_pickle=True)
        for i in np.arange(3):
            temp_data.append([np.array(sess_data[i])[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data)[:,:6989])
    train_data = np.hstack(train_data)
    return train_data[0]

def load_test_videomae(modality, story, eachlayer):
    test_data = []
    temp_data = []
    sess_data = np.load('features_representations/life_videomae.npy', allow_pickle=True)
    for i in np.arange(5):
        temp_data.append([np.array(sess_data[i])[:,eachlayer,:]])
    test_data.append(np.hstack(temp_data)[0,:2008,:])
    return test_data[0]


def load_vith(modality, story, eachlayer):
    train_data = []
    if story=='bourn':
        temp_data = []
        for i in np.arange(10):
            if i+1<10:
                sess_data = np.load('features_representations/bourne0'+str(i+1)+'_'+modality+'.npy', allow_pickle=True)
            else:
                sess_data = np.load('features_representations/bourne'+str(i+1)+'_'+modality+'.npy', allow_pickle=True)[:379]
            temp_data.append([sess_data[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data))
    elif story=='wolf':
        temp_data = []
        for i in np.arange(17):
            if i+1<10:
                sess_data = np.load('features_representations/wolf0'+str(i+1)+'_'+modality+'.npy', allow_pickle=True)
            else:
                sess_data = np.load('features_representations/wolf'+str(i+1)+'_'+modality+'.npy', allow_pickle=True)
            temp_data.append([sess_data[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data)[:,:6989])
    elif story=='all':
        temp_data = []
        for i in np.arange(10):
            if i+1<10:
                sess_data = np.load('features_representations/bourne0'+str(i+1)+'_'+modality+'.npy', allow_pickle=True)
            else:
                sess_data = np.load('features_representations/bourne'+str(i+1)+'_'+modality+'.npy', allow_pickle=True)[:379]
            temp_data.append([sess_data[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data))
        temp_data = []
        for i in np.arange(17):
            if i+1<10:
                sess_data = np.load('features_representations/wolf0'+str(i+1)+'_'+modality+'.npy', allow_pickle=True)
            else:
                sess_data = np.load('features_representations/wolf'+str(i+1)+'_'+modality+'.npy', allow_pickle=True)
            temp_data.append([sess_data[:,eachlayer,:]])
        train_data.append(np.hstack(temp_data)[:,:6989])
    train_data = np.hstack(train_data)
    return train_data[0]


def load_test_vith(modality, eachlayer):
    train_data = []
    temp_data = []
    for i in np.arange(5):
        if i+1<10:
            sess_data = np.load('features_representations/life0'+str(i+1)+'_'+modality+'.npy', allow_pickle=True)
        temp_data.append([sess_data[:,eachlayer,:]])
    train_data.append(np.hstack(temp_data)[0,:2008,:])
    return train_data[0]


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
        Rstim, Pstim = load_tvlt(args.story, eachlayer), load_test_tvlt(eachlayer)
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