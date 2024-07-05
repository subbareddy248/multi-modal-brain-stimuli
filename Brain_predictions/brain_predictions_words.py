#!/usr/bin/env python
# coding: utf-8

from matplotlib.pyplot import figure, cm
import numpy as np
import logging
import argparse
import os
from npp import zscore
import h5py
from ridge_utils.interpdata import lanczosinterp2D
from natsort import natsorted
import pandas as pd

class TRFile(object):
    def __init__(self, trfilename, expectedtr=1.49):
        """Loads data from [trfilename], should be output from stimulus presentation code.
        """
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr
        
        if trfilename is not None:
            self.load_from_file(trfilename)
        

    def load_from_file(self, trfilename):
        """Loads TR data from report with given [trfilename].
        """
        ## Read the report file and populate the datastructure
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label=="sound-start":
                self.soundstarttime = time

            elif label=="sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))
        
        ## Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes>(itrtimes.mean()*1.49))[0]
        newtrs = []
        for btr in badtrtimes:
            ## Insert new TR where it was missing..
            newtrtime = self.trtimes[btr]+self.expectedtr
            newtrs.append((newtrtime,btr))

        for ntr,btr in newtrs:
            self.trtimes.insert(btr+1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR.
        """
        self.trtimes = list(np.arange(ntrs)*self.expectedtr)
    
    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound.
        """
        return np.array(self.trtimes)-self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run.
        """
        return np.diff(self.trtimes).mean()

def load_generic_trfiles(stories, root="./"):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the session
    in which the data was collected.. this should be fine) for the given stories.
    """
    trdict = dict()

    for story in stories:
        try:
            trf = TRFile(os.path.join(root, "%s.report"%story))
            trdict[story] = [trf]
        except Exception as e:
            print (e)
    
    return trdict

trfiles = load_generic_trfiles(['bourn'])
wolf_trfiles = load_generic_trfiles(['wolf'])
wolf17_trfiles = load_generic_trfiles(['wolf17'])
life_trfiles = load_generic_trfiles(['life'])


wolf_dataset = np.load("../audio-textfMRi/wolf_bert_bert-base_20.npy",allow_pickle=True)
bourne_dataset = np.load("../audio-textfMRi/bourne_bert_bert-base_20.npy",allow_pickle=True)
life_dataset = np.load("../audio-textfMRi/life_bert_bert-base_20.npy",allow_pickle=True)


def load_word_data(story, eachlayer):
    train_data = []
    if story=='bourn':
        temp_data = []
        start_index = 0
        for i in np.arange(10):
            if i+1<10:
                read_file = pd.read_csv("./bourne_text/bourne0"+str(i+1)+'_words.csv')
                read_file[['start_time', 'end_time']] = read_file[['start_time', 'end_time']].apply(pd.to_numeric)
                read_file['word_times'] = (read_file['start_time']+read_file['end_time'])/2
                features = np.array(bourne_dataset.item()[eachlayer])[start_index:start_index+read_file.shape[0],:]
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, read_file['word_times'], trfiles['bourn'][0].trtimes)
            else:
                read_file = pd.read_csv("./bourne_text/bourne"+str(i+1)+'_words.csv')
                read_file[['start_time', 'end_time']] = read_file[['start_time', 'end_time']].apply(pd.to_numeric)
                read_file['word_times'] = (read_file['start_time']+read_file['end_time'])/2
                features = np.array(bourne_dataset.item()[eachlayer])[start_index:start_index+read_file.shape[0],:]
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, read_file['word_times'], trfiles['bourn'][0].trtimes)[:379,:]
            temp_data.append([downsampled_features])
            start_index = read_file.shape[0]
        train_data.append(np.hstack(temp_data))
    elif story=='wolf':
        temp_data = []
        start_index = 0
        for i in np.arange(17):
            if i+1<10:
                read_file = pd.read_csv("./wolf_captions/wolf0"+str(i+1)+'_words.csv')
                read_file[['start_time', 'end_time']] = read_file[['start_time', 'end_time']].apply(pd.to_numeric)
                read_file['word_times'] = (read_file['start_time']+read_file['end_time'])/2
                features = np.array(wolf_dataset.item()[eachlayer])[start_index:start_index+read_file.shape[0],:]
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, read_file['word_times'], wolf_trfiles['wolf'][0].trtimes)
            else:
                read_file = pd.read_csv("./wolf_captions/wolf"+str(i+1)+'_words.csv')
                read_file[['start_time', 'end_time']] = read_file[['start_time', 'end_time']].apply(pd.to_numeric)
                read_file['word_times'] = (read_file['start_time']+read_file['end_time'])/2
                features = np.array(wolf_dataset.item()[eachlayer])[start_index:start_index+read_file.shape[0],:]
                # you will need `wordseqs` from the notebook
                if i+1==17:
                    downsampled_features = lanczosinterp2D(features, read_file['word_times'], wolf17_trfiles['wolf17'][0].trtimes)[:493,:]
                else:
                    downsampled_features = lanczosinterp2D(features, read_file['word_times'], wolf_trfiles['wolf'][0].trtimes)
            start_index = read_file.shape[0]
            print(downsampled_features.shape)
            temp_data.append([downsampled_features])
        train_data.append(np.hstack(temp_data))
    elif story=='all':
        temp_data = []
        start_index = 0
        for i in np.arange(10):
            if i+1<10:
                read_file = pd.read_csv("./bourne_text/bourne0"+str(i+1)+'_words.csv')
                read_file[['start_time', 'end_time']] = read_file[['start_time', 'end_time']].apply(pd.to_numeric)
                read_file['word_times'] = (read_file['start_time']+read_file['end_time'])/2
                features = np.array(bourne_dataset.item()[eachlayer])[start_index:start_index+read_file.shape[0],:]
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, read_file['word_times'], trfiles['bourn'][0].trtimes)
            else:
                read_file = pd.read_csv("./bourne_text/bourne"+str(i+1)+'_words.csv')
                read_file[['start_time', 'end_time']] = read_file[['start_time', 'end_time']].apply(pd.to_numeric)
                read_file['word_times'] = (read_file['start_time']+read_file['end_time'])/2
                features = np.array(bourne_dataset.item()[eachlayer])[start_index:start_index+read_file.shape[0],:]
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, read_file['word_times'], trfiles['bourn'][0].trtimes)[:379,:]
            temp_data.append([downsampled_features])
            start_index = read_file.shape[0]
        train_data.append(np.hstack(temp_data))
        temp_data = []
        start_index = 0
        for i in np.arange(17):
            if i+1<10:
                read_file = pd.read_csv("./wolf_captions/wolf0"+str(i+1)+'_words.csv')
                read_file[['start_time', 'end_time']] = read_file[['start_time', 'end_time']].apply(pd.to_numeric)
                read_file['word_times'] = (read_file['start_time']+read_file['end_time'])/2
                features = np.array(wolf_dataset.item()[eachlayer])[start_index:start_index+read_file.shape[0],:]
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, read_file['word_times'], wolf_trfiles['wolf'][0].trtimes)
            else:
                read_file = pd.read_csv("./wolf_captions/wolf"+str(i+1)+'_words.csv')
                read_file[['start_time', 'end_time']] = read_file[['start_time', 'end_time']].apply(pd.to_numeric)
                read_file['word_times'] = (read_file['start_time']+read_file['end_time'])/2
                features = np.array(wolf_dataset.item()[eachlayer])[start_index:start_index+read_file.shape[0],:]
                # you will need `wordseqs` from the notebook
                if i+1==17:
                    downsampled_features = lanczosinterp2D(features, read_file['word_times'], wolf17_trfiles['wolf17'][0].trtimes)[:493,:]
                else:
                    downsampled_features = lanczosinterp2D(features, read_file['word_times'], wolf_trfiles['wolf'][0].trtimes)
            start_index = read_file.shape[0]
            print(downsampled_features.shape)
            temp_data.append([downsampled_features])
        train_data.append(np.hstack(temp_data)) 
    train_data = np.hstack(train_data)
    return train_data[0]

def load_test_word_data(eachlayer):
    test_data = []
    temp_data = []
    start_index = 0
    for i in np.arange(5):
        if i+1<10:
            read_file = pd.read_csv("./life_captions/life0"+str(i+1)+'_words.csv')
            read_file[['start_time', 'end_time']] = read_file[['start_time', 'end_time']].apply(pd.to_numeric)
            read_file['word_times'] = (read_file['start_time']+read_file['end_time'])/2
            features = np.array(life_dataset.item()[eachlayer])[start_index:start_index+read_file.shape[0],:]
            # you will need `wordseqs` from the notebook
            downsampled_features = lanczosinterp2D(features, read_file['word_times'], life_trfiles['life'][0].trtimes)
        temp_data.append([downsampled_features])
        start_index = read_file.shape[0]
    test_data.append(np.hstack(temp_data))  
    test_data = np.hstack(test_data)[:,:2008,:]
    return test_data[0]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("subjectNum", help="Choose subject", type = int)
    parser.add_argument("modality", help="Choose modality", type = str)
    parser.add_argument("story", help="Choose story", type = str)
    parser.add_argument("outputdir", help="Output directory", type = str)
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
    for eachlayer in np.arange(12):
        Rstim, Pstim = load_word_data(args.story, eachlayer), load_test_word_data(eachlayer)
        # Delay stimuli
        from util import make_delayed
        ndelays = 5
        delays = range(1, ndelays+1)

        print ("FIR model delays: ", delays)

        delRstim = []
        for ss in np.arange(1):
            delRstim.append(make_delayed(np.array(Rstim[:,:]), delays))
            
        delPstim = []
        for ss in np.arange(1):
            delPstim.append(make_delayed(np.array(Pstim[:,:]), delays))

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

