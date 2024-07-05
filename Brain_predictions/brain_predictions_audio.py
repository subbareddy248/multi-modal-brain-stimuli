#!/usr/bin/env python
# coding: utf-8

# In[25]:


from matplotlib.pyplot import figure, cm
import numpy as np
import logging
import argparse
import os
from npp import zscore
import h5py
from interpdata import lanczosinterp2D


# In[11]:


# f = open('bourn.report','w')
# f.write('0.0 sound-start\n')
# for i in np.arange(1,405):
#     f.write(str(i*1.49)+' trigger\n')
# f.close()


# In[33]:


# f = open('wolf.report','w')
# f.write('0.0 trigger\n')
# f.write('0.0 sound-start\n')
# for i in np.arange(1,406):
#     f.write(str(i*1.49)+' trigger\n')
# f.write(str(i*1.49)+' sound-stop\n')
# f.close()


# In[55]:


# f = open('wolf17.report','w')
# f.write('0.0 trigger\n')
# f.write('0.0 sound-start\n')
# for i in np.arange(1,497):
#     f.write(str(i*1.49)+' trigger\n')
# f.write(str(i*1.49)+' sound-stop\n')
# f.close()


# In[71]:


# f = open('life.report','w')
# f.write('0.0 trigger\n')
# f.write('0.0 sound-start\n')
# for i in np.arange(1,406):
#     f.write(str(i*1.49)+' trigger\n')
# f.write(str(i*1.49)+' sound-stop\n')
# f.close()


# In[26]:


class TRFile(object):
    def __init__(self, trfilename, expectedtr=1.5):
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
        badtrtimes = np.nonzero(itrtimes>(itrtimes.mean()*1.5))[0]
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


# In[72]:


trfiles = load_generic_trfiles(['bourn'])
wolf_trfiles = load_generic_trfiles(['wolf'])
wolf17_trfiles = load_generic_trfiles(['wolf17'])
life_trfiles = load_generic_trfiles(['life'])


# In[28]:


from pathlib import Path
import numpy as np

chunk_sz, context_sz = 0.1, 16.0
model = 'wav2vec2-base'

base_features_path = Path(f"features_cnk{chunk_sz:0.1f}_ctx{context_sz:0.1f}/{model}")


# In[70]:


from npp import zscore
def load_audio_data(story, eachlayer):
    train_data = []
    if story=='bourn':
        temp_data = []
        for i in np.arange(10):
            if i+1<10:
                times = np.load(base_features_path / f"bourne0{i+1}_times.npz")['times'][:,1] # shape: (time,)
                features = np.load(base_features_path / f"layer.{eachlayer}" / f"bourne0{i+1}.npz")['features'] # shape: (time, model dim.)
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, times, trfiles['bourn'][0].trtimes)
            else:
                times = np.load(base_features_path / f"bourne{i+1}_times.npz")['times'][:,1] # shape: (time,)
                features = np.load(base_features_path / f"layer.{eachlayer}" / f"bourne{i+1}.npz")['features'] # shape: (time, model dim.)
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, times, trfiles['bourn'][0].trtimes)[:379,:]
            temp_data.append([downsampled_features])
        train_data.append(np.hstack(temp_data))
    elif story=='wolf':
        temp_data = []
        for i in np.arange(17):
            if i+1<10:
                times = np.load(base_features_path / f"wolf0{i+1}_times.npz")['times'][:,1] # shape: (time,)
                features = np.load(base_features_path / f"layer.{eachlayer}" / f"wolf0{i+1}.npz")['features'] # shape: (time, model dim.)
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, times, wolf_trfiles['wolf'][0].trtimes)
            else:
                times = np.load(base_features_path / f"wolf{i+1}_times.npz")['times'][:,1] # shape: (time,)
                features = np.load(base_features_path / f"layer.{eachlayer}" / f"wolf{i+1}.npz")['features'] # shape: (time, model dim.)
                # you will need `wordseqs` from the notebook
                if i+1==17:
                    downsampled_features = lanczosinterp2D(features, times, wolf17_trfiles['wolf17'][0].trtimes)[:493,:]
                else:
                    downsampled_features = lanczosinterp2D(features, times, wolf_trfiles['wolf'][0].trtimes)
            print(downsampled_features.shape)
            temp_data.append([downsampled_features])
        train_data.append(np.hstack(temp_data))
    elif story=='all':
        temp_data = []
        for i in np.arange(10):
            if i+1<10:
                times = np.load(base_features_path / f"bourne0{i+1}_times.npz")['times'][:,1] # shape: (time,)
                features = np.load(base_features_path / f"layer.{eachlayer}" / f"bourne0{i+1}.npz")['features'] # shape: (time, model dim.)
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, times, trfiles['bourn'][0].trtimes)
            else:
                times = np.load(base_features_path / f"bourne{i+1}_times.npz")['times'][:,1] # shape: (time,)
                features = np.load(base_features_path / f"layer.{eachlayer}" / f"bourne{i+1}.npz")['features'] # shape: (time, model dim.)
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, times, trfiles['bourn'][0].trtimes)[:379,:]
            temp_data.append([downsampled_features])
        train_data.append(np.hstack(temp_data))
        temp_data = []
        for i in np.arange(17):
            if i+1<10:
                times = np.load(base_features_path / f"wolf0{i+1}_times.npz")['times'][:,1] # shape: (time,)
                features = np.load(base_features_path / f"layer.{eachlayer}" / f"wolf0{i+1}.npz")['features'] # shape: (time, model dim.)
                # you will need `wordseqs` from the notebook
                downsampled_features = lanczosinterp2D(features, times, wolf_trfiles['wolf'][0].trtimes)
            else:
                times = np.load(base_features_path / f"wolf{i+1}_times.npz")['times'][:,1] # shape: (time,)
                features = np.load(base_features_path / f"layer.{eachlayer}" / f"wolf{i+1}.npz")['features'] # shape: (time, model dim.)
                # you will need `wordseqs` from the notebook
                if i+1==17:
                    downsampled_features = lanczosinterp2D(features, times, wolf17_trfiles['wolf17'][0].trtimes)[:493,:]
                else:
                    downsampled_features = lanczosinterp2D(features, times, wolf_trfiles['wolf'][0].trtimes)
            temp_data.append([downsampled_features])
        train_data.append(np.hstack(temp_data))  
    train_data = np.hstack(train_data)
    return train_data[0]


# In[87]:


def load_test_audio_data(eachlayer):
    test_data = []
    temp_data = []
    for i in np.arange(5):
        if i+1<10:
            times = np.load(base_features_path / f"life0{i+1}_times.npz")['times'][:,1] # shape: (time,)
            features = np.load(base_features_path / f"layer.{eachlayer}" / f"life0{i+1}.npz")['features'] # shape: (time, model dim.)
            # you will need `wordseqs` from the notebook
            downsampled_features = lanczosinterp2D(features, times, life_trfiles['life'][0].trtimes)
        temp_data.append([downsampled_features])
    test_data.append(np.hstack(temp_data))  
    test_data = np.hstack(test_data)[:,:2008,:]
    return test_data[0]


# In[4]:


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


# In[48]:


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
    for eachlayer in np.arange(args.layer,10):
        Rstim, Pstim = load_audio_data(args.story, eachlayer), load_test_audio_data(eachlayer)
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
