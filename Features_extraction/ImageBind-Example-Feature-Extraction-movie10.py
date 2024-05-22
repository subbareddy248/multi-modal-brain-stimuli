#!/usr/bin/env python
# coding: utf-8

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import pandas as pd
import os
import natsort
import numpy as np
import argparse

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description = "CheXpert NN argparser")
    parser.add_argument("sessionnum", help="Choose session", type = str)
    parser.add_argument("devicenum", help="Choose cuda device", type = int)
    parser.add_argument("layer", help="Choose layernum", type = int)
    args = parser.parse_args()
    device = "cuda:"+str(args.devicenum) if torch.cuda.is_available() else "cpu"

    video_paths = natsort.natsorted(os.listdir('../bourne_videos/bourne'+args.sessionnum+'/'))
    audio_paths = natsort.natsorted(os.listdir('../bourne_videos/bourne'+args.sessionnum+'/'))

    # Instantiate model
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    text_session = []
    video_session = []
    audio_session = []

    for i in np.arange(int(len(video_paths)/2)):
        if  i%50:
            print(i)
        video_list = ['./bourne_videos/bourne'+args.sessionnum+'/part_'+str(i+1)+'.mp4']
        audio_list = ['./bourne_videos/bourne'+args.sessionnum+'/part_'+str(i+1)+'.wav']
        
        # Load data
        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(video_list, device, clips_per_video=30),
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_list, device, clips_per_video=30),
        }

        with torch.no_grad():
            embeddings = model(inputs)
            
        video_session.append(embeddings['vision'].cpu().detach().numpy())
        audio_session.append(embeddings['audio'].cpu().detach().numpy())


    np.save('bourn_video_'+args.sessionnum,np.array(video_session))
    np.save('bourn_audio_'+args.sessionnum,np.array(audio_session))
