#!/usr/bin/env python
# coding: utf-8

from transformers import AutoImageProcessor, ViTModel, ViTMAEForPreTraining, AutoFeatureExtractor, AutoModel

import torch
from PIL import Image
import cv2
import numpy as np
from natsort import natsorted
import os


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def video2frame(video):
    frames = []
    ret = True
    while ret:
        ret, img = video.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    videoframes = np.stack(frames, axis=0) # dimensions (T, H, W, C)
    return videoframes

extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k",cache_dir="./")
model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k",cache_dir="./")
model.eval()
model.to(device)

moviesnames = ['wolf01','wolf02', 'wolf03', 'wolf04', 'wolf05', 'wolf06', 'wolf07', 'wolf08', 'wolf09', 'wolf10',
              'wolf11','wolf12','wolf13','wolf14','wolf15','wolf16','wolf17']


bourne_dict = {}
for eachmovie in moviesnames[0:6]:
    print(eachmovie)
    bourne_dict[eachmovie] = []
    files = natsorted(os.listdir('../wolf_videos/'+eachmovie))
    for eachclip in files:
        if '.mp4' in eachclip:
            vidcap = cv2.VideoCapture('../wolf_videos/'+eachmovie+'/'+eachclip)
            videoframes = video2frame(vidcap)
            temp1 = []
            for j in np.arange(videoframes.shape[0]):
                inputs = extractor(images=videoframes[j], return_tensors="pt")
                inputs = inputs.to(device)
                with torch.no_grad():
                    embeddings = model(**inputs, output_hidden_states = True)
                temp = []
                for eachlayer in np.arange(12):
                    temp.append(embeddings['hidden_states'][eachlayer][0].mean(0).cpu().detach().numpy())
                temp1.append(np.array(temp))
            bourne_dict[eachmovie].append(np.array(temp1).mean(0))
    np.save(eachmovie+'_vith',np.array(bourne_dict[eachmovie]))

