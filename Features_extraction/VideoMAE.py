#!/usr/bin/env python
# coding: utf-8

import av

import numpy as np

from transformers import AutoImageProcessor, VideoMAEModel, VivitModel, VivitImageProcessor
from natsort import natsorted
import os
import cv2
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def read_video_pyav(container, indices):

    '''

    Decode the video with PyAV decoder.

    Args:

        container (`av.container.input.InputContainer`): PyAV container.

        indices (`List[int]`): List of frame indices to decode.

    Returns:

        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).

    '''

    frames = []

    container.seek(0)

    start_index = indices[0]

    end_index = indices[-1]

    for i, frame in enumerate(container.decode(video=0)):

        if i > end_index:

            break

        if i >= start_index and i in indices:

            frames.append(frame)

    return np.stack([cv2.resize(frame.to_ndarray(format="rgb24"),(224,224)) for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):

    '''

    Sample a given number of frame indices from the video.

    Args:

        clip_len (`int`): Total number of frames to sample.

        frame_sample_rate (`int`): Sample every n-th frame.

        seg_len (`int`): Maximum allowed index of sample's last frame.

    Returns:

        indices (`List[int]`): List of sampled frame indices

    '''

    converted_len = int(clip_len * frame_sample_rate)

    end_idx = np.random.randint(converted_len, seg_len)

    start_idx = end_idx - converted_len

    indices = np.linspace(start_idx, end_idx, num=clip_len)

    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

    return indices


moviesnames = ['bourne01','bourne02', 'bourne03', 'bourne04', 'bourne05', 'bourne06', 'bourne07', 'bourne08', 'bourne09', 'bourne10']


image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400", cache_dir='./')

model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", cache_dir='./')
model.to(device)

bourne_dict = {}
for eachmovie in moviesnames[3:]:
    print(eachmovie)
    bourne_dict[eachmovie] = []
    files = natsorted(os.listdir('../bourne_videos/'+eachmovie))
    for eachclip in files:
        if '.mp4' in eachclip:
            container = av.open('../bourne_videos/'+eachmovie+'/'+eachclip)
            # sample 16 frames
            indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video = read_video_pyav(container, indices)
            inputs = image_processor(list(video), return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs, output_hidden_states = True)
            temp = []
            for eachlayer in np.arange(12):
                temp.append(outputs['hidden_states'][eachlayer][0].mean(0).cpu().detach().numpy())
            bourne_dict[eachmovie].append(np.array(temp))
    np.save(eachmovie+'_vivit',np.array(bourne_dict[eachmovie]))
