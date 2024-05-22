#!/usr/bin/env python
# coding: utf-8

# In[41]:


import os
import pickle
from transformers import TvltProcessor, TvltModel
import torch
from torchvision.io import read_video
import torchaudio
from natsort import natsorted
import numpy as np

# In[43]:

device = "cuda:0" if torch.cuda.is_available() else "cpu"
def process_video_audio(video_path, audio_path):
    # Load and preprocess the video
    video_frames, _, _ = read_video(video_path, pts_unit='sec')
    # Ensure the number of frames is within the model's limit
    max_frames = model.config.num_frames
    num_frames = video_frames.shape[0]
    if num_frames > max_frames:
        indices = torch.linspace(0, num_frames - 1, steps=max_frames).long()
        video_frames = video_frames[indices]
    video_inputs = processor(images=video_frames.permute(0, 3, 1, 2), return_tensors="pt")

    # Load and preprocess the audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    audio_inputs = processor(audio=waveform, sampling_rate=sample_rate, return_tensors="pt")

    # Extracting features using the model
    with torch.no_grad():
        outputs = model(pixel_values=video_inputs['pixel_values'].to(device), audio_values=audio_inputs['audio_values'].to(device), output_hidden_states=True)
    video_embedding = outputs.last_pixel_hidden_state[0].mean(0).cpu().detach().numpy()
    audio_embedding = outputs.last_audio_hidden_state[0].mean(0).cpu().detach().numpy()
    joint_embedding = []
    for eachlayer in np.arange(12):
        joint_embedding.append(outputs['hidden_states'][eachlayer][0].mean(0).cpu().detach().numpy())

    return video_embedding, audio_embedding, np.array(joint_embedding) 


# In[3]:


moviesnames = ['bourne01','bourne02', 'bourne03', 'bourne04', 'bourne05', 'bourne06', 'bourne07', 'bourne08', 'bourne09', 'bourne10']
moviesnames = ['wolf01','wolf02', 'wolf03', 'wolf04', 'wolf05', 'wolf06', 'wolf07', 'wolf08', 'wolf09', 'wolf10',
              'wolf11','wolf12','wolf13','wolf14','wolf15','wolf16','wolf17']

moviesnames = ['life01','life02', 'life03', 'life04', 'life05']

# In[42]:


processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base", cache_dir='./')

model = TvltModel.from_pretrained("ZinengTang/tvlt-base", cache_dir='./')
model.to(device)

# In[9]:


# waveform = AudioSegment.from_wav('./bourne_videos/bourne01/part_1.wav')
# waveform = waveform.set_channels(1)
# waveform = waveform.get_array_of_samples()


# In[40]:


# container = av.open('./bourne_videos/bourne01/part_2.mp4')
# # sample 16 frames
# indices = sample_frame_indices(clip_len=1, frame_sample_rate=16, seg_len=container.streams.video[0].frames)
# video = read_video_pyav(container, indices)


# In[24]:


# input_dict = processor(list(video), waveform, sampling_rate=44100, return_tensors="pt")


# In[47]:


def process_folder(folder_path):
    video_embeddings = []
    audio_embeddings = []
    joint_embeddings = {}
    joint_embeddings[folder_path.split('/')[2]]=[]
    file_pairs = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):
            base_filename = filename[:-4]
            video_path = os.path.join(folder_path, filename)
            audio_path = os.path.join(folder_path, base_filename + '.wav')
            if os.path.exists(audio_path):
                file_pairs.append((video_path, audio_path))

    # Sort the file pairs in natural order
    file_pairs = natsorted(file_pairs)

    for video_path, audio_path in file_pairs:
        video_embedding, audio_embedding, joint_embedding = process_video_audio(video_path, audio_path)
        video_embeddings.append(video_embedding)
        audio_embeddings.append(audio_embedding)
        joint_embeddings[folder_path.split('/')[2]].append(joint_embedding)
    # Save embeddings to pickle files
    with open(os.path.join('.', 'video_embeddings_'+folder_path.split('/')[2]+'.pkl'), 'wb') as f:
        pickle.dump(video_embeddings, f)
    with open(os.path.join('.', 'audio_embeddings_'+folder_path.split('/')[2]+'.pkl'), 'wb') as f:
        pickle.dump(audio_embeddings, f)
    with open(os.path.join('.', 'joint_embeddings_'+folder_path.split('/')[2]+'.pkl'), 'wb') as f:
        pickle.dump(joint_embeddings, f)


# In[48]:


for eachmovie in moviesnames[3:]:
    process_folder('./life_videos/'+eachmovie)
