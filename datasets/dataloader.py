import os
import json
import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from PIL import Image
import glob
import sys
from scipy import signal
import random
import ffmpeg
from ffmpeg import Error
import pandas as pd
from datetime import timedelta
import glob

class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        
        self.fps = 24
        self.stride = 8
        self.win_size = 16
        self.rate = 44100
        self.seconds_per_snippet = self.win_size/self.fps

        self.mode = mode
        self.transforms = transforms

        # initialize audio transform
        self._init_atransform()
        #  Retrieve list of audio and video files
        self.video_files = []
        self._set_video_files(os.path.join(args.csv_path, args.paths_video))
        self.durations = {}
        self._set_video_duration(os.path.join(args.csv_path, args.duration_csv))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def _set_video_files(self, paths_video):
        # paths_df = pd.read_csv(paths_csv)
        self.video_files = glob.glob(f'{paths_video}/*.mp4')
        print(f'# of audio files = {len(self.video_files):d} ')

    def _set_video_duration(self, duration_csv):
        duration_df = pd.read_csv(duration_csv)
        self.durations = {row[1].videoid:int(row[1].duration) for row in duration_df.iterrows()}

    def __len__(self):
        return len(self.video_files)  

    def extract_audio(self, filename):    
        try:
            out, err = (
                ffmpeg
                .input(filename)
                .output('-', format='f32le', acodec='pcm_f32le', ac=1, ar=str(self.rate))
                .run(capture_stdout=True, capture_stderr=True)
            )
        except Error as err:
            print(err.stderr)
            raise
        
        return np.frombuffer(out, np.float32)


    def __getitem__(self, idx):
        mp4file = self.video_files[idx]
        print(f'{mp4file}')
        # import ipdb; ipdb.set_trace()
        video_name = os.path.splitext(os.path.basename(mp4file))[0]
        video_spectograms = []
        video_samples = []

        # Audio
        # import ipdb; ipdb.set_trace()
        sample = self.extract_audio(mp4file)
        duration = self.durations[video_name]
        sample = sample[0:duration*self.rate] # Remove last 30 seconds
        padded_duration = int(duration + (self.win_size/self.fps - duration%(self.win_size/self.fps)))
        sound_stride = int((self.stride/self.fps)*self.rate)
        sound_win_size = (self.win_size/self.fps)*self.rate
        for i, time_stamp in enumerate(range(0, int(padded_duration*self.rate)-sound_stride, sound_stride)):
            this_start = int(time_stamp)
            this_end = int(this_start + sound_win_size)
            this_sample = sample[this_start:this_end].copy()
            # repeat in case audio is too short
            # if (sample.shape[0]-this_start) < sound_win_size:
            #     num_to_pad = int(np.ceil(sound_win_size - (sample.shape[0]-this_start)))
            #     this_sample = np.pad(this_sample, 
            #                         [(0, num_to_pad)],
            #                         'constant')
            if i>1 and not this_sample.shape[-1] == video_samples[-1].shape[-1]:
                
                num_to_pad = video_samples[-1].shape[-1] - this_sample.shape[-1]
                this_sample = np.pad(this_sample, 
                                    [(0, num_to_pad)],
                                    'constant')
            this_sample[this_sample > 1.] = 1.
            this_sample[this_sample < -1.] = -1.
            video_samples.append(this_sample)
            frequencies, times, spectrogram = signal.spectrogram(this_sample, self.rate, nperseg=512,noverlap=353)
            spectrogram = np.log(spectrogram+ 1e-7)

            mean = np.mean(spectrogram)
            std = np.std(spectrogram)
            spectrogram = np.divide(spectrogram-mean,std+1e-9)
            video_spectograms.append(spectrogram)
        try:
            out = np.array(video_spectograms)
        except:
            import ipdb; ipdb.set_trace()
        return video_name, torch.tensor(np.array(video_spectograms)), mp4file
    
    def collate_fn(self, data_lst):
        video_names = [_video_name for _video_name, _, _ in data_lst]
        video_name_to_slice = {}
        video_name_to_path = {}
        current_ind = 0
        for this_video_name, this_features, this_video_path in data_lst:
            video_name_to_slice[this_video_name] = (current_ind, current_ind + this_features.shape[0])
            current_ind += this_features.shape[0]
            video_name_to_path[this_video_name] = this_video_path
        features    = torch.cat([_features for _, _features, _ in data_lst],dim=0)

        targets = {'video-names': video_names,
                   'video-name-to-slice': video_name_to_slice,
                   'video-name-to-path': video_name_to_path, 
                   }
        return features, targets

