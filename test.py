import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import glob
import argparse
import csv
from model import AVENet
from datasets import GetAudioVideoDataset

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_classes',
        default=309,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--summaries',
        default='./pretrained_models/vggsound_avgpool.pth.tar',
        type=str,
        help='Directory path of pretrained model')
    parser.add_argument(
        '--pool',
        default="avgpool",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--paths_video',
        default='/home/pardogl/scratch/data/movies/youtube/*/',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--duration_csv',
        default='data/durations.csv',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--batch_size', 
        default=1, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--window_size',
        default=16,
        type=int,
        help='Window size for feature extraction')
    parser.add_argument(
        '--gpu_number',
        default='0',
        type=str,
        help='Window size for feature extraction')
    return parser.parse_args() 



def main():
    args = get_arguments()

    # init network
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_number
    model= AVENet(args) 
    model = model.cuda()
    # load pretrained models
    checkpoint = torch.load(args.summaries)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('load pretrained model.')

    # create dataloader
    testdataset = GetAudioVideoDataset(args,  mode='test')
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=4)

    print("Loaded dataloader.")

    model.eval()

    existent_feats = glob.glob(f'{args.paths_video}/*_audio.npy')
    with torch.no_grad():
        for step, (name, spectograms, path) in enumerate(testdataloader):
            name = name[0]
            path = path[0]
            if not spectograms.numpy().any():
                print(f'Could not read {path}')
                continue
            if path.replace('.mp4', f'_audio_{args.window_size}.npy') in existent_feats:
                print(f'{name} skiped, features already computed')
                continue
            else:
                print(f'Computing features for {name}, {step:d} / {len(testdataloader) - 1:d}')
            sample = spectograms.permute(1,0,2,3)
            n_chunk = sample.shape[0]
            features = torch.FloatTensor(n_chunk, 512).fill_(0)
            n_iter = int(np.ceil(n_chunk / args.batch_size))
            for i in range(n_iter):
                min_ind = i * args.batch_size
                max_ind = (i + 1) * args.batch_size
                video_batch = sample[min_ind:max_ind].cuda()
                _, batch_feats = model(video_batch)
                features[min_ind:max_ind] = batch_feats.cpu()
            features = features.numpy()
            np.save(path.replace('.mp4',f'_audio_{args.window_size}.npy'), features)


if __name__ == "__main__":
    main()

