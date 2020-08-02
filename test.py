import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
import json
import argparse
import csv
from model import AVENet
from datasets import GetAudioVideoDataset




def get_arguments():
    parser = argparse.ArgumentParser()
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
        '--csv_path',
        default='./data/',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--paths_csv',
        default='video_path.csv',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--duration_csv',
        default='durations.csv',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--batch_size', 
        default=32, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=309,
        type=int,
        help=
        'Number of classes')
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
    return parser.parse_args() 



def main():
    args = get_arguments()

    # init network
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    model= AVENet(args) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    
    # load pretrained models
    checkpoint = torch.load(args.summaries)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print('load pretrained model.')

    # create dataloader
    testdataset = GetAudioVideoDataset(args,  mode='test')
    testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,num_workers=4,
                                collate_fn=testdataset.collate_fn)

    softmax = nn.Softmax(dim=1)
    print("Loaded dataloader.")

    model.eval()
    for step, (features, targets) in enumerate(testdataloader):
        print('%d / %d' % (step,len(testdataloader) - 1))
        spec = Variable(features[0:300,:,:]).cuda()
        aud_o, feats = model(spec.unsqueeze(1).float())
        import ipdb; ipdb.set_trace()
        # prediction = softmax(aud_o)

        for name in targets['video-names']:
            this_start = targets['video-name-to-slice'][name][0]
            this_end = targets['video-name-to-slice'][name][1]
            this_vid_feat = feats[this_start:this_end,:]
            np.save(targets['video-name-to-path'][name].replace('.mp4','_audio.npy'), this_vid_feat.cpu().data.numpy())


if __name__ == "__main__":
    main()

