import h5py as h5
import numpy as np
import os.path as osp
import glob
import tqdm
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feat_type',
        default='audio',
        type=str)
    return parser.parse_args()

def npy2h5(feat_type='audio'):
    features_path = ('/home/pardogl/scratch/data/movies/youtube/*')
    videos = glob.glob(f'{features_path}/*_{feat_type}.npy')
    print(f'{len(videos)} features found')
    features = []
    names = []
    for video in tqdm.tqdm(videos):
        feature = np.load(open(video,'rb'))
        features.append(feature)
        name = osp.basename(video).replace('_audio.npy','')
        names.append(name)

    print('Saving hdf5 file')
    with h5.File(osp.join(features_path, f'/home/pardogl/scratch/data/movies/ResNet-18_{feat_type}_features.h5'),'w') as f:
        for name, feature in tqdm.tqdm(zip(names, features), total=len(names)):
            f.create_dataset(name, data=feature, chunks=True)

if __name__ == "__main__":
    args = get_arguments()
    npy2h5(args.feat_type)
