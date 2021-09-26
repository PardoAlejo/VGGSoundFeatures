import h5py as h5
import numpy as np
import os.path as osp
import glob
import tqdm
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--suffix',
        default='audio_16',
        type=str)
    parser.add_argument(
        '--features_path',
        type=str)
    parser.add_argument(
        '--out_path',
        type=str)
    parser.add_argument(
        '--out_name',
        default='audio_feats',
        type=str)
    return parser.parse_args()

def npy2h5(features_path, suffix, out_path, out_name):
    videos = glob.glob(f'{features_path}/*_{suffix}.npy')
    print(f'{len(videos)} features found')
    features = []
    names = []
    for video in tqdm.tqdm(videos):
        feature = np.load(open(video,'rb'))
        features.append(feature)
        name = osp.basename(video).replace(f'_{suffix}.npy','')
        names.append(name)

    print('Saving hdf5 file')
    with h5.File(f'{out_path}/{out_name}.h5','w') as f:
        for name, feature in tqdm.tqdm(zip(names, features), total=len(names)):
            f.create_dataset(name, data=feature, chunks=True)

if __name__ == "__main__":
    args = get_arguments()
    npy2h5(features_path=args.features_path, suffix=args.suffix, out_path=args.out_path, out_name=args.out_name)
