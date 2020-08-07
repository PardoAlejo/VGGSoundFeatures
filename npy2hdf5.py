import h5py as h5
import numpy as np
import os.path as osp
import glob
import tqdm

source = 'youtube'
features_path = (f'~/scratch/data/movies/movies/{source}/')
feat_type = 'audio'
videos = glob.glob(f'{features_path}/*/*{feat_type}.npy')
print(f'{len(videos)} features found')
features = []
names = []
for video in tqdm.tqdm(videos):
    feature = np.load(open(video,'rb'))
    features.append(feature)
    name = osp.basename(video).replace('.npy','')
    names.append(name)

print('Saving hdf5 file')
with h5.File(osp.join(features_path, f'../ResNet-18_{source}_{feat_type}_features.h5'),'w') as f:
    for name, feature in tqdm.tqdm(zip(names, features), total=len(names)):
        f.create_dataset(name, data=feature, chunks=True)

