import subprocess
import os
import glob
import numpy as np
import pandas as pd
import tqdm
def get_length(filename):
    result = subprocess.Popen(["ffprobe", filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    duration = None
    creation_time = None
    for x in result.stdout.readlines():
        x = x.decode("utf-8")
        if "Duration" in x:
            duration = x[13:23]
            if duration:
                break
    if not duration:
        print(filename)
        return 0
    h, m, s = duration.split(':')
    duration_s = int(h) * 3600 + int(m) * 60 + float(s)
    return np.ceil(duration_s)

if __name__ == "__main__":
    path = '/home/pardogl/datasets/movies/youtube/*/*.mp4'
    vids = glob.glob(path)
    names = []
    durations = []
    for vid in tqdm.tqdm(vids):
        video_name = os.path.splitext(os.path.basename(vid))[0]
        names.append(video_name)
        duration = get_length(vid)
        durations.append(duration)
    df = pd.DataFrame(list(zip(names, durations)), 
                        columns=['videoid','durations'])
    df.to_csv('data/durations.csv', index=False)