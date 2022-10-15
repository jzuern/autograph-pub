import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



ind_root_dir = '/home/zuern/Desktop/inD/data/'

scaling_factor = {
    1: 10,
    4: 6.55,
}


for recording_id in range(30):

    img = np.asarray(Image.open(ind_root_dir + '{:02d}_background.png'.format(recording_id)))
    plt.imshow(img)

    meta = pd.read_csv(ind_root_dir + '{:02d}_recordingMeta.csv'.format(recording_id))
    # orthoPxToMeter = meta['orthoPxToMeter'].values[0]
    location_id = meta['locationId'].values[0]

    print(recording_id, location_id)

    csv_file = ind_root_dir + '{:02d}_tracks.csv'.format(recording_id)
    tracks = pd.read_csv(csv_file, index_col=0)
    tracks_xy = tracks[['xCenter', 'yCenter']].values
    tracks_xy[:, 1] = -tracks_xy[:, 1]
    tracks_xy *= scaling_factor[location_id]


    tracks_id = np.array(tracks[['trackId']].values).flatten()
    tracks_id_unique = np.unique(tracks_id)
    for t in tracks_id_unique:
        track = tracks_xy[tracks_id == t]
        plt.plot(track[:,0], track[:,1])


    plt.show()