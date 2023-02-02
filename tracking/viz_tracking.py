
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from data.av2.settings import get_transform_params
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from aggregation.utils import Tracklet, filter_tracklet

cmap = cm.Dark2
norm = Normalize(vmin=0, vmax=4)


city_name_dict ={
    "PIT": "pittsburgh",
    "MIA": "miami",
    "ATX": "austin"
}


# Get tracking file locations
tracking_files = glob('/data/argoverse2-full/*_tracking.pickle')

city_name = "PIT"
[transform_R, transform_c, transform_t] = get_transform_params(city_name_dict[city_name])


fig, axarr = plt.subplots(1, 2, figsize=(15, 8), sharex=True, sharey=True)
axarr[0].set_aspect('equal')
axarr[1].set_aspect('equal')


for fcounter, p in tqdm(enumerate(tracking_files), total=len(tracking_files)):

    if fcounter > 100:
        break

    with open(p, "rb") as f:
        try:
            av2_annos = pickle.load(f)
        except Exception as e:
            print(e)
            continue

    if av2_annos["city_name"] == city_name:

        ego_pos = np.array([p.translation for p in av2_annos["ego_pos"]])

        # Coordinate transformation
        bb = np.hstack([ego_pos[:, 0:2], np.zeros((len(ego_pos), 1))])
        tmp = transform_t[np.newaxis, :] + transform_c * np.einsum('jk,ik', transform_R, bb)
        ego_pos = tmp[:, 0:2]

        axarr[0].plot(ego_pos[:, 0], ego_pos[:, 1], color=cmap(norm(0)))
        axarr[1].plot(ego_pos[:, 0], ego_pos[:, 1], color=cmap(norm(0)))

        tracklets = {}

        for counter, anno in enumerate(av2_annos["results"].keys()):
            for t in av2_annos["results"][anno]:
                t_id = t["tracking_id"]
                t_trans = t["translation"]
                if t_id in tracklets.keys():
                    tracklets[t_id].path.append(t_trans)
                else:
                    tracklets[t_id] = Tracklet(label=t["label"])
                    tracklets[t_id].path.append(t_trans)

        # Now we filter tracklets and smooth them
        tracklets_filtered = []
        tracklets_unfiltered = []

        for counter, tracklet in enumerate(tracklets):
            t = tracklets[tracklet]
            t.path = np.array(t.path)

            if t.label in [1, 2, 3, 4, 5, 7]:  # vehicle
                t.label = 1
            else:
                t.label = 2  # pedestrian

            t.transform(transform_t, transform_c, transform_R)

            axarr[0].plot(t.path[:, 0], t.path[:, 1], alpha=0.7, color=cmap(norm(t.label)))

            t_f = filter_tracklet(t)

            if t_f is not None:
                axarr[1].plot(t_f.path[:, 0], t_f.path[:, 1], alpha=0.7, color=cmap(norm(t_f.label)))


axarr[0].legend(["Ego", "Pedestrian", "Vehicle"])
axarr[1].legend(["Ego", "Pedestrian", "Vehicle"])


plt.show()

