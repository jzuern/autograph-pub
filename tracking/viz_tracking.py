
import numpy as np
import pickle
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from data.av2.settings import get_transform_params
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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


class Tracklet(object):
    def __init__(self, label):
        self.label = label
        self.path = []
        self.timesteps = []


def filter_tracklet(tracklet):

    traj = np.array(tracklet.path)
    traj_type = tracklet.label

    traj = traj[:, 0:2]

    # ALL UNITS IN m
    if traj_type in [1, 2, 3, 4, 5, 7]:  # vehicle
        tracklet.label = 1
        MIN_TRAJ_LEN = 15
        MIN_START_END_DIFF = 10
        MIN_TIMESTEPS = 10
        MAX_JUMPSIZE = 2
    else:
        tracklet.label = 2
        MIN_TRAJ_LEN = 5
        MIN_START_END_DIFF = 5
        MIN_TIMESTEPS = 10
        MAX_JUMPSIZE = 1

    # Based on overall length
    total_length = np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1))
    if total_length < MIN_TRAJ_LEN:
        return None

    # Based on start end diff
    start_end_diff = np.linalg.norm(traj[0, :] - traj[-1, :])
    if start_end_diff < MIN_START_END_DIFF:
        return None

    # Based on number of timesteps
    if len(traj) < MIN_TIMESTEPS:
        return None

    # Remove big jumps in trajectory
    if np.max(np.linalg.norm(traj[1:] - traj[:-1], axis=1)) > MAX_JUMPSIZE:
        return None

    # Coordinate transformation
    bb = np.hstack([traj, np.zeros((len(traj), 1))])
    tmp = transform_t[np.newaxis, :] + transform_c * np.einsum('jk,ik', transform_R, bb)
    traj = tmp[:, 0:2]

    tracklet.path = traj

    return tracklet



for fcounter, p in tqdm(enumerate(tracking_files)):
    if fcounter > 1000:
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

        for counter, tracklet in enumerate(tracklets):

            tracklet = tracklets[tracklet]
            tracklet = filter_tracklet(tracklet)

            if tracklet is not None:
                tracklets_filtered.append(tracklet)

        for tracklet in tracklets_filtered:
            axarr[0].plot(tracklet.path[:, 0], tracklet.path[:, 1], alpha=0.7, color=cmap(norm(tracklet.label)))

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

