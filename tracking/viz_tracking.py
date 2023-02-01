
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
norm = Normalize(vmin=0, vmax=10)


city_name_dict ={
    "PIT": "pittsburgh",
    "MIA": "miami",
    "ATX": "austin"
}


# Get tracking file locations
tracking_files = glob('/data/argoverse2-full/tracking-results/*_tracking.pickle')

city_name = "PIT"
[transform_R, transform_c, transform_t] = get_transform_params(city_name_dict[city_name])


fig, ax = plt.subplots()
ax.set_aspect('equal')


for fcounter, p in tqdm(enumerate(tracking_files)):
    if fcounter > 1000:
        break


    with open(p, "rb") as f:
        try:
            av2_annos = pickle.load(f)
        except Exception as e:
            print(e)
            continue

    tracklets = {}

    if av2_annos["city_name"] == city_name:

        ego_pos = np.array([p.translation for p in av2_annos["ego_pos"]])

        # Coordinate transformation
        bb = np.hstack([ego_pos[:, 0:2], np.zeros((len(ego_pos), 1))])
        tmp = transform_t[np.newaxis, :] + transform_c * np.einsum('jk,ik', transform_R, bb)
        ego_pos = tmp[:, 0:2]

        ax.plot(ego_pos[:, 0], ego_pos[:, 1], "k-")

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

            traj = np.array(tracklet.path)
            traj_type = tracklet.label

            print("Type: ", traj_type)

            traj = traj[:, 0:2]

            if traj_type == 1:  # vehicle
                continue

                # Based on overall length
                total_length = np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1))
                if total_length < 15:
                    continue

                # Based on start end diff
                start_end_diff = np.linalg.norm(traj[0,:] - traj[-1, :])
                if start_end_diff < 10:
                    continue

                # Based on number of timesteps
                if len(traj) < 10:
                    continue

                # Remove big jumps in trajectory
                if np.max(np.linalg.norm(traj[1:] - traj[:-1], axis=1)) > 5:
                    continue

            elif traj_type == 2:  # pedestrian
                pass

            # Also smooth them
            #traj_filtered = np.asarray(get_bezier_parameters(traj[:, 0], traj[:, 1], degree=4))

            # Coordinate transformation
            bb = np.hstack([traj, np.zeros((len(traj), 1))])
            tmp = transform_t[np.newaxis, :] + transform_c * np.einsum('jk,ik', transform_R, bb)
            traj = tmp[:, 0:2]

            tracklet.path = traj

            tracklets_filtered.append(tracklet)


        for tracklet in tracklets_filtered:
            ax.plot(tracklet.path[:, 0], tracklet.path[:, 1], alpha=0.7, color=cmap(norm(tracklet.label)))
        ax.legend(["Ego", "Pedestrian", "Vehicle"])


plt.show()

