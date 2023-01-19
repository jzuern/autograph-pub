
import numpy as np
import copy
import pickle
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from bezier import get_bezier_parameters


# Visualize tracklets
predictions_files = sorted(glob("/data/argoverse2-full/*/*/*/tracking.pickle"))

city_name = "PIT"


fig, ax = plt.subplots()
ax.set_aspect('equal')

for p in tqdm(predictions_files):
    with open(p, "rb") as f:
        try:
            av2_annos = pickle.load(f)
        except Exception as e:
            print(e)
            continue

    tracklets = {}

    if av2_annos["city_name"] == city_name:

        ego_pos = np.array([p.translation for p in av2_annos["ego_pos"]])
        ax.plot(ego_pos[:, 0], ego_pos[:, 1], "k-")

        for counter, anno in enumerate(av2_annos["results"].keys()):
            for t in av2_annos["results"][anno]:
                t_id = t["tracking_id"]
                t_trans = t["translation"]
                if t_id in tracklets.keys():
                    tracklets[t_id].append(t_trans)
                else:
                    tracklets[t_id] = [t_trans]

        # Now we filter tracklets and smooth them
        tracklets_filtered = []
        tracklets_unfiltered = []

        for c, tracklet in enumerate(tracklets):
            traj = np.array(tracklets[tracklet])

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

            # Also smooth them
            traj_filtered = traj
            #traj_filtered = np.asarray(get_bezier_parameters(traj[:, 0], traj[:, 1], degree=4))

            tracklets_filtered.append(traj_filtered)
            tracklets_unfiltered.append(traj)


        for traj in tracklets_filtered:
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.7)
        #for traj in tracklets_unfiltered:
        #    ax.plot(traj[:, 0], traj[:, 1], c='r')

plt.show()

