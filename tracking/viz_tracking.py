from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from nuscenes import NuScenes
import json
import time
from nuscenes.utils import splits
import numpy as np
import copy
import pickle
from scipy.optimize import linear_sum_assignment
from glob import glob
import matplotlib.pyplot as plt



# Visualize tracklets
predictions_files = sorted(glob("/data/argoverse2-full/*/*/*/tracking.pickle"))

city_name = "MIA"


fig, ax = plt.subplots()
ax.set_aspect('equal')

for p in predictions_files:
    with open(p, "rb") as f:
        av2_annos = pickle.load(f)

    tracklets = {}

    ego_pos = np.array([p.translation for p in av2_annos["ego_pos"]])
    ax.plot(ego_pos[:, 0], ego_pos[:, 1], "k-")

    if av2_annos["city_name"] == city_name:

        for counter, anno in enumerate(av2_annos["results"].keys()):
            for t in av2_annos["results"][anno]:
                t_id = t["tracking_id"]
                t_trans = t["translation"]
                if t_id in tracklets.keys():
                    tracklets[t_id].append(t_trans)
                else:
                    tracklets[t_id] = [t_trans]

        for tracklet in tracklets:
            traj = np.array(tracklets[tracklet])
            ax.plot(traj[:, 0], traj[:, 1])
plt.show()

