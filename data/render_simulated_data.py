import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import pprint

root_dir = "/data/self-supervised-graph/"

# get lidar files
lidar_files = glob(os.path.join(root_dir, "lidar_render", "*.png"))

# get vehicle_pos_files
pos_files = glob(os.path.join(root_dir, "vehicles_pos", "*.txt"))

# generate trajectories from vehicle_pos_files
vehicles_pos_dict = {}

for pos_file in pos_files:
    with open(pos_file, "r") as f:
        time_step = int(pos_file.split("/")[-1].split(".")[0])
        lines = f.readlines()
        for line in lines:
            vehicle_id, x, y, z = line.strip().split(",")
            if vehicle_id not in vehicles_pos_dict:
                vehicles_pos_dict[vehicle_id] = [[time_step, float(x), float(y), float(z)]]
            else:
                vehicles_pos_dict[vehicle_id].append([time_step, float(x), float(y), float(z)])

# sort vehicle dict by time_step
for vehicle_id in vehicles_pos_dict:
    vehicles_pos_dict[vehicle_id] = sorted(vehicles_pos_dict[vehicle_id], key=lambda x: x[0])

pprint.pprint(vehicles_pos_dict)


# plot trajectories
fig, ax = plt.subplots()
ax.set_aspect('equal')

for vehicle_id in vehicles_pos_dict:
    vehicle_pos = np.array(vehicles_pos_dict[vehicle_id])
    ax.plot(vehicle_pos[:, 1], vehicle_pos[:, 2], label=vehicle_id)
ax.legend()
plt.show()



