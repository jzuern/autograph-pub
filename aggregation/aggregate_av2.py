import numpy as np
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from data.av2.settings import get_transform_params
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2334477275000
import open3d as o3d
import os
from lanegnn.utils import poisson_disk_sampling, get_random_edges
import networkx as nx
from scipy.spatial.distance import cdist



def bayes_update_gridmap(map, x, y, p):
    try:
        map[y, x] = map[y, x] * p / (map[y, x] * p + (1 - map[y, x]) * (1 - p))
    except IndexError:
        pass
    return map

def bayes_update_gridmap_angle(map, angle, x, y, p):
    closest_angle = np.argmin(np.abs(angle - np.linspace(-np.pi, np.pi, map.shape[2])))

    map[y, x, closest_angle] = map[y, x, closest_angle] * p / (map[y, x, closest_angle] * p + (1 - map[y, x, closest_angle]) * (1 - p))

    return map


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))



def initialize_graph(roi_xxyy):

    s = roi_xxyy[3] - roi_xxyy[2], roi_xxyy[1] - roi_xxyy[0]
    r_min = 5

    points = poisson_disk_sampling(r_min=r_min, width=s[1], height=s[0])
    # edges = get_random_edges(points, min_point_dist=r_min, max_point_dist=2*r_min)

    G = nx.Graph()

    for i in range(len(points)):
        G.add_node(i, pos=points[i], p=0.5)
    # for i in range(len(edges)):
    #     G.add_edge(edges[i][0], edges[i][1])

    return G

def bayes_update_graph(G, angle, x, y, p):

    closest_angle = np.argmin(np.abs(angle - np.linspace(-np.pi, np.pi, 8)))

    pos = np.array([G.nodes[i]["pos"] for i in G.nodes])
    distances = cdist(pos, np.array([[x, y]]))

    # get closest nodes
    closest_nodes = np.argwhere(distances < 10).flatten()

    for node in closest_nodes:

        d = distances[node][0]

        p_scaled = 0.5 + 0.5 * gaussian(d, 0, 1)

        node = list(G.nodes)[node]

        p_node = G.nodes[node]["p"]
        p_node = p_node * p_scaled / (p_node * p_scaled + (1 - p_node) * (1 - p_scaled))

        nx.set_node_attributes(G, {node: {"p": p_node,
                                          "closest_angle": closest_angle}})

    return G


if __name__ == "__main__":

    city_name = "austin"

    #sat_image = np.asarray(Image.open("/data/lanegraph/woven-data/Austin.png"))
    roi_xxyy = np.array([16800, 17300, 35200, 35600])
    #sat_image = sat_image[roi_xxyy[2]:roi_xxyy[3], roi_xxyy[0]:roi_xxyy[1], :]

    '''find /data/argoverse2/motion-forecasting -type f -wholename '/data/argoverse2/motion-forecasting/val/*/*.parquet' > scenario_files.txt '''

    all_scenario_files = np.loadtxt("/home/zuern/self-supervised-graph/scenario_files.txt", dtype=str).tolist()
    #all_scenario_files = all_scenario_files[0:3000]

    [R, c, t] = get_transform_params(city_name)

    if not os.path.exists("trajectories.npy"):

        trajectories = []

        for scenario_path in tqdm(all_scenario_files):
            scenario_path = Path(scenario_path)
            scenario_id = scenario_path.stem.split("_")[-1]
            static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)

            if scenario.city_name != city_name:
                continue

            for track in scenario.tracks:
                # Get actor trajectory and heading history
                actor_trajectory = np.array([list(object_state.position) for object_state in track.object_states])
                actor_headings = np.array([object_state.heading for object_state in track.object_states])

                #actor_trajectory = actor_trajectory[::2]
                #actor_headings = actor_headings[::2]

                if track.object_type != "vehicle":
                    continue

                # Coordinate transformation
                for i in range(len(actor_trajectory)):
                    bb = np.array([actor_trajectory[i, 0], actor_trajectory[i, 1], 0])
                    tmp = t + c * R @ bb
                    actor_trajectory[i] = tmp[0:2] - roi_xxyy[::2]

                is_in_roi = np.logical_and(actor_trajectory[:, 0] > 0, actor_trajectory[:, 0] < roi_xxyy[1] - roi_xxyy[0])
                is_in_roi = np.logical_and(is_in_roi, actor_trajectory[:, 1] > 0)
                is_in_roi = np.logical_and(is_in_roi, actor_trajectory[:, 1] < roi_xxyy[3] - roi_xxyy[2])
                actor_trajectory = actor_trajectory[is_in_roi]

                if len(actor_trajectory) < 2:
                    continue

                # ignore standing vehicles
                if np.linalg.norm(actor_trajectory[0] - actor_trajectory[-1]) < 5:
                    continue


                trajectories.append(actor_trajectory)

        trajectories = np.array(trajectories)

        # save trajectories
        np.save("trajectories.npy", trajectories)

    else:
        trajectories = np.load("trajectories.npy", allow_pickle=True)


        # No we start to bayes aggregate the trajectories
        # First we create a grid map, uniform prior
        grid_size = 1
        num_angle_bins = 16

        grid_map_occ = np.ones((int((roi_xxyy[3] - roi_xxyy[2]) / grid_size), int((roi_xxyy[1] - roi_xxyy[0]) / grid_size))) * 0.5
        grid_map_angle = np.ones((int((roi_xxyy[3] - roi_xxyy[2]) / grid_size), int((roi_xxyy[1] - roi_xxyy[0]) / grid_size), num_angle_bins)) * float(1 / num_angle_bins)

        for trajectory in trajectories:
            for pos in trajectory:
                grid_map_occ = bayes_update_gridmap(grid_map_occ, x=int(pos[0] / grid_size), y=int(pos[1] / grid_size), p=0.9)

            # Now we update the angular gridmap
            for i in range(len(trajectory) - 1):
                pos = trajectory[i]
                next_pos = trajectory[i + 1]
                angle = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
                grid_map_angle = bayes_update_gridmap_angle(grid_map_angle, angle, x=int(pos[0] / grid_size), y=int(pos[1] / grid_size), p=0.9)

        # plt.imshow(grid_map_occ)
        # plt.show()
        #
        # am = np.argmax(grid_map_angle, axis=2)
        #
        # plt.imshow(am)
        # plt.show()
        #
        # # plotting
        # # for t in trajectories:
        # #     t = t / grid_size
        # #     plt.plot(t[:, 0], t[:, 1], color='b', linewidth=0.5)
        # #     for i in range(len(t)):
        # #         plt.plot(t[i, 0], t[i, 1], color='b', marker="o", markersize=2)
        #
        # multimodality = np.sum(grid_map_angle > 0.3, axis=2)
        # plt.imshow(multimodality)
        # plt.show()
        #
        #
        # # get entropy of grid map angle
        # entropy = -np.sum(grid_map_angle * np.log(grid_map_angle), axis=2)
        #
        # plt.imshow(entropy)
        # plt.show()

        G = initialize_graph(roi_xxyy)

        for trajectory in tqdm(trajectories):

            # check length of trajectory
            if np.linalg.norm(trajectory[0] - trajectory[-1]) < 50:
                continue

            # Now we update the angular gridmap
            for i in range(len(trajectory) - 1):
                pos = trajectory[i]
                next_pos = trajectory[i + 1]
                angle = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
                G = bayes_update_graph(G, angle, x=pos[0], y=pos[1], p=0.9)

        node_colors = np.array([G.nodes[n]["p"] for n in G.nodes])
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=0.5, vmax=1)
        node_colors = cmap(norm(node_colors))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')

        # ax.imshow(grid_map_occ, alpha=0.5)

        nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
                         edge_color="k",
                         node_color=node_colors,
                         with_labels=False,
                         node_size=10,
                         arrowsize=3.0,
                         width=0,
                         )
        plt.show()