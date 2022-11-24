import numpy as np
from pathlib import Path
from av2.datasets.motion_forecasting import scenario_serialization
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from data.av2.settings import get_transform_params
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2334477275000
import os
from lanegnn.utils import poisson_disk_sampling, get_random_edges, get_oriented_crop, transform2vgg
import networkx as nx
from scipy.spatial.distance import cdist
import torch
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from torchvision.models import resnet18
import torch.nn as nn



def edge_feature_encoder(out_features=64, in_channels=3):
    model = resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model


def preprocess_sample(G_gt_nx, rgb, sample_no, rgb_encoder):


    node_gt_score = []
    node_pos_feats = []

    for node in G_gt_nx.nodes:
        node_gt_score.append(G_gt_nx.nodes[node]["p"])
        node_pos_feats.append(G_gt_nx.nodes[node]["pos"])

    node_gt_score = torch.tensor(node_gt_score)
    node_pos_feats = torch.tensor(node_pos_feats)


    edge_pos_feats = []
    edge_indices = []
    edge_gt_score = []
    edge_img_feats = []

    for edge_idx, edge in enumerate(G_gt_nx.edges):
        i, j = edge
        s_x, s_y = G_gt_nx.nodes[i]["pos"]
        e_x, e_y = G_gt_nx.nodes[j]["pos"]

        delta_x, delta_y = e_x - s_x, e_y - s_y
        mid_x, mid_y = s_x + delta_x / 2, s_y + delta_y / 2

        edge_len = np.sqrt(delta_x ** 2 + delta_y ** 2)
        edge_angle = np.arctan(delta_y / (delta_x + 1e-6))

        crop_img_rgb = get_oriented_crop(edge_angle, mid_x, mid_y, rgb)
        crop_img_rgb = transform2vgg(crop_img_rgb).unsqueeze(0)

        #crop_img_feat = rgb_encoder(crop_img_rgb).detach()
        #edge_img_feats.append(crop_img_feat)
        edge_img_feats.append(crop_img_rgb)

        edge_tensor = torch.tensor([edge_angle, edge_len, mid_x, mid_y]).reshape(1, -1)
        edge_pos_feats.append(edge_tensor)
        edge_indices.append((i, j))

        edge_gt_score.append(G_gt_nx.edges[i, j]["p"])

    edge_indices = torch.tensor(edge_indices)
    edge_pos_feats = torch.cat(edge_pos_feats, dim=0)
    edge_img_feats = torch.cat(edge_img_feats, dim=0)
    edge_gt_score = torch.tensor(edge_gt_score)

    output_dir = "/data/self-supervised-graph/av2"

    print("saving to {}/*.pth".format(output_dir))

    torch.save({
        "rgb": torch.FloatTensor(rgb),
        "node_feats": node_pos_feats,
        "edge_pos_feats": edge_pos_feats,
        "edge_img_feats": edge_img_feats,
        "edge_indices": edge_indices,
        "edge_scores": edge_gt_score,
        "node_scores": node_gt_score,
        "graph": G_gt_nx,
    }, os.path.join(output_dir, "{:04d}.pth".format(sample_no)))



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



def initialize_graph(roi_xxyy, r_min):

    s = roi_xxyy[3] - roi_xxyy[2], roi_xxyy[1] - roi_xxyy[0]

    points = poisson_disk_sampling(r_min=r_min,
                                   width=s[1],
                                   height=s[0])
    points = np.array(points)
    edges = get_random_edges(points,
                             min_point_dist=r_min,
                             max_point_dist=2*r_min)

    G = nx.Graph()

    for i in range(len(points)):
        G.add_node(i, pos=points[i],
                   p=0.5,
                   angle_observations=[])
    for i in range(len(edges)):
        G.add_edge(edges[i][0], edges[i][1], p=0.5)


    return G

def bayes_update_graph(G, angle, x, y, p, r_min):


    pos = np.array([G.nodes[i]["pos"] for i in G.nodes])
    distances = cdist(pos, np.array([[x, y]]))

    # get closest nodes
    closest_nodes = np.argwhere(distances < r_min).flatten()

    for node in closest_nodes:

        d = distances[node][0]

        p_scaled = 0.5 + 0.5 * gaussian(d, 0, 1)

        node = list(G.nodes)[node]

        p_node = G.nodes[node]["p"]
        p_node = p_node * p_scaled / (p_node * p_scaled + (1 - p_node) * (1 - p_scaled))

        G.nodes[node]["angle_observations"].append(angle)
        G.nodes[node]["p"] = p_node


    return G


def angle_kde(G):

    for e in G.edges:
        G.edges[e]["p"] = 0.5

    for n in G.nodes:
        angle_observations = np.array(G.nodes[n]["angle_observations"])
        if len(angle_observations) > 0:

            kde = KernelDensity(kernel="gaussian", bandwidth=0.25).fit(angle_observations.reshape(-1, 1))
            angles_dense = np.linspace(-np.pi, np.pi, 100)
            log_dens = kde.score_samples(angles_dense.reshape(-1, 1))

            peaks, _ = find_peaks(np.exp(log_dens), height=0.2)
            G.nodes[n]["angle_peaks"] = angles_dense[peaks]
        else:
            G.nodes[n]["angle_peaks"] = []



    # edge scores
    for n in G.nodes:
        for neighbor in G.neighbors(n):
            p_edge = G.edges[n, neighbor]["p"]


            angle_peaks = G.nodes[n]["angle_peaks"]
            neighbor_edge_vec = G.nodes[neighbor]["pos"] - G.nodes[n]["pos"]
            neighbor_edge_angle = np.arctan2(neighbor_edge_vec[1], neighbor_edge_vec[0])

            for angle_peak in angle_peaks:
                if np.abs(neighbor_edge_angle - angle_peak) < np.pi / 4:
                    p_edge = p_edge * 0.8 / (p_edge * 0.8 + (1 - p_edge) * 0.2)
                    G.edges[n, neighbor]["p"] = p_edge



        # if len(angle_observations > 50):
        #
        #     fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        #     ax.scatter(angle_observations, np.zeros_like(angle_observations), c="k", s=20, zorder=10)
        #     ax.plot(angles_dense, np.exp(log_dens))
        #     ax.scatter(angles_dense[peaks], np.exp(log_dens)[peaks], c="r", s=20, zorder=10)
        #     plt.show()



    return G





if __name__ == "__main__":

    city_name = "austin"
    sample_no = 0
    '''find /data/argoverse2/motion-forecasting -type f -wholename '/data/argoverse2/motion-forecasting/val/*/*.parquet' > scenario_files.txt '''

    rgb_encoder = edge_feature_encoder(in_channels=3, out_features=32)


    sat_image_ = np.asarray(Image.open("/data/lanegraph/woven-data/Austin.png"))

    roi_xxyy_list = [
        np.array([17000, 17200, 35300, 35500])
    ]

    # generate roi_xxyy list over full satellite image in sliding window fashion
    roi_xxyy_list = []
    for i in range(17000, 20000, 200):
        for j in range(35300, 37000, 200):
            roi_xxyy_list.append(np.array([j, j + 200, i, i + 200]))

    all_scenario_files = np.loadtxt("/home/zuern/self-supervised-graph/scenario_files.txt", dtype=str).tolist()

    [R, c, t] = get_transform_params(city_name)

    if not os.path.exists("trajectories.npy"):
        # all_scenario_files = all_scenario_files[0:1000]

        trajectories_ = []

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

                if track.object_type != "vehicle":
                    continue

                # Coordinate transformation
                for i in range(len(actor_trajectory)):
                    bb = np.array([actor_trajectory[i, 0], actor_trajectory[i, 1], 0])
                    tmp = t + c * R @ bb
                    actor_trajectory[i] = tmp[0:2]

                # ignore standing vehicles
                if np.linalg.norm(actor_trajectory[0] - actor_trajectory[-1]) < 5:
                    continue

                trajectories_.append(actor_trajectory)

        trajectories_ = np.array(trajectories_)

        # save trajectories
        np.save("trajectories.npy", trajectories_)
    else:
        trajectories_ = np.load("trajectories.npy", allow_pickle=True)

    for roi_xxyy in roi_xxyy_list:
        sat_image = sat_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3], :].copy()

        trajectories = []
        for t in range(len(trajectories_)):
            trajectory = trajectories_[t]
            trajectory = trajectory - np.array([roi_xxyy[2], roi_xxyy[0]])
            # filter trajectories according to current roi_xxyy

            is_in_roi = np.logical_and(trajectory[:, 0] > 0, trajectory[:, 0] < sat_image.shape[1])
            is_in_roi = np.logical_and(is_in_roi, trajectory[:, 1] > 0)
            is_in_roi = np.logical_and(is_in_roi, trajectory[:, 1] < sat_image.shape[0])
            trajectory = trajectory[is_in_roi]

            if len(trajectory) > 2:
                trajectories.append(trajectory)

        if len(trajectories) < 2:
            print("no trajectories in roi. skipping")
            continue

        print("number of trajectories in roi: ", len(trajectories))

        # No we start to bayes aggregate the trajectories
        # First we create a grid map, uniform prior
        grid_size = 1
        num_angle_bins = 16

        grid_map_occ = np.ones((int((roi_xxyy[3] - roi_xxyy[2]) / grid_size), int((roi_xxyy[1] - roi_xxyy[0]) / grid_size))) * 0.5
        grid_map_angle = np.ones((int((roi_xxyy[3] - roi_xxyy[2]) / grid_size), int((roi_xxyy[1] - roi_xxyy[0]) / grid_size), num_angle_bins)) * float(1 / num_angle_bins)

        print(grid_map_angle.shape, grid_map_angle.shape, sat_image.shape)

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


        r_min = 10  # minimum radius of the circle for poisson disc sampling
        G = initialize_graph(roi_xxyy, r_min=r_min)

        for trajectory in tqdm(trajectories):

            # check length of trajectory
            if np.linalg.norm(trajectory[0] - trajectory[-1]) < 50:
                continue

            # Now we update the angular gridmap
            for i in range(len(trajectory) - 1):
                pos = trajectory[i]
                next_pos = trajectory[i + 1]
                angle = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
                G = bayes_update_graph(G, angle, x=pos[0], y=pos[1], p=0.9, r_min=r_min)

        node_colors = np.array([G.nodes[n]["p"] for n in G.nodes])
        cmap = plt.get_cmap('jet')
        norm = plt.Normalize(vmin=0.5, vmax=1)
        node_colors = cmap(norm(node_colors))

        # perform angle kernel density estimation and peak detection
        G = angle_kde(G)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.imshow(sat_image)

        for n in G.nodes:
            angle_peaks = G.nodes[n]["angle_peaks"]
            pos = G.nodes[n]["pos"]
            for peak in angle_peaks:
                ax.arrow(pos[0], pos[1], np.cos(peak) * 3, np.sin(peak) * 3, color='r', width=0.3)

        edge_colors = np.array([G.edges[e]["p"] for e in G.edges])
        edge_colors = cmap(norm(edge_colors))

        nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
                         edge_color=edge_colors,
                         node_color=node_colors,
                         with_labels=False,
                         node_size=10,
                         arrowsize=3.0,
                         width=1,
                         )
        plt.savefig("/data/self-supervised-graph/av2/{:04d}.png".format(sample_no))

        sample = preprocess_sample(G, rgb=sat_image, sample_no=sample_no, rgb_encoder=rgb_encoder)
        sample_no += 1