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
from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap


def edge_feature_encoder(out_features=64, in_channels=3):
    model = resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model


def preprocess_sample(G_gt_nx, sat_image_, centerline_image_, roi_xxyy, sample_no, out_path):

    margin = 200

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

    rgb_context = sat_image_[roi_xxyy[0] - margin:roi_xxyy[1] + margin,
                             roi_xxyy[2] - margin:roi_xxyy[3] + margin].copy()

    rgb = sat_image_[roi_xxyy[0] : roi_xxyy[1],
                     roi_xxyy[2] : roi_xxyy[3]]

    centerline_context = centerline_image_[roi_xxyy[0] - margin:roi_xxyy[1] + margin,
                                           roi_xxyy[2] - margin:roi_xxyy[3] + margin].copy().astype(np.float32)


    for edge_idx, edge in enumerate(G_gt_nx.edges):
        i, j = edge
        s_x, s_y = G_gt_nx.nodes[i]["pos"]
        e_x, e_y = G_gt_nx.nodes[j]["pos"]

        delta_x, delta_y = e_x - s_x, e_y - s_y
        mid_x, mid_y = s_x + delta_x / 2, s_y + delta_y / 2

        edge_len = np.sqrt(delta_x ** 2 + delta_y ** 2)
        edge_angle = np.arctan(delta_y / (delta_x + 1e-6))

        center = np.array([mid_x + margin, mid_y + margin])

        crop_img_rgb = get_oriented_crop(edge_angle, center[0], center[1], rgb_context)
        crop_centerline = get_oriented_crop(edge_angle, center[0], center[1], centerline_context)

        crop_img_rgb_ = transform2vgg(crop_img_rgb).unsqueeze(0).numpy()
        crop_centerline_ = transform2vgg(crop_centerline).unsqueeze(0).numpy()
        feats = np.concatenate([crop_img_rgb_, crop_centerline_], axis=1)

        edge_img_feats.append(torch.tensor(feats))

        # print(center)
        # print(delta_x, delta_y)
        # print(rgb_context.shape)
        # print(crop_img_rgb.shape)
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(rgb_context)
        # ax[1].imshow(rgb)
        # ax[2].imshow(crop_img_rgb)

        # # plot edges
        # for e in G_gt_nx.edges:
        #     ax[0].plot([G_gt_nx.nodes[e[0]]["pos"][0] + margin, G_gt_nx.nodes[e[1]]["pos"][0] + margin],
        #                [G_gt_nx.nodes[e[0]]["pos"][1] + margin, G_gt_nx.nodes[e[1]]["pos"][1] + margin], "b")
        #     ax[1].plot([G_gt_nx.nodes[e[0]]["pos"][0] , G_gt_nx.nodes[e[1]]["pos"][0] ],
        #                   [G_gt_nx.nodes[e[0]]["pos"][1] , G_gt_nx.nodes[e[1]]["pos"][1] ], "b")
        #
        # ax[0].plot([s_x + margin, e_x + margin], [s_y + margin, e_y + margin], color="red")
        # ax[1].plot([s_x , e_x ], [s_y , e_y ], color="red")
        #
        #
        # plt.show()

        edge_tensor = torch.tensor([edge_angle, edge_len, mid_x, mid_y]).reshape(1, -1)
        edge_pos_feats.append(edge_tensor)
        edge_indices.append((i, j))
        edge_gt_score.append(G_gt_nx.edges[i, j]["p"])

    edge_indices = torch.tensor(edge_indices)
    edge_pos_feats = torch.cat(edge_pos_feats, dim=0)
    edge_img_feats = torch.cat(edge_img_feats, dim=0)
    edge_gt_score = torch.tensor(edge_gt_score)


    torch.save({
        "rgb": torch.FloatTensor(rgb),
        "node_feats": node_pos_feats,
        "edge_pos_feats": edge_pos_feats,
        "edge_img_feats": edge_img_feats,
        "edge_indices": edge_indices,
        "edge_scores": edge_gt_score,
        "node_scores": node_gt_score,
        "graph": G_gt_nx,
    }, os.path.join(out_path, "{:04d}.pth".format(sample_no)))



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

    roi_shape = roi_xxyy[3] - roi_xxyy[2], roi_xxyy[1] - roi_xxyy[0]

    points = poisson_disk_sampling(r_min=r_min,
                                   width=roi_shape[1],
                                   height=roi_shape[0])
    points = np.array(points)
    edges = get_random_edges(points,
                             min_point_dist=r_min,
                             max_point_dist=2*r_min)

    G = nx.Graph()

    for i in range(len(points)):
        G.add_node(i, pos=points[i],
                   log_odds=0.0,
                   angle_observations=[])
    for i in range(len(edges)):
        G.add_edge(edges[i][0], edges[i][1])

    return G


def bayes_update_graph(G, angle, x, y, p, r_min):

    pos = np.array([G.nodes[i]["pos"] for i in G.nodes])
    distances = cdist(pos, np.array([[x, y]]))

    # get closest nodes
    closest_nodes = np.argwhere(distances < r_min).flatten()

    for node in closest_nodes:
        node_id = list(G.nodes)[node]
        d = distances[node][0]

        #p = gaussian(d, 0, r_min)
        p = 1

        G.nodes[node]["angle_observations"].append(angle)
        G.nodes[node_id]["log_odds"] = G.nodes[node_id]["log_odds"] + p

    return G


def assign_centerline_probs(G, centerline):

    for node in G.nodes:
        pos = G.nodes[node]["pos"]
        G.nodes[node]["p"] = centerline[int(pos[1]), int(pos[0])] ** 2

    for edge in G.edges:
        midpoint = (G.nodes[edge[0]]["pos"] + G.nodes[edge[1]]["pos"]) / 2
        centerline_val = centerline[int(midpoint[1]), int(midpoint[0])]  ** 2
        G.edges[edge]["p"] = centerline_val

    return G

def angle_kde(G):

    for e in G.edges:
        G.edges[e]["log_odds"] = 0.0

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
            angle_peaks = G.nodes[n]["angle_peaks"]
            neighbor_edge_vec = G.nodes[neighbor]["pos"] - G.nodes[n]["pos"]
            neighbor_edge_angle = np.arctan2(neighbor_edge_vec[1], neighbor_edge_vec[0])

            for angle_peak in angle_peaks:
                rel_angle = np.abs(angle_peak - neighbor_edge_angle)
                if rel_angle < np.pi / 4:
                    G.edges[n, neighbor]["log_odds"] += np.pi / 4 - rel_angle

    return G



def get_scenario_centerlines(static_map_path):

    avm = ArgoverseStaticMap.from_json(Path(static_map_path))

    centerlines = []

    for lane_id, lane_obj in avm.vector_lane_segments.items():

        right_lane_boundary = lane_obj.right_lane_boundary.xyz
        left_lane_boundary = lane_obj.left_lane_boundary.xyz
        centerline, length = compute_midpoint_line(left_lane_boundary, right_lane_boundary)
        centerline = np.array(centerline)
        centerline = centerline[:, :2]

        centerlines.append(centerline)


    return np.array(centerlines)



if __name__ == "__main__":

    city_name = "austin"
    sample_no = 0
    imsize = 0

    '''find /data/argoverse2/motion-forecasting -type f -wholename '/data/argoverse2/motion-forecasting/val/*/*.parquet' > scenario_files.txt '''


    sat_image_ = np.asarray(Image.open("/data/lanegraph/woven-data/Austin.png"))
    centerline_image_ = np.asarray(Image.open("/data/lanegraph/woven-data/Austin_centerlines.png"))
    centerline_image_ = centerline_image_ / 255.0

    # sat_image_ = 128 * np.ones((60000, 60000, 3), dtype=np.uint8)
    # centerline_image_ = np.zeros((60000, 60000), dtype=np.uint8)

    # generate roi_xxyy list over full satellite image in sliding window fashion
    meta_roi = [25000, 35000, 15000, 25000]
    roi_xxyy_list = []
    for i in range(meta_roi[2], meta_roi[3], 256):
        for j in range(meta_roi[0], meta_roi[1], 256):
            roi_xxyy_list.append(np.array([j, j + 256, i, i + 256]))

    all_scenario_files = np.loadtxt("/home/zuern/self-supervised-graph/scenario_files.txt", dtype=str).tolist()

    [R, c, t] = get_transform_params(city_name)

    if not os.path.exists("lanes.npy"):
        trajectories_ = []
        lanes_ = []

        for scenario_path in tqdm(all_scenario_files):
            scenario_path = Path(scenario_path)
            scenario_id = scenario_path.stem.split("_")[-1]
            static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)

            if scenario.city_name != city_name:
                continue

            scenario_lanes = get_scenario_centerlines(static_map_path)

            for lane in scenario_lanes:
                for i in range(len(lane)):
                    bb = np.array([lane[i, 0], lane[i, 1], 0])
                    tmp = t + c * R @ bb
                    lane[i] = tmp[0:2]

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

            for lane in scenario_lanes:
                lanes_.append(lane)

        trajectories_ = np.array(trajectories_)
        lanes_ = np.array(lanes_)

        # save trajectories
        np.save("trajectories.npy", trajectories_)
        np.save("lanes.npy", lanes_)
    else:
        trajectories_ = np.load("trajectories.npy", allow_pickle=True)
        #trajectories_ = np.load("lanes.npy", allow_pickle=True)

        ts = []
        for trajectory in trajectories_:
            if np.mean(trajectory[:, 0]) < meta_roi[3] and np.mean(trajectory[:, 1]) < meta_roi[1] and \
                    np.mean(trajectory[:, 0]) > meta_roi[2] and np.mean(trajectory[:, 1]) > meta_roi[0]:
                ts.append(trajectory)
        trajectories_ = np.array(ts)
        # print(len(trajectories_))
        #
        # for t in trajectories_:
        #     plt.scatter(t[:, 0], t[:, 1], s=0.1, c="blue")
        # plt.show()
    meta_roi

    for roi_xxyy in roi_xxyy_list:
        sat_image = sat_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3], :].copy()
        centerline_image = centerline_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3]].copy()

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

        if len(trajectories) < 1:
            print("no trajectories in roi {}. skipping".format(roi_xxyy))
            continue

        print("number of trajectories in roi: ", len(trajectories))


        # r_min = 5  # minimum radius of the circle for poisson disc sampling


        # for trajectory in trajectories:
        #     G = initialize_graph(roi_xxyy, r_min=r_min)
        #
        #     # Now we update the angular gridmap
        #     for i in range(len(trajectory) - 1):
        #         pos = trajectory[i]
        #         next_pos = trajectory[i + 1]
        #         angle = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
        #         G = bayes_update_graph(G, angle, x=pos[0], y=pos[1], p=0.9, r_min=r_min)
        #
        #     node_log_odds = np.array([G.nodes[n]["log_odds"] for n in G.nodes])
        #     node_probabilities = np.exp(node_log_odds) / (1 + np.exp(node_log_odds))
        #     for i, n in enumerate(G.nodes):
        #         G.nodes[n]["p"] = node_probabilities[i]
        #
        #     for e in G.edges:
        #         G.edges[e]["log_odds"] = G.nodes[e[0]]["log_odds"] + G.nodes[e[1]]["log_odds"]
        #     edge_log_odds = np.array([G.edges[e]["log_odds"] for e in G.edges])
        #     edge_probabilities = np.exp(edge_log_odds) / (1 + np.exp(edge_log_odds))
        #     for i, e in enumerate(G.edges):
        #         G.edges[e]["p"] = edge_probabilities[i]
        #
        #     if np.count_nonzero(edge_log_odds[edge_log_odds > 1]) < 10:
        #         print("no edge with high probability. skipping")
        #         continue
        #
        #     # rescale probabilities
        #     #node_probabilities = (node_probabilities - np.min(node_probabilities)) / (
        #     #            np.max(node_probabilities) - np.min(node_probabilities))
        #     #edge_probabilities = (edge_probabilities - np.min(edge_probabilities)) / (
        #     #            np.max(edge_probabilities) - np.min(edge_probabilities))
        #
        #     cmap = plt.get_cmap('viridis')
        #     norm = plt.Normalize(vmin=0.0, vmax=1.0)
        #     node_colors = cmap(norm(node_probabilities))
        #
        #     fig, ax = plt.subplots(figsize=(10, 10))
        #     ax.set_aspect('equal')
        #     ax.imshow(sat_image)
        #     #ax.imshow(centerline_image, alpha=0.5)
        #
        #     # draw edges
        #     for t in trajectories:
        #         ax.plot(t[:, 0], t[:, 1], 'rx', markersize=1)
        #
        #     edge_colors = cmap(norm(edge_probabilities))
        #
        #     nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
        #                      edge_color=edge_colors,
        #                      node_color=node_colors,
        #                      with_labels=False,
        #                      node_size=10,
        #                      arrowsize=3.0,
        #                      width=1,
        #                      )
        #
        #     out_path = '/data/self-supervised-graph/preprocessed/av2'
        #     plt.savefig("{}/{:04d}.png".format(out_path, sample_no))
        #
        #     # preprocess sample into pth file
        #     preprocess_sample(G,
        #                       sat_image_=sat_image_,
        #                       centerline_image_=centerline_image_,
        #                       roi_xxyy=roi_xxyy,
        #                       sample_no=sample_no,
        #                       out_path=out_path)
        #     sample_no += 1
        #
        # G = initialize_graph(roi_xxyy, r_min=r_min)

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

        r_min = 8  # minimum radius of the circle for poisson disc sampling
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

        node_log_odds = np.array([G.nodes[n]["log_odds"] for n in G.nodes])
        node_probabilities = np.exp(node_log_odds) / (1 + np.exp(node_log_odds))

        # perform angle kernel density estimation and peak detection
        G = angle_kde(G)

        edge_log_odds = np.array([G.edges[e]["log_odds"] for e in G.edges])
        edge_probabilities = np.exp(edge_log_odds) / (1 + np.exp(edge_log_odds))

        if np.count_nonzero(edge_probabilities[edge_probabilities > 0.5]) < 2:
            print("no edge with high probability. skipping")
            continue

        # rescale probabilities
        node_probabilities = (node_probabilities - np.min(node_probabilities)) / (np.max(node_probabilities) - np.min(node_probabilities))
        edge_probabilities = (edge_probabilities - np.min(edge_probabilities)) / (np.max(edge_probabilities) - np.min(edge_probabilities))

        # assign probabilities to edges
        for i, e in enumerate(G.edges):
            G.edges[e]["p"] = edge_probabilities[i]
        # assign probabilities to nodes
        for i, n in enumerate(G.nodes):
            G.nodes[n]["p"] = node_probabilities[i]

        # # ignore all before and just assign centerline probs
        # G = assign_centerline_probs(G, centerline_image)
        #
        # # remove all nodes with low probability
        # G_ = G.copy()
        # for n in G_.nodes:
        #     if G.nodes[n]["p"] < 0.5:
        #         G.remove_node(n)
        #
        # # remap node ids and edges
        # node_ids = list(G.nodes)
        # node_id_map = {node_ids[i]: i for i in range(len(node_ids))}
        # G = nx.relabel_nodes(G, node_id_map)


        node_probabilities = np.array([G.nodes[n]["p"] for n in G.nodes])
        edge_probabilities = np.array([G.edges[e]["p"] for e in G.edges])

        if np.any(np.isnan(node_probabilities)) or np.any(np.isnan(edge_probabilities)):
            print("nan in node or edge probabilities. skipping")
            continue

        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=0.0, vmax=1.0)
        node_colors = cmap(norm(node_probabilities))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.imshow(sat_image)
        #ax.imshow(centerline_image, alpha=0.5)

        # draw edges
        for t in trajectories:
            ax.plot(t[:, 0], t[:, 1], 'rx', markersize=1)

        for n in G.nodes:
            angle_peaks = G.nodes[n]["angle_peaks"]
            pos = G.nodes[n]["pos"]
            for peak in angle_peaks:
                ax.arrow(pos[0], pos[1], np.cos(peak) * 3, np.sin(peak) * 3, color='r', width=0.3)

        edge_colors = cmap(norm(edge_probabilities))

        nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
                         edge_color=edge_colors,
                         node_color=node_colors,
                         with_labels=False,
                         node_size=10,
                         arrowsize=3.0,
                         width=1,
                         )

        out_path = '/data/self-supervised-graph/preprocessed/av2'
        plt.savefig("{}/{:04d}.png".format(out_path, sample_no))

        # preprocess sample into pth file
        preprocess_sample(G,
                          sat_image_=sat_image_,
                          centerline_image_=centerline_image_,
                          roi_xxyy=roi_xxyy,
                          sample_no=sample_no,
                          out_path=out_path)
        sample_no += 1
