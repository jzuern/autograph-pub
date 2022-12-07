import random

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
import cv2
import skfmm
from ray.util.multiprocessing import Pool


def edge_feature_encoder(out_features=64, in_channels=3):
    model = resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model


def preprocess_sample(G_gt_nx, sat_image_, global_sdf, global_angles, centerline_image_, roi_xxyy, sample_id, out_path):

    margin = 200

    node_gt_score = []
    node_pos_feats = []

    for node in G_gt_nx.nodes:
        node_gt_score.append(G_gt_nx.nodes[node]["p"])
        node_pos_feats.append(G_gt_nx.nodes[node]["pos"])

    node_gt_score = torch.tensor(node_gt_score)
    node_pos_feats = torch.tensor(np.array(node_pos_feats))


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

        edge_img_feats.append(torch.ByteTensor(feats * 255.))

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
    edge_pos_feats = torch.cat(edge_pos_feats, dim=0).float()
    edge_img_feats = torch.ByteTensor(torch.cat(edge_img_feats, dim=0))
    edge_gt_score = torch.tensor(edge_gt_score)


    torch.save({
        "rgb": torch.FloatTensor(rgb),
        "sdf": torch.FloatTensor(global_sdf),
        "angles": torch.FloatTensor(global_angles),
        "node_feats": node_pos_feats,
        "edge_pos_feats": edge_pos_feats,
        "edge_img_feats": edge_img_feats,
        "edge_indices": edge_indices,
        "edge_scores": edge_gt_score,
        "node_scores": node_gt_score,
        "graph": G_gt_nx,
    }, os.path.join(out_path, "{}.pth".format(sample_id)))



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

    G = nx.DiGraph()

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
        p = gaussian(d, 0, r_min)
        #p = 1

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


def dijkstra_trajectories(G, trajectories, imsize):
    '''
    This function assigns the graph edge probabilities according to the dijkstra traversal along the recorded
    trajectories
    :param G: input graph
    :param trajectories: list of trajectories
    :return: updated graph, angle mask, sdf mask
    '''


    global_sdf = np.zeros(imsize, dtype=np.float32)
    global_angle = np.zeros(imsize, dtype=np.float32)
    global_mask = np.zeros((imsize[0], imsize[1], 3), dtype=np.uint8)

    for n in G.nodes:
        G.nodes[n]["log_odds_dijkstra"] = 0.0
    for e in G.edges:
        G.edges[e]["log_odds_dijkstra"] = 0.0

    for t in trajectories:

        # create sdf of trajectory
        sdf = np.zeros(imsize, dtype=np.float32)
        for i in range(len(t) - 1):
            x1 = int(t[i][0])
            y1 = int(t[i][1])
            x2 = int(t[i + 1][0])
            y2 = int(t[i + 1][1])
            cv2.line(sdf, (x1, y1), (x2, y2), 1, 1)

            angle = np.arctan2(y2 - y1, x2 - x1)
            cv2.line(global_angle, (x1, y1), (x2, y2), angle, thickness=5)
            cv2.line(global_mask, (x1, y1), (x2, y2), (1, 1, 1), thickness=5)


        f = 10  # distance function scale
        sdf = skfmm.distance(1 - sdf)
        sdf[sdf > f] = f
        sdf = sdf / f

        global_sdf = np.maximum(global_sdf, 1-sdf)


        directions_hsv = np.zeros_like(global_mask)
        directions_hsv[:, :, 0] = np.sin(global_angle) * 127 + 127
        directions_hsv[:, :, 1] = np.cos(global_angle) * 127 + 127
        directions_hsv[:, :, 2] = global_mask[:, :, 0] * 255

        directions_hsv = directions_hsv * global_mask


        # assign cost to nodes and edges
        for e in G.edges:
            start = G.nodes[e[0]]["pos"]
            end = G.nodes[e[1]]["pos"]
            midpoint = (start + end) / 2
            G.edges[e]["c"] = sdf[int(midpoint[1]), int(midpoint[0])] ** 0.5 + \
                              sdf[int(start[1]), int(start[0])] ** 0.5 +  \
                              sdf[int(end[1]), int(end[0])] ** 0.5

        start_node = np.argmin(np.linalg.norm(np.array([G.nodes[i]["pos"] for i in G.nodes]) - t[0], axis=1))
        end_node = np.argmin(np.linalg.norm(np.array([G.nodes[i]["pos"] for i in G.nodes]) - t[-1], axis=1))

        path = nx.dijkstra_path(G, start_node, end_node, weight="c")

        for i in range(len(path) - 1):
            G.edges[path[i], path[i+1]]["log_odds_dijkstra"] += 1
        for i in range(len(path)):
            G.nodes[path[i]]["log_odds_dijkstra"] += 1

    return G, global_sdf, directions_hsv



def resample_trajectory(trajectory, dist=5):
    '''
    Resample a trajectory to a fixed distance between points

    :param trajectory:
    :param dist:
    :return:
    '''

    new_trajectory = [trajectory[0]]
    curr_pos = trajectory[0]
    for i in range(1, len(trajectory)):
        dist_travelled = np.linalg.norm(trajectory[i] - curr_pos)
        if dist_travelled > dist:
            new_trajectory.append(trajectory[i])
            curr_pos = trajectory[i]
    return np.array(new_trajectory)


def process_chunk(roi_xxyy_list, trajectories_, lanes_, sat_image_, centerline_image_):

    for roi_xxyy in tqdm(roi_xxyy_list):
        sample_id = "{}-{}-{}".format(city_name, roi_xxyy[0], roi_xxyy[2])
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

            # resample trajectory to have equally distant points
            if len(trajectory) > 5:
                trajectory = resample_trajectory(trajectory, dist=5)
            if len(trajectory) > 5:
                trajectories.append(trajectory)

        if len(trajectories) < 1:
            #print("no trajectories in roi {}. skipping".format(roi_xxyy))
            continue

        r_min = 8  # minimum radius of the circle for poisson disc sampling
        G = initialize_graph(roi_xxyy, r_min=r_min)

        for trajectory in trajectories:

            # check length of trajectory
            if np.linalg.norm(trajectory[0] - trajectory[-1]) < 50:
                continue

            # Now we update the angular gridmap
            for i in range(len(trajectory) - 1):
                pos = trajectory[i]
                next_pos = trajectory[i + 1]
                angle = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
                G = bayes_update_graph(G, angle, x=pos[0], y=pos[1], p=0.9, r_min=r_min)

        # node_log_odds = np.array([G.nodes[n]["log_odds"] for n in G.nodes])
        # node_probabilities = np.exp(node_log_odds) / (1 + np.exp(node_log_odds))
        #
        # # perform angle kernel density estimation and peak detection
        G = angle_kde(G)
        #
        # edge_log_odds = np.array([G.edges[e]["log_odds"] for e in G.edges])
        # edge_probabilities = np.exp(edge_log_odds) / (1 + np.exp(edge_log_odds))

        # assign edge probabilities according to dijstra-approximated trajectories
        G, global_sdf, global_angles = dijkstra_trajectories(G, trajectories, imsize=sat_image.shape[:2])
        # edge_probabilities = np.array([G.edges[e]["p_dijkstra"] for e in G.edges])
        # node_probabilities = np.array([G.nodes[n]["p_dijkstra"] for n in G.nodes])

        log_odds_e = np.array([G.edges[e]["log_odds_dijkstra"] for e in G.edges])
        log_odds_n = np.array([G.nodes[n]["log_odds_dijkstra"] for n in G.nodes])

        node_probabilities = np.exp(log_odds_n) / (1 + np.exp(log_odds_n))
        edge_probabilities = np.exp(log_odds_e) / (1 + np.exp(log_odds_e))




        if np.count_nonzero(edge_probabilities[edge_probabilities > 0.5]) < 20:
            print("too few edges with high probability. skipping")
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
        plt.tight_layout()
        ax.set_aspect('equal')
        ax.imshow(sat_image)
        #ax.imshow(centerline_image, alpha=0.5)
        ax.imshow(global_sdf, alpha=0.3)
        ax.imshow(global_angles, alpha=0.3)

        Image.fromarray(sat_image).save("{}/{}-rgb.png".format(out_path, sample_id))
        Image.fromarray(global_angles).save("{}/{}-angles.png".format(out_path, sample_id))
        Image.fromarray(global_sdf * 255.).convert("L").save("{}/{}-sdf.png".format(out_path, sample_id))

        # draw edges
        for t in trajectories:
            for i in range(len(t) - 1):
                ax.arrow(t[i][0], t[i][1],
                         t[i + 1][0] - t[i][0],
                         t[i + 1][1] - t[i][1],
                         color="white", alpha=0.5, width=0.5, head_width=1, head_length=1)

        for n in G.nodes:
            angle_peaks = G.nodes[n]["angle_peaks"]
            pos = G.nodes[n]["pos"]
            for peak in angle_peaks:
                ax.arrow(pos[0], pos[1], np.cos(peak) * 3, np.sin(peak) * 3, color='r', width=0.3)

        edge_colors = cmap(norm(edge_probabilities))
        edge_colors[:, -1] = 0.5

        nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
                         edge_color=edge_colors,
                         node_color=node_colors,
                         with_labels=False,
                         node_size=10,
                         arrowsize=3.0,
                         width=1,
                         )

        plt.savefig("{}/{}.png".format(out_path, sample_id), dpi=400)

        # preprocess sample into pth file
        print("Processing {}...".format(sample_id))
        preprocess_sample(G,
                          sat_image_=sat_image_,
                          global_sdf=global_sdf,
                          global_angles=global_angles,
                          centerline_image_=centerline_image_,
                          roi_xxyy=roi_xxyy,
                          sample_id=sample_id,
                          out_path=out_path)



if __name__ == "__main__":

    city_name = "Pittsburgh"
    out_path = '/data/self-supervised-graph/preprocessed/av2-continuous'
    os.makedirs(out_path, exist_ok=True)

    sample_no = 0
    imsize = 0

    '''find /data/argoverse2/motion-forecasting -type f -wholename '/data/argoverse2/motion-forecasting/val/*/*.parquet' > scenario_files.txt '''


    sat_image_ = np.asarray(Image.open("/data/lanegraph/woven-data/{}.png".format(city_name))).astype(np.uint8)
    centerline_image_ = np.asarray(Image.open("/data/lanegraph/woven-data/{}_centerlines.png".format(city_name)))
    centerline_image_ = centerline_image_ / 255.0

    print("Satellite resolution: {}x{}".format(sat_image_.shape[1], sat_image_.shape[0]))



    # sat_image_ = 128 * np.ones((60000, 60000, 3), dtype=np.uint8)
    # centerline_image_ = np.zeros((60000, 60000), dtype=np.uint8)

    # generate roi_xxyy list over full satellite image in sliding window fashion
    meta_roi = [0, 25000, 0, 25000]   # ymin, ymax, xmin, xmax, ymin, ymax PIT

    sat_image_ = np.ascontiguousarray(sat_image_[meta_roi[0]:meta_roi[1], meta_roi[2]:meta_roi[3], :])
    centerline_image_ = np.ascontiguousarray(centerline_image_[meta_roi[0]:meta_roi[1], meta_roi[2]:meta_roi[3]])

    roi_xxyy_list = []
    for i in range(meta_roi[2], meta_roi[3], 100):
        for j in range(meta_roi[0], meta_roi[1], 100):
            roi_xxyy_list.append(np.array([j, j + 256, i, i + 256]))
    random.shuffle(roi_xxyy_list)

    all_scenario_files = np.loadtxt("/home/zuern/self-supervised-graph/scenario_files.txt", dtype=str).tolist()

    [R, c, t] = get_transform_params(city_name.lower())

    if not os.path.exists("lanes_{}.npy".format(city_name)) or not os.path.exists("trajectories_{}.npy".format(city_name)):
        print("Generating trajectories and gt-lanes")
        trajectories_ = []
        lanes_ = []

        for scenario_path in tqdm(all_scenario_files):
            scenario_path = Path(scenario_path)
            scenario_id = scenario_path.stem.split("_")[-1]
            static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)

            if scenario.city_name != city_name.lower():
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
        np.save("trajectories_{}.npy".format(city_name), trajectories_)
        np.save("lanes_{}.npy".format(city_name), lanes_)
    else:
        trajectories_ = np.load("trajectories_{}.npy".format(city_name), allow_pickle=True)
        lanes_ = np.load("lanes_{}.npy".format(city_name), allow_pickle=True)

        ts = []
        for trajectory in trajectories_:
            if np.mean(trajectory[:, 0]) < meta_roi[3] and np.mean(trajectory[:, 1]) < meta_roi[1] and \
                    np.mean(trajectory[:, 0]) > meta_roi[2] and np.mean(trajectory[:, 1]) > meta_roi[0]:
                ts.append(trajectory)
        trajectories_ = np.array(ts)

        # plot
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ax.set_aspect('equal')
        # for trajectory in trajectories_[0:100000]:
        #     ax.plot(trajectory[:, 0], trajectory[:, 1], c="r")
        # plt.show()




    # single core
    #process_chunk(roi_xxyy_list, trajectories_, lanes_, sat_image_, centerline_image_)


    # multi core
    def chunkify(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    num_cpus = 8
    num_samples = len(roi_xxyy_list)
    num_chunks = int(np.ceil(num_samples / num_cpus))
    roi_chunks = list(chunkify(roi_xxyy_list, n=num_chunks))

    from itertools import repeat

    arguments = zip(roi_chunks,
                    repeat(trajectories_),
                    repeat(lanes_),
                    repeat(sat_image_),
                    repeat(centerline_image_))

    Pool(num_cpus).starmap(process_chunk, arguments)



