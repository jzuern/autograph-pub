import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2334477275000
import os
import networkx as nx
from scipy.spatial.distance import cdist
import torch
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from itertools import repeat
from ray.util.multiprocessing import Pool
import cv2
import skfmm
import argparse

from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization

from lanegnn.utils import poisson_disk_sampling, get_random_edges, get_oriented_crop, transform2vgg
from data.av2.settings import get_transform_params
from lanegnn.utils import visualize_angles

# random shuffle seed
random.seed(0)


def preprocess_sample(G_gt_nx, sat_image_, sdf, sdf_regressor, angles, angles_viz, angles_regressor, roi_xxyy,
                      sample_id, out_path, output_name):

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

    # From raw tracklets
    # sdf_context = (np.pad(sdf, margin, mode="constant", constant_values=0) * 255).astype(np.uint8)
    # angles_viz_context = (np.asarray([np.pad(angles_viz[..., i], margin, mode="constant", constant_values=0)
    #                                  for i in range(3)]).transpose(1, 2, 0)).astype(np.uint8)

    # From regressor
    sdf_context = (np.pad(sdf_regressor, margin, mode="constant", constant_values=0) * 255).astype(np.uint8)
    angles_viz_context = (np.asarray([np.pad(angles_regressor[..., i], margin, mode="constant", constant_values=0)
                                     for i in range(3)]).transpose(1, 2, 0)).astype(np.uint8)

    # fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
    # axarr[0].imshow(rgb_context)
    # axarr[1].imshow(sdf_context)
    # axarr[2].imshow(angles_viz_context)
    # plt.show()

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
        crop_sdf = get_oriented_crop(edge_angle, center[0], center[1], sdf_context)
        crop_angles_viz = get_oriented_crop(edge_angle, center[0], center[1], angles_viz_context)

        crop_img_rgb_ = transform2vgg(crop_img_rgb).unsqueeze(0).numpy()
        crop_sdf_ = transform2vgg(crop_sdf).unsqueeze(0).numpy()
        crop_angles_viz_ = transform2vgg(crop_angles_viz).unsqueeze(0).numpy()

        feats = np.concatenate([crop_img_rgb_, crop_sdf_, crop_angles_viz_], axis=1)
        #feats = crop_img_rgb_

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

    out_file = os.path.join(out_path, "{}-{}.pth".format(sample_id, output_name))
    torch.save({
        "rgb": torch.FloatTensor(rgb),
        "sdf": torch.FloatTensor(sdf),
        "angles": torch.FloatTensor(angles),
        "node_feats": node_pos_feats,
        "edge_pos_feats": edge_pos_feats,
        "edge_img_feats": edge_img_feats,
        "edge_indices": edge_indices,
        "edge_scores": edge_gt_score,
        "node_scores": node_gt_score,
        "graph": G_gt_nx,
    }, out_file)

    print("Saving to {}".format(out_file))



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
                             max_point_dist=3*r_min)

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
            cv2.line(global_angle, (x1, y1), (x2, y2), angle, thickness=12)
            cv2.line(global_mask, (x1, y1), (x2, y2), (1, 1, 1), thickness=5)

        f = 15  # distance function scale
        try:
            sdf = skfmm.distance(1 - sdf)
        except:
            continue
        sdf[sdf > f] = f
        sdf = sdf / f

        global_sdf = np.maximum(global_sdf, 1-sdf)

        # assign cost to nodes and edges
        for e in G.edges:
            start = G.nodes[e[0]]["pos"]
            end = G.nodes[e[1]]["pos"]
            midpoint = (start + end) / 2
            try:
                G.edges[e]["cost"] = sdf[int(midpoint[1]), int(midpoint[0])] ** 0.5 + \
                                     sdf[int(start[1]), int(start[0])] ** 0.5 +  \
                                     sdf[int(end[1]), int(end[0])] ** 0.5
            except:
                continue

        start_node = np.argmin(np.linalg.norm(np.array([G.nodes[i]["pos"] for i in G.nodes]) - t[0], axis=1))
        end_node = np.argmin(np.linalg.norm(np.array([G.nodes[i]["pos"] for i in G.nodes]) - t[-1], axis=1))

        path = nx.dijkstra_path(G, start_node, end_node, weight="cost")

        for i in range(len(path) - 1):
            G.edges[path[i], path[i+1]]["log_odds_dijkstra"] += 1
        for i in range(len(path)):
            G.nodes[path[i]]["log_odds_dijkstra"] += 1

    angles_viz = visualize_angles(np.cos(global_angle),
                                  np.sin(global_angle),
                                  mask=global_mask[:, :, 0])

    return G, global_sdf, global_angle, angles_viz


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



def process_chunk(source, roi_xxyy_list, export_final, trajectories_, lanes_, sat_image_, out_path_root, centerline_image_=None):
    #
    # for l in lanes_[::5]:
    #     plt.plot(l[:, 0], l[:, 1], linewidth=0.5)
    # for t in trajectories_[::5]:
    #     plt.plot(t[:, 0], t[:, 1], linewidth=2.0)
    # plt.show()

    if "tracklets" in source:
        annotations_centers = [np.mean(t, axis=0) for t in trajectories_]
        annot_ = trajectories_
    elif "lanes" in source:
        annotations_centers = [np.mean(t, axis=0) for t in lanes_]
        annot_ = lanes_
    else:
        raise ValueError("Invalid source")

    for roi_xxyy in tqdm(roi_xxyy_list):

        if centerline_image_ is not None:
            if np.sum(centerline_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3]]) == 0:
                # no centerlines (and therefore no tracklets to be evaluated) in this roi
                continue

        if roi_xxyy[0] < 16000:
            out_path = os.path.join(out_path_root, "val")
        else:
            out_path = os.path.join(out_path_root, "train")

        sample_id = "{}-{}-{}".format(city_name, roi_xxyy[0], roi_xxyy[2])

        if os.path.exists(os.path.join(out_path, "{}.pth".format(sample_id))):
            continue

        sat_image = sat_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3], :].copy()

        # Filter non-square sat_images:
        if sat_image.shape[0] != sat_image.shape[1]:
            continue

        annot_candidates = []
        for i in range(len(annot_)):
            if np.linalg.norm(annotations_centers[i] - [roi_xxyy[2], roi_xxyy[0]]) < 500:
                annot_candidates.append(annot_[i])

        annots = []
        for annot in annot_candidates:

            annot = annot - np.array([roi_xxyy[2], roi_xxyy[0]])

            # # filter trajectories according to current roi_xxyy
            is_in_roi = np.logical_and(annot[:, 0] > 0, annot[:, 0] < sat_image.shape[1])
            if not np.any(is_in_roi):
                continue
            is_in_roi = np.logical_and(is_in_roi, annot[:, 1] > 0)
            if not np.any(is_in_roi):
                continue

            is_in_roi = np.logical_and(is_in_roi, annot[:, 1] < sat_image.shape[0])
            if not np.any(is_in_roi):
                continue


            annot = annot[is_in_roi]

            # resample trajectory to have equally distant points
            annot = resample_trajectory(annot, dist=5)

            # filter based on number of points
            if "tracklets" in source:
                if len(annot) < 15:
                    continue

            # and on physical length of trajectory
            if "tracklets" in source:
                total_length = np.sum(np.linalg.norm(annot[1:] - annot[:-1], axis=1))
                if total_length < 50:
                    continue

            annots.append(annot)

        if len(annots) < 1:
            continue

        min_distance_from_center = min([np.min(np.linalg.norm(trajectory - np.array(sat_image.shape[:2]) / 2, axis=1)) for trajectory in annots])
        if min_distance_from_center > 30:
            continue

        def get_sdf(t):
            sdf = np.zeros(sat_image.shape[0:2], dtype=np.float32)
            for i in range(len(t) - 1):
                x1 = int(t[i][0])
                y1 = int(t[i][1])
                x2 = int(t[i + 1][0])
                y2 = int(t[i + 1][1])
                cv2.line(sdf, (x1, y1), (x2, y2), (1, 1, 1), thickness=5)
            f = 15  # distance function scale
            sdf = skfmm.distance(1 - sdf)
            sdf[sdf > f] = f
            sdf = sdf / f
            sdf = 1 - sdf

            return sdf


        if "tracklets" in source:
            # Filter out redundant trajectories:
            filtered_annots = []
            global_sdf = np.zeros(sat_image.shape[0:2], dtype=np.float32)

            for t in annots:
                if len(filtered_annots) == 0:
                    filtered_annots.append(t)
                t_sdf = get_sdf(t)

                # get overlap between t_sdf and global_sdf
                overlap = np.sum(np.logical_and(t_sdf > 0.1, global_sdf > 0.1)) + 1
                t_sdf_sum = np.sum(t_sdf > 0.1)

                if t_sdf_sum / overlap > 2:
                    filtered_annots.append(t)
                    global_sdf += t_sdf
                else:
                    continue

        # Whether to export sparse or dense trajectory annotations
        if source == "tracklets_sparse":
            trajectories = filtered_annots
            output_name = "sparse"
        elif source == "tracklets_dense":
            output_name = "dense"
            trajectories = annots
        elif source == "lanes":
            output_name = "lanes"
            trajectories = annots
        else:
            raise ValueError("Invalid source")


        G = initialize_graph(roi_xxyy, r_min=r_min)

        for trajectory in trajectories:

            # check length of trajectory
            if "tracklets" in source:
                if np.linalg.norm(trajectory[0] - trajectory[-1]) < 50:
                    print("skipping trajectory with length < 50")
                    continue

            # Now we update the angular gridmap
            for i in range(len(trajectory) - 1):
                pos = trajectory[i]
                next_pos = trajectory[i + 1]
                angle = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
                G = bayes_update_graph(G, angle, x=pos[0], y=pos[1], p=0.9, r_min=r_min)

        # perform angle kernel density estimation and peak detection
        G = angle_kde(G)

        # assign edge probabilities according to dijstra-approximated trajectories
        G, sdf, angles, angles_viz = dijkstra_trajectories(G, trajectories, imsize=sat_image.shape[:2])

        log_odds_e = np.array([G.edges[e]["log_odds_dijkstra"] for e in G.edges])
        log_odds_n = np.array([G.nodes[n]["log_odds_dijkstra"] for n in G.nodes])

        node_probabilities = np.exp(log_odds_n) / (1 + np.exp(log_odds_n))
        edge_probabilities = np.exp(log_odds_e) / (1 + np.exp(log_odds_e))

        if np.count_nonzero(edge_probabilities[edge_probabilities > 0.5]) < 10:
            print("Only {} edges with probability > 0.5".format(np.count_nonzero(edge_probabilities[edge_probabilities > 0.5])))
            continue


        # rescale probabilities
        node_probabilities = (node_probabilities - np.min(node_probabilities)) / (np.max(node_probabilities) - np.min(node_probabilities))
        edge_probabilities = (edge_probabilities - np.min(edge_probabilities)) / (np.max(edge_probabilities) - np.min(edge_probabilities))

        node_probabilities = (node_probabilities > 0.1).astype(np.float32)
        edge_probabilities = (edge_probabilities > 0.1).astype(np.float32)

        # assign probabilities to edges
        for i, e in enumerate(G.edges):
            G.edges[e]["p"] = edge_probabilities[i]
        # assign probabilities to nodes
        for i, n in enumerate(G.nodes):
            G.nodes[n]["p"] = node_probabilities[i]

        print("Saving to {}/{}-rgb.png".format(out_path, sample_id))

        Image.fromarray(sat_image).save("{}/{}-rgb.png".format(out_path, sample_id))
        Image.fromarray(angles_viz).save("{}/{}-angles-tracklets-{}.png".format(out_path, sample_id, output_name))
        Image.fromarray(sdf * 255.).convert("L").save("{}/{}-sdf-tracklets-{}.png".format(out_path, sample_id, output_name))

        if export_final:

            sdf_regressor = cv2.imread(
                os.path.join(out_path.replace("-post", "-pre"), "{}-sdf-reg.png".format(sample_id)))[:,:,0] / 255.
            angles_regressor = cv2.imread(
                os.path.join(out_path.replace("-post", "-pre"), "{}-angles-reg.png".format(sample_id)))

            # Filter graph according to predicted sdf
            G_ = G.copy()
            for n in G_.nodes:
                pos = G.nodes[n]["pos"]
                if sdf_regressor[int(pos[1]), int(pos[0])] < 0.3:
                    G.remove_node(n)

            # remap node ids and edges
            node_ids = list(G.nodes)
            node_id_map = {node_ids[i]: i for i in range(len(node_ids))}
            G = nx.relabel_nodes(G, node_id_map)

            node_probabilities = np.array([G.nodes[n]["p"] for n in G.nodes])
            edge_probabilities = np.array([G.edges[e]["p"] for e in G.edges])

            if np.any(np.isnan(node_probabilities)) or np.any(np.isnan(edge_probabilities)):
                continue

            print("Processing {}...".format(sample_id))

            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(vmin=0.0, vmax=1.0)
            node_colors = cmap(norm(node_probabilities))

            fig, ax = plt.subplots(figsize=(10, 10))
            plt.tight_layout()
            ax.set_aspect('equal')
            ax.imshow(sat_image)

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
            edge_colors[:, -1] = edge_probabilities

            nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
                             edge_color=edge_colors,
                             node_color=node_colors,
                             with_labels=False,
                             node_size=10,
                             arrowsize=3.0,
                             width=1,
                             )

            plt.savefig("{}/{}-{}.png".format(out_path, sample_id, output_name), dpi=400)

            # preprocess sample into pth file
            preprocess_sample(G,
                              sat_image_=sat_image_,
                              sdf=sdf,
                              sdf_regressor=sdf_regressor,
                              angles=angles,
                              angles_viz=angles_viz,
                              angles_regressor=angles_regressor,
                              roi_xxyy=roi_xxyy,
                              sample_id=sample_id,
                              out_path=out_path,
                              output_name=output_name)


if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_final", action="store_true")
    parser.add_argument("--city_name", type=str, default="data")
    parser.add_argument("--out_path_root", type=str, default="data")
    parser.add_argument("--source", type=str, default="tracklets_sparse", choices=["tracklets_sparse", "tracklets_dense", "lanes"])
    args = parser.parse_args()

    out_path_root = args.out_path_root
    city_name = args.city_name.capitalize()
    if city_name == "Paloalto":
        city_name = "PaloAlto"
    export_final = args.export_final

    # parameters
    num_cpus = 4
    r_min = 10  # minimum radius of the circle for poisson disc sampling

    os.makedirs(os.path.join(out_path_root), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'val'), exist_ok=True)

    '''find /data/argoverse2/motion-forecasting -type f -wholename '/data/argoverse2/motion-forecasting/val/*/*.parquet' > scenario_files.txt '''

    sat_image_ = np.asarray(Image.open("/data/lanegraph/woven-data/{}.png".format(city_name))).astype(np.uint8)
    centerline_image_ = np.asarray(Image.open("/data/lanegraph/woven-data/{}_centerlines.png".format(city_name)))

    print("Satellite resolution: {}x{}".format(sat_image_.shape[1], sat_image_.shape[0]))

    # generate roi_xxyy list over full satellite image in sliding window fashion
    roi_xxyy_list = []
    for i in range(0, sat_image_.shape[1]-256, 100):
        for j in range(0, sat_image_.shape[0]-256, 100):
            roi_xxyy_list.append(np.array([j, j + 256, i, i + 256]))

    random.shuffle(roi_xxyy_list)
    roi_xxyy_list = roi_xxyy_list[0:10000]
    print("Careful! Using reduced list of rois ({})".format(len(roi_xxyy_list)))

    if args.source == "tracklets_sparse":
        print("Exporting SPARSE tracklet annotations!")
    elif args.source == "tracklets_dense":
        print("Exporting DENSE tracklet annotations!")
    elif args.source == "lanes":
        print("Exporting LANE annotations!")
    else:
        raise ValueError("Invalid source!")

    all_scenario_files = np.loadtxt("/home/zuern/self-supervised-graph/aggregation/scenario_files.txt", dtype=str).tolist()

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

            if city_name == "PaloAlto":
                city_name_log = "palo-alto"
            elif city_name == "Washington":
                city_name_log = "washington-dc"
            elif city_name == "Detroit":
                city_name_log = "dearborn"
            else:
                city_name_log = city_name

            if scenario.city_name != city_name_log.lower():
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

    print("Number of trajectories: {}".format(len(trajectories_)))

    # single core
    if num_cpus <= 1:
        process_chunk(args.source,
                      roi_xxyy_list,
                      export_final,
                      trajectories_,
                      lanes_,
                      sat_image_,
                      out_path_root,
                      centerline_image_,
                      )
    else:

        # multi core
        def chunkify(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        num_samples = len(roi_xxyy_list)
        num_chunks = int(np.ceil(num_samples / num_cpus))
        roi_chunks = list(chunkify(roi_xxyy_list, n=num_chunks))

        arguments = zip(repeat(args.source),
                        roi_chunks,
                        repeat(export_final),
                        repeat(trajectories_),
                        repeat(lanes_),
                        repeat(sat_image_),
                        repeat(out_path_root),
                        repeat(centerline_image_),
                        )

        Pool(num_cpus).starmap(process_chunk, arguments)



