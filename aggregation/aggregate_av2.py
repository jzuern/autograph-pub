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
from itertools import repeat
from ray.util.multiprocessing import Pool
import cv2
import skfmm
import argparse
from sklearn.cluster import DBSCAN
import skimage
from glob import glob
import logging
from av2.datasets.motion_forecasting import scenario_serialization
import pickle
import time

from lanegnn.utils import poisson_disk_sampling, get_random_edges
from data.av2.settings import get_transform_params
from lanegnn.utils import visualize_angles
from aggregation.utils import get_scenario_centerlines, assign_graph_traversals, resample_trajectory, Tracklet, \
    filter_tracklet, preprocess_sample, merge_successor_trajectories, bayes_update_graph, angle_kde, bm0, \
    initialize_graph, dijkstra_trajectories

# random shuffle seed
#random.seed(0)
np.random.seed(seed=int(time.time()))



def get_succ_graph(q, succ_traj, sat_image_viz, r_min=10):

    endpoints = []

    for t in succ_traj:
        if np.any(np.isclose(t[-1], np.array([255, 255]), atol=10.0)) or \
                np.any(np.isclose(t[-1], np.array([0, 0]), atol=10.0))  :
            coords = (int(t[-1, 0]), int(t[-1, 1]))
            cv2.circle(sat_image_viz, coords, 5, (255, 255, 255), -1)
            endpoints.append(coords)

    # sample halting points everywhere
    points = poisson_disk_sampling(r_min=r_min,
                                   width=256,
                                   height=256)
    edges = get_random_edges(points,
                             min_point_dist=r_min,
                             max_point_dist=2*r_min)

    points = [np.array([p[0], p[1]]) for p in points]

    G = nx.DiGraph()
    for i in range(len(points)):
        G.add_node(i, pos=points[i], log_odds_dijkstra=0.0, p=0)
    for i in range(len(edges)):
        G.add_edge(edges[i][0], edges[i][1], log_odds_dijkstra=0.0, p=0)

    # Assign traversal cost according to tracklets in succ_traj
    G, mask, mask_thin, sdf, angles = assign_graph_traversals(G, succ_traj, imsize=sat_image_viz.shape[0:2])
    mask = (mask / 255.).astype(np.float32)

    # Cluster endpoints
    endpoints = np.array(endpoints)

    try:
        clustering = DBSCAN(eps=15, min_samples=2).fit(endpoints)
    except:
        logging.error("DBSCAN endpoint clustering failed. Skipping")
        return None, None, None, None, None, None

    endpoints_centroids = []
    for c in np.unique(clustering.labels_):
        endpoints_centroids.append(np.mean(endpoints[clustering.labels_ == c], axis=0))
    endpoints_centroids = np.array(endpoints_centroids)

    # Get planning from query point to end points
    G_start = np.argmin(np.linalg.norm(np.array([G.nodes[i]["pos"] for i in G.nodes]) - q, axis=1))
    G_ends = [np.argmin(np.linalg.norm(np.array([G.nodes[i]["pos"] for i in G.nodes]) - endnode, axis=1)) for endnode in endpoints_centroids]

    for G_end in G_ends:
        path = nx.dijkstra_path(G, G_start, G_end, weight="cost")
        for i in range(len(path) - 1):
            G.edges[path[i], path[i + 1]]["p"] = 1
        for i in range(len(path)):
            G.nodes[path[i]]["p"] = 1

    # plot start node and end nodes
    pos_start = G.nodes[G_start]["pos"]
    pos_ends = [G.nodes[G_end]["pos"] for G_end in G_ends]

    cv2.circle(sat_image_viz, (int(pos_start[0]), int(pos_start[1])), 4, (0, 255, 0), -1)
    [cv2.circle(sat_image_viz, (int(p[0]), int(p[1])), 4, (0, 0, 0), -1) for p in pos_ends]

    angles_viz = visualize_angles(np.cos(angles),
                                  np.sin(angles),
                                  mask=mask[:, :, 0])

    try:
        sdf_thin = skfmm.distance(1 - mask_thin) - skfmm.distance(mask_thin)
        sdf_thin = sdf_thin - sdf_thin.min() + 1
    except:
        logging.error("SDF thinning failed. Skipping")
        return None, None, None, None, None, None

    kernel = np.zeros((20, 20), np.uint8)
    kernel = cv2.circle(kernel, (10, 10), 10, 1, -1)

    path_imgs = []

    if len(pos_ends) < 1:
        logging.error("Too few endpoints found ({}). Skipping".format(len(pos_ends)))
        return None, None, None, None, None, None

    for pos_end in pos_ends:

        path, cost = skimage.graph.route_through_array(sdf_thin.T,
                                                       start=tuple(pos_start.astype(np.int32)),
                                                       end=tuple(pos_end.astype(np.int32)),
                                                       fully_connected=True)
        path = np.stack(path, axis=-1)
        path_img = np.zeros_like(sdf, dtype=np.int32)
        path_img[path[1], path[0]] = 1
        path_imgs.append(cv2.dilate(path_img.astype(np.uint8), kernel, iterations=1))

    mask_succ_sparse = np.max(np.array(path_imgs), axis=0)


    return G, sat_image_viz, mask, angles, angles_viz, mask_succ_sparse




def process_chunk_successors(source, roi_xxyy_list, export_final, trajectories_, trajectories_ped_, lanes_, sat_image_, out_path_root, centerline_image_=None, city_name=None):

    if "tracklets" in source:
        annotations_centers = [np.mean(t, axis=0) for t in trajectories_]
        annotations_ped_centers = [np.mean(t, axis=0) for t in trajectories_ped_]
        annot_ = trajectories_
        annot_ped_ = trajectories_ped_
    elif "lanes" in source:
        annotations_centers = [np.mean(t, axis=0) for t in lanes_]
        annot_ = lanes_
    else:
        raise ValueError("Invalid annotation source")

    for roi_num, roi_xxyy in tqdm(enumerate(roi_xxyy_list), total=len(roi_xxyy_list)):

        if centerline_image_ is not None:
            if np.sum(centerline_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3]]) == 0:
                continue

        if np.random.rand() < 0.1:
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
        annot_ped_candidates = []
        for i in range(len(annot_)):
            if np.linalg.norm(annotations_centers[i] - [roi_xxyy[2], roi_xxyy[0]]) < 500:
                annot_candidates.append(annot_[i])
        for i in range(len(annot_ped_)):
            if np.linalg.norm(annotations_ped_centers[i] - [roi_xxyy[2], roi_xxyy[0]]) < 500:
                annot_ped_candidates.append(annot_ped_[i])

        annots = []
        for annot in annot_candidates:
            annot = annot - np.array([roi_xxyy[2], roi_xxyy[0]])
            # filter trajectories according to current roi_xxyy
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

            annots.append(annot)

        annots_ped = []
        for annot in annot_ped_candidates:
            annot = annot - np.array([roi_xxyy[2], roi_xxyy[0]])

            # filter trajectories according to current roi_xxyy
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

            annots_ped.append(annot)


        # filter out too short annots
        annots = [a for a in annots if len(a) > 5]
        annots_ped = [a for a in annots_ped if len(a) > 5]

        if len(annots) < 1:
            continue

        min_distance_from_center = min([np.min(np.linalg.norm(trajectory - np.array(sat_image.shape[:2]) / 2, axis=1)) for trajectory in annots])
        if min_distance_from_center > 30:
            continue

        query_points = np.random.randint(0, sat_image.shape[0], size=(200, 2))

        # Whether to export sparse or dense trajectory annotations
        if source == "tracklets_sparse":
            trajectories = annots  # not filtered
            output_name = "sparse"
        elif source == "tracklets_dense":
            output_name = "dense"
            trajectories = annots
        elif source == "lanes":
            output_name = "lanes"
            trajectories = annots
        else:
            raise ValueError("Invalid source")

        tracklets_im_list = []

        for i_query, q in enumerate(query_points):

            succ_traj, mask_total, sat_image_viz = merge_successor_trajectories(q, trajectories, annots_ped, sat_image)

            if len(succ_traj) < 3:
                logging.info("Too few successor trajectories")
                continue

            G, sat_image_viz, sdf, angles, angles_viz, mask_succ_sparse = get_succ_graph(q, succ_traj,  sat_image_viz, r_min=r_min)
            if G is None:
                logging.info("Successor graph is None. Skipping")
                continue

            # Minimum of X percent of pixels must be covered by the trajectory
            if np.sum(sdf > 0.01) < 0.02 * np.prod(sdf.shape):
                logging.info("Not enough pixels covered with successor graph. Skipping")
                continue

            # Must be sufficiently dissimilar from any previous sample
            max_iou = max([bm0(mask_succ_sparse, m) for m in tracklets_im_list]) if len(tracklets_im_list) > 0 else 0
            if max_iou > 0.7:
                logging.info("Sample too similar to previous samples. Skipping")
                continue
            tracklets_im_list.append(mask_succ_sparse)


            # Build 3 channel mask
            mask_all = np.zeros(sat_image.shape, dtype=np.uint8)
            mask_all[:, :, 0] = (mask_succ_sparse * 255.).astype(np.uint8)
            mask_all[:, :, 1] = mask_total[:, :, 1]
            mask_all[:, :, 2] = mask_total[:, :, 2]


            pos_encoding = np.zeros(sat_image.shape, dtype=np.float32)
            x, y = np.meshgrid(np.arange(sat_image.shape[1]), np.arange(sat_image.shape[0]))
            pos_encoding[q[1], q[0], 0] = 1
            pos_encoding[..., 1] = np.abs((x - q[0])) / sat_image.shape[1]
            pos_encoding[..., 2] = np.abs((y - q[1])) / sat_image.shape[0]
            pos_encoding = (pos_encoding * 255).astype(np.uint8)

            print("-------------\nSaving to {}-{}-{}-{}".format(out_path, sample_id, i_query, output_name))

            Image.fromarray(pos_encoding).save("{}/{}-{}-pos-encoding-{}.png".format(out_path, sample_id, i_query, output_name))
            Image.fromarray(sat_image).save("{}/{}-{}-rgb.png".format(out_path, sample_id, i_query))
            Image.fromarray(sat_image_viz).save("{}/{}-{}-rgb-viz.png".format(out_path, sample_id, i_query))
            #Image.fromarray(angles_viz).save("{}/{}-{}-angles-tracklets-{}.png".format(out_path, sample_id, i_query, output_name))
            #Image.fromarray((sdf * 255.).astype(np.uint8)).convert("L").save("{}/{}-{}-sdf-tracklets-{}.png".format(out_path, sample_id, i_query, output_name))
            #Image.fromarray((mask_succ_sparse * 255.).astype(np.uint8)).convert("L").save("{}/{}-{}-sdf-tracklets-{}-sparse.png".format(out_path, sample_id, i_query, output_name))
            Image.fromarray((mask_all).astype(np.uint8)).save("{}/{}-{}-masks-{}.png".format(out_path, sample_id, i_query, output_name))


            if export_final:
                sdf_regressor = cv2.imread(os.path.join(out_path.replace("-post", "-pre"), "{}-sdf-reg.png".format(sample_id)))[:,:,0] / 255.
                angles_regressor = cv2.imread(os.path.join(out_path.replace("-post", "-pre"), "{}-angles-reg.png".format(sample_id)))

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

                plt.savefig("{}/{}-{}-{}.png".format(out_path, sample_id, i_query, output_name), dpi=400)
                plt.close(fig)

                #fig.close()

                # preprocess sample into pth file
                preprocess_sample(G,
                                  sat_image_=sat_image_,
                                  pos_encoding=pos_encoding,
                                  sdf=sdf,
                                  sdf_regressor=sdf_regressor,
                                  angles=angles,
                                  angles_viz=angles_viz,
                                  angles_regressor=angles_regressor,
                                  roi_xxyy=roi_xxyy,
                                  sample_id=sample_id,
                                  out_path=out_path,
                                  i_query=i_query,
                                  output_name=output_name)

    #np.save("data/roi_usable_{}.npy".format(city_name), roi_usable)


def process_chunk(source, roi_xxyy_list, export_final, trajectories_, lanes_, sat_image_, out_path_root, centerline_image_=None, city_name=None):


    if "tracklets" in source:
        annotations_centers = [np.mean(t, axis=0) for t in trajectories_]
        annot_ = trajectories_
    elif "lanes" in source:
        annotations_centers = [np.mean(t, axis=0) for t in lanes_]
        annot_ = lanes_
    else:
        raise ValueError("Invalid source")

    for roi_num, roi_xxyy in tqdm(enumerate(roi_xxyy_list), total=len(roi_xxyy_list)):

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

            # filter trajectories according to current roi_xxyy
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
            #print("Only {} edges with probability > 0.5".format(np.count_nonzero(edge_probabilities[edge_probabilities > 0.5])))
            continue

        #roi_usable[roi_num] = True
        #continue

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

        print("-------------")
        print("Saving to {}-{}".format(sample_id, output_name))

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

    np.save("data/roi_usable_{}.npy".format(city_name), roi_usable)


def random_cropping(sat_image, tracklet_image, drivable_gt):

    crop_size = 256


    while True:

        # random center position
        center_x = np.random.randint(crop_size // 2, sat_image.shape[1] - crop_size // 2)
        center_y = np.random.randint(crop_size // 2, sat_image.shape[0] - crop_size // 2)
        angle = np.random.uniform(0, 2 * np.pi)

        sat_image_precrop = sat_image[center_y - crop_size:center_y + crop_size,
                                      center_x - crop_size:center_x + crop_size, :].copy()
        tracklet_image_precrop = tracklet_image[center_y - crop_size:center_y + crop_size,
                                                center_x - crop_size:center_x + crop_size, :].copy()
        drivable_gt_precrop = drivable_gt[center_y - crop_size:center_y + crop_size,
                                          center_x - crop_size:center_x + crop_size].copy()

        # check if tracklet image is empty in central region
        half = sat_image_precrop.shape[1] // 2

        if np.sum(tracklet_image_precrop[half - 64:half + 64, half - 64:half + 64, 2]) == 0:
            continue

        # get crop at center and with angle
        M = cv2.getRotationMatrix2D((256, 256), angle * 180 / np.pi, 1)
        sat_image_crop = cv2.warpAffine(sat_image_precrop, M, (512, 512))[256 - 128:256 + 128,
                                                                          256 - 128:256 + 128]
        tracklet_image_crop = cv2.warpAffine(tracklet_image_precrop, M, (512, 512), cv2.INTER_NEAREST)[256 - 128:256 + 128,
                                                                                                       256 - 128:256 + 128]
        drivable_gt_crop = cv2.warpAffine(drivable_gt_precrop, M, (512, 512), cv2.INTER_NEAREST)[256 - 128:256 + 128,
                                                                                                 256 - 128:256 + 128]

        return sat_image_crop, tracklet_image_crop, drivable_gt_crop, [center_x, center_y, angle]




def process_chunk_final(city_name, source, trajectories_, trajectories_ped_, lanes_,
                        sat_image_, tracklets_image, drivable_gt, out_path_root,
                        max_num_samples=100):

    if "tracklets" in source:
        annotations_centers = [np.mean(t, axis=0) for t in trajectories_]
        annotations_ped_centers = [np.mean(t, axis=0) for t in trajectories_ped_]
        annot_ = trajectories_
        annot_ped_ = trajectories_ped_
    elif "lanes" in source:
        annotations_centers = [np.mean(t, axis=0) for t in lanes_]
        annot_ = lanes_
    else:
        raise ValueError("Invalid annotation source")

    print("here")

    sample_num = 0
    while sample_num < max_num_samples:

        print(sample_num)

        # this is not guaranteed to give valid crops
        sat_image_crop, tracklet_image_crop, drivable_gt_crop, crop_pose = \
            random_cropping(sat_image_, tracklets_image, drivable_gt)

        angle = crop_pose[2]
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])

        print(crop_pose)

        if np.random.rand() < 0.1:
            out_path = os.path.join(out_path_root, "val")
        else:
            out_path = os.path.join(out_path_root, "train")

        sample_id = "{}-{}-{}".format(city_name, crop_pose[0], crop_pose[1])

        if os.path.exists(os.path.join(out_path, "{}.pth".format(sample_id))):
            continue

        annot_candidates = []
        annot_ped_candidates = []
        for i in range(len(annot_)):
            if np.linalg.norm(annotations_centers[i] - [crop_pose[0], crop_pose[1]]) < 500:
                annot_candidates.append(annot_[i])
        for i in range(len(annot_ped_)):
            if np.linalg.norm(annotations_ped_centers[i] - [crop_pose[0], crop_pose[1]]) < 500:
                annot_ped_candidates.append(annot_ped_[i])

        # Debug
        # plt.imshow(sat_image_crop)
        # plt.imshow(tracklet_image_crop, alpha=0.5)
        # for annot in annot_candidates:
        #     annot_transformed = np.dot(annot - [crop_pose[0], crop_pose[1]], R) + [128, 128]
        #     plt.plot(annot_transformed[:, 0], annot_transformed[:, 1], c='g')
        # plt.show()


        annots = []
        for annot in annot_candidates:
            # transform to crop coordinates
            annot = np.array(annot)
            annot = np.dot(annot - [crop_pose[0], crop_pose[1]], R)  + [128, 128]

            is_in_roi = np.logical_and(annot[:, 0] > 0, annot[:, 0] < sat_image_crop.shape[1])
            if not np.any(is_in_roi):
                continue
            is_in_roi = np.logical_and(is_in_roi, annot[:, 1] > 0)
            if not np.any(is_in_roi):
                continue
            is_in_roi = np.logical_and(is_in_roi, annot[:, 1] < sat_image_crop.shape[0])
            if not np.any(is_in_roi):
                continue
            annot = annot[is_in_roi]

            # resample trajectory to have equally distant points
            annot = resample_trajectory(annot, dist=5)

            annots.append(annot)

        annots_ped = []
        for annot in annot_ped_candidates:
            annot = np.array(annot)
            annot = np.dot(annot - [crop_pose[0], crop_pose[1]], R) + [128, 128]

            # filter trajectories according to current roi_xxyy
            is_in_roi = np.logical_and(annot[:, 0] > 0, annot[:, 0] < sat_image_crop.shape[1])
            if not np.any(is_in_roi):
                continue
            is_in_roi = np.logical_and(is_in_roi, annot[:, 1] > 0)
            if not np.any(is_in_roi):
                continue
            is_in_roi = np.logical_and(is_in_roi, annot[:, 1] < sat_image_crop.shape[0])
            if not np.any(is_in_roi):
                continue
            annot = annot[is_in_roi]

            # resample trajectory to have equally distant points
            annot = resample_trajectory(annot, dist=5)

            annots_ped.append(annot)


        # filter out too short annots
        annots = [a for a in annots if len(a) > 5]
        annots_ped = [a for a in annots_ped if len(a) > 5]

        if len(annots) < 1:
            continue

        # switch axis order to x,y
        #annots = [a[:, [1, 0]] for a in annots]
        #annots_ped = [a[:, [1, 0]] for a in annots_ped]

        # Whether to export sparse or dense trajectory annotations
        if source == "tracklets_sparse":
            trajectories = annots  # not filtered
            output_name = "sparse"
        elif source == "tracklets_dense":
            output_name = "dense"
            trajectories = annots
        elif source == "lanes":
            output_name = "lanes"
            trajectories = annots
        else:
            raise ValueError("Invalid source")


        # get random query points on image
        #query_points = np.random.randint(0, sat_image_crop.shape[0], size=(200, 2))
        # get random query points at nonzeros of tracklet image

        query_points = np.argwhere(tracklet_image_crop[:, :, 2] > 0)
        np.random.shuffle(query_points)
        query_points = query_points[:50]

        # Debug plot
        # plt.imshow(sat_image_crop)
        # plt.imshow(tracklet_image_crop, alpha=0.5)
        # plt.scatter(query_points[:, 1], query_points[:, 0], c='r')
        # for t in trajectories:
        #     plt.plot(t[:, 0], t[:, 1], c='g')
        # plt.show()

        tracklets_im_list = []

        for i_query, q in enumerate(query_points):

            succ_traj, mask_total, sat_image_viz = merge_successor_trajectories(q, trajectories, annots_ped, sat_image_crop)

            if len(succ_traj) < 3:
                logging.info("Too few successor trajectories")
                continue

            G, sat_image_viz, sdf, angles, angles_viz, mask_succ_sparse = get_succ_graph(q, succ_traj,  sat_image_viz, r_min=r_min)
            if G is None:
                logging.info("Successor graph is None. Skipping")
                continue

            # Minimum of X percent of pixels must be covered by the trajectory
            sdf = sdf[:, :, 0]
            if np.sum(sdf > 0.01) < 0.02 * np.prod(sdf.shape):
                logging.info("Not enough pixels covered with successor graph. Skipping")
                continue

            # Must be sufficiently dissimilar from any previous sample
            max_iou = max([bm0(mask_succ_sparse, m) for m in tracklets_im_list]) if len(tracklets_im_list) > 0 else 0
            if max_iou > 0.7:
                logging.info("Sample too similar to previous samples. Skipping")
                continue
            tracklets_im_list.append(mask_succ_sparse)

            # Build 3 channel mask
            mask_all = np.zeros(sat_image_crop.shape, dtype=np.uint8)
            mask_all[:, :, 0] = (mask_succ_sparse * 255.).astype(np.uint8)
            mask_all[:, :, 1] = mask_total[:, :, 1]
            mask_all[:, :, 2] = mask_total[:, :, 2]

            pos_encoding = np.zeros(sat_image_crop.shape, dtype=np.float32)
            x, y = np.meshgrid(np.arange(sat_image_crop.shape[1]), np.arange(sat_image_crop.shape[0]))
            pos_encoding[q[1], q[0], 0] = 1
            pos_encoding[..., 1] = np.abs((x - q[0])) / sat_image_crop.shape[1]
            pos_encoding[..., 2] = np.abs((y - q[1])) / sat_image_crop.shape[0]
            pos_encoding = (pos_encoding * 255).astype(np.uint8)

            print("-------------\nSaving to {}/{}-{}".format(out_path, sample_id, i_query))

            Image.fromarray(pos_encoding).save("{}/{}-{}-pos-encoding.png".format(out_path, sample_id, i_query))
            Image.fromarray(sat_image_crop).save("{}/{}-{}-rgb.png".format(out_path, sample_id, i_query))
            Image.fromarray(sat_image_viz).save("{}/{}-{}-rgb-viz.png".format(out_path, sample_id, i_query))
            Image.fromarray((mask_all).astype(np.uint8)).save("{}/{}-{}-masks.png".format(out_path, sample_id, i_query))
            Image.fromarray((drivable_gt_crop).astype(np.uint8)).save("{}/{}-{}-drivable-gt.png".format(out_path, sample_id, i_query))

            sample_num += 1
            print("Generated sample {} / {}".format(sample_num, max_num_samples))



if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_final", action="store_true")
    parser.add_argument("--city_name", type=str, default="data")
    parser.add_argument("--sat_image_root", type=str, default="/data/lanegraph/woven-data/")
    parser.add_argument("--out_path_root", type=str, default="data")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--source", type=str, default="tracklets_sparse", choices=["tracklets_sparse", "tracklets_dense", "lanes"])
    parser.add_argument("--max_num_samples", type=int, default=100, help="Number of samples to generate per city")
    args = parser.parse_args()

    out_path_root = args.out_path_root
    city_name = args.city_name.capitalize()
    if city_name == "Paloalto":
        city_name = "PaloAlto"

    city_name_dict = {
        "PIT": "Pittsburgh",
        "MIA": "Miami",
        "ATX": "Austin",
        "WDC": "Washington",
        "PAO": "PaloAlto",
        "DTW": "Detroit",
    }


    export_final = args.export_final

    # parameters
    num_cpus = args.num_cpus
    r_min = 7  # minimum radius of the circle for poisson disc sampling

    os.makedirs(os.path.join(out_path_root), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'val'), exist_ok=True)

    sat_image_ = np.asarray(Image.open(os.path.join(args.sat_image_root, "{}.png".format(city_name)))).astype(np.uint8)
    # centerline_image_ = np.asarray(Image.open(os.path.join(args.sat_image_root, "{}_centerlines.png".format(city_name))))

    print("Satellite resolution: {}x{}".format(sat_image_.shape[1], sat_image_.shape[0]))

    if args.source == "tracklets_sparse":
        print("Exporting SPARSE tracklet annotations!")
    elif args.source == "tracklets_dense":
        print("Exporting DENSE tracklet annotations!")
    elif args.source == "lanes":
        print("Exporting LANE annotations!")
    else:
        raise ValueError("Invalid source!")

    all_tracking_files = glob('/data/argoverse2-full/*_tracking.pickle')

    [transform_R, transform_c, transform_t] = get_transform_params(city_name.lower())

    if not os.path.exists("data/lanes_{}.npy".format(city_name)) or not os.path.exists("data/trajectories_gt_{}.npy".format(city_name)):
        print("Generating trajectories and gt-lanes")
        trajectories_ = []
        lanes_ = []

        for scenario_path in tqdm(all_tracking_files):
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
                    tmp = transform_t + transform_c * transform_R @ bb
                    lane[i] = tmp[0:2]

            for track in scenario.tracks:
                # Get actor trajectory and heading history
                actor_trajectory = np.array([list(object_state.position) for object_state in track.object_states])

                if track.object_type != "vehicle":
                    continue

                # Coordinate transformation
                for i in range(len(actor_trajectory)):
                    bb = np.array([actor_trajectory[i, 0], actor_trajectory[i, 1], 0])
                    tmp = transform_t + transform_c * transform_R @ bb
                    actor_trajectory[i] = tmp[0:2]
                trajectories_.append(actor_trajectory)

            for lane in scenario_lanes:
                lanes_.append(lane)

        trajectories_ = np.array(trajectories_)
        lanes_ = np.array(lanes_)

        # save trajectories
        np.save("data/trajectories_gt_{}.npy".format(city_name), trajectories_)
        np.save("data/lanes_{}.npy".format(city_name), lanes_)
    else:
        trajectories_gt_ = np.load("data/trajectories_gt_{}.npy".format(city_name), allow_pickle=True)  # GT
        lanes_ = np.load("data/lanes_{}.npy".format(city_name), allow_pickle=True)

    if not os.path.exists("data/trajectories_pred_{}.npy".format(city_name)):
        trajectories_pred_ = []
        trajectories_ped_pred_ = []

        for fcounter, p in tqdm(enumerate(all_tracking_files)):
            with open(p, "rb") as f:
                try:
                    av2_annos = pickle.load(f)
                except Exception as e:
                    continue

            tracklets = {}
            annot_city_name = city_name_dict[av2_annos["city_name"]]

            if annot_city_name == city_name:

                ego_pos = np.array([p.translation for p in av2_annos["ego_pos"]])

                # Coordinate transformation
                bb = np.hstack([ego_pos[:, 0:2], np.zeros((len(ego_pos), 1))])
                tmp = transform_t[np.newaxis, :] + transform_c * np.einsum('jk,ik', transform_R, bb)
                ego_pos = tmp[:, 0:2]

                trajectories_pred_.append(ego_pos)

                for anno in av2_annos["results"].keys():
                    for t in av2_annos["results"][anno]:
                        t_id = t["tracking_id"]
                        t_trans = t["translation"]
                        if t_id in tracklets.keys():
                            tracklets[t_id].path.append(t_trans)
                        else:
                            tracklets[t_id] = Tracklet(label=t["label"])
                            tracklets[t_id].path.append(t_trans)

                for counter, tracklet in enumerate(tracklets):
                    tracklet = tracklets[tracklet]

                    if tracklet.label in [1, 2, 3, 4, 5, 7]:  # vehicle
                        tracklet.label = 1
                    else:
                        tracklet.label = 2  # pedestrian


                    tracklet.transform(transform_t, transform_c, transform_R)

                    tracklet = filter_tracklet(tracklet)

                    if tracklet is not None:

                        if tracklet.label == 1: # vehicle
                            trajectories_pred_.append(tracklet.path)
                        else:
                            trajectories_ped_pred_.append(tracklet.path)

        np.save("data/trajectories_pred_{}.npy".format(city_name), trajectories_pred_)
        np.save("data/trajectories_ped_pred_{}.npy".format(city_name), trajectories_ped_pred_)
    else:
        trajectories_pred_ = np.load("data/trajectories_pred_{}.npy".format(city_name), allow_pickle=True)  # PRED VEHICLES
        trajectories_ped_pred_ = np.load("data/trajectories_ped_pred_{}.npy".format(city_name), allow_pickle=True)  # PRED PEDESTRIANS


    # USE PRED TRAJECTORIES
    trajectories_ = trajectories_pred_
    trajectories_ped_ = trajectories_ped_pred_


    # OR USE GT TRAJETORIES
    #trajectories_ = trajectories_gt_


    print("vehicle trajectories: ",  trajectories_.shape)
    print("pedestrian trajectories: ",  trajectories_ped_.shape)


    # Visualize tracklets
    sat_image_viz = sat_image_.copy()

    try:
        tracklets_image = np.asarray(Image.open(os.path.join(args.sat_image_root, "{}-tracklets.png".format(city_name)))).astype(np.uint8)
    except:
        tracklets_image = np.zeros_like(sat_image_viz).astype(np.uint8)

        for t in trajectories_:
            rc = (np.array(plt.get_cmap('viridis')(np.random.rand())) * 255)[0:3]
            rc = (int(rc[0]), int(rc[1]), int(rc[2]))
            for i in range(len(t)-1):
                cv2.line(sat_image_viz, (int(t[i, 0]), int(t[i, 1])), (int(t[i+1, 0]), int(t[i+1, 1])), rc, 1, cv2.LINE_AA)
                cv2.line(tracklets_image, (int(t[i, 0]), int(t[i, 1])), (int(t[i+1, 0]), int(t[i+1, 1])), (255, 0, 0), 7)
        for t in trajectories_ped_pred_:
            rc = (np.array(plt.get_cmap('magma')(np.random.rand())) * 255)[0:3]
            rc = (int(rc[0]), int(rc[1]), int(rc[2]))
            for i in range(len(t)-1):
                cv2.line(sat_image_viz, (int(t[i, 0]), int(t[i, 1])), (int(t[i+1, 0]), int(t[i+1, 1])), rc, 1, cv2.LINE_AA)
                cv2.line(tracklets_image, (int(t[i, 0]), int(t[i, 1])), (int(t[i+1, 0]), int(t[i+1, 1])), (0, 255, 0), 3)

        viz_file = os.path.join(args.sat_image_root, "{}-viz-tracklets.png".format(city_name))
        tracklet_file = os.path.join(args.sat_image_root, "{}-tracklets.png".format(city_name))
        cv2.imwrite(viz_file, cv2.cvtColor(sat_image_viz, cv2.COLOR_RGB2BGR))
        print("Saved tracklet visualization to {}".format(viz_file))
        cv2.imwrite(tracklet_file, tracklets_image)
        print("Saved tracklet visualization to {}".format(tracklet_file))

    drivable_gt = np.asarray(
        Image.open(os.path.join(args.sat_image_root, "{}_drivable.png".format(city_name)))).astype(np.uint8)
    drivable_gt[drivable_gt > 1] = 255


    # single core
    if num_cpus <= 1:
        process_chunk_final(city_name,
                            args.source,
                            trajectories_,
                            trajectories_ped_,
                            lanes_,
                            sat_image_,
                            tracklets_image,
                            drivable_gt,
                            out_path_root,
                            args.max_num_samples,
                            )
    # else:
    #     arguments = zip(repeat(city_name),
    #                     repeat(args.source),
    #                     repeat(trajectories_),
    #                     repeat(trajectories_ped_),
    #                     repeat(lanes_),
    #                     repeat(sat_image_),
    #                     repeat(tracklets_image),
    #                     repeat(out_path_root),
    #                     repeat(args.max_num_samples),
    #                     )
    #
    #
    #     Pool(num_cpus).starmap(process_chunk_final, arguments)



