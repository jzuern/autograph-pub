from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
import cv2
import argparse
from glob import glob
import logging
from av2.datasets.motion_forecasting import scenario_serialization
import pickle
import time
import networkx as nx

from data.av2.settings import *
from aggregation.utils import get_scenario_centerlines, resample_trajectory, Tracklet, \
    filter_tracklet, merge_successor_trajectories, iou_mask, smooth_trajectory, get_endpoints, \
    crop_graph, filter_subgraph


# random shuffle seed
# np.random.seed(seed=int(time.time()))
np.random.seed(seed=0)


def get_dataset_split(city_name, x, y):
    coordinates = city_split_coordinates_dict[city_name]

    # check if in train:
    for train_coord in coordinates["train"]:
        if x >= train_coord[0] and x <= train_coord[2] and y >= train_coord[1] and y <= train_coord[3]:
            return "train"

    # check if in val:
    for val_coord in coordinates["eval"]:
        if x >= val_coord[0] and x <= val_coord[2] and y >= val_coord[1] and y <= val_coord[3]:
            return "eval"

    # check if in test:
    for test_coord in coordinates["test"]:
        if x >= test_coord[0] and x <= test_coord[2] and y >= test_coord[1] and y <= test_coord[3]:
            return "test"

    return None




def crop_img_at_pose(img, pose, crop_size):

    crop_size_large = crop_size * 2

    x, y, yaw = pose
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])

    # Crop source and dest points
    img_crop_ = img[int(y - crop_size_large):int(y + crop_size_large),
                    int(x - crop_size_large):int(x + crop_size_large)].copy()

    # Source points are around center in satellite image crop
    center = np.array([crop_size_large, crop_size_large])

    # For bottom centered
    src_pts = np.array([[-crop_size//2,   0],
                        [-crop_size//2,   -crop_size + 1],
                        [ crop_size//2-1, -crop_size + 1],
                        [ crop_size//2-1, 0]])

    # src_pts_context = np.array([[-256, 128],
    #                             [-256, -383],
    #                             [ 255, -383],
    #                             [ 255, 128]])

    # Rotate source points
    src_pts = (np.matmul(R, src_pts.T).T + center).astype(np.float32)
    # src_pts_context = (np.matmul(R, src_pts_context.T).T + center).astype(np.float32)

    # Destination points are simply the corner points
    dst_pts = np.array([[0, crop_size - 1],
                        [0, 0],
                        [crop_size - 1, 0],
                        [crop_size - 1, crop_size - 1]],
                       dtype="float32")

    # dst_pts_context = np.array([[0, crop_size - 1],
    #                             [0, 0],
    #                             [crop_size - 1, 0],
    #                             [crop_size - 1, crop_size - 1]],
    #                            dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # M_context = cv2.getPerspectiveTransform(src_pts_context, dst_pts_context)

    # directly warp the rotated rectangle to get the straightened rectangle
    try:
        img_crop = cv2.warpPerspective(img_crop_, M, (crop_size, crop_size), cv2.INTER_LINEAR)
        return img_crop
    except:
        logging.debug("Perspective transform failed. Skipping. x, y, yaw: {}".format(pose), exc_info=True)
        return None


# def get_succ_graph_fast(q, succ_traj, sat_image_viz, r_min=10, crop_size=256):
#
#     endpoints = []
#
#     # find endpoints by checking if they are close to the image border
#     for t in succ_traj:
#         if np.any(np.isclose(t[-1], np.array([crop_size-1, crop_size-1]), atol=10.0)) or np.any(np.isclose(t[-1], np.array([0, 0]), atol=10.0)):
#             coords = (int(t[-1, 0]), int(t[-1, 1]))
#             cv2.circle(sat_image_viz, coords, 5, (255, 255, 255), -1)
#         else:
#             coords = (int(t[-1, 0]), int(t[-1, 1]))
#             cv2.circle(sat_image_viz, coords, 5, (129, 129, 129), -1)
#         endpoints.append(coords)
#     endpoints = np.array(endpoints)
#
#     mask_thin = np.zeros(sat_image_viz.shape[0:2], dtype=np.uint8)
#     for t in succ_traj:
#         for i in range(len(t) - 1):
#             x1 = int(t[i][0])
#             y1 = int(t[i][1])
#             x2 = int(t[i + 1][0])
#             y2 = int(t[i + 1][1])
#             cv2.line(mask_thin, (x1, y1), (x2, y2), 1, thickness=2)
#
#     # Cluster endpoints
#     try:
#         clustering = DBSCAN(eps=15, min_samples=DBSCAN_MIN_N_SAMPLES).fit(endpoints)
#     except:
#         logging.debug("DBSCAN endpoint clustering failed. Skipping", exc_info=True)
#         return None, None, None
#
#     endpoints_centroids = []
#     for c in np.unique(clustering.labels_):
#         endpoints_centroids.append(np.mean(endpoints[clustering.labels_ == c], axis=0))
#     endpoints_centroids = np.array(endpoints_centroids)
#
#     # If at least 2 endpoints are found
#     num_endpoints = len(endpoints_centroids)
#     if num_endpoints <= NUM_ENDPOINTS_MIN:
#         logging.debug("Too few endpoints found. Skipping")
#         return None, None, None
#
#     cv2.circle(sat_image_viz, (int(q[0]), int(q[1])), 4, (0, 255, 0), -1)
#     [cv2.circle(sat_image_viz, (int(p[0]), int(p[1])), 4, (0, 0, 0), -1) for p in endpoints_centroids]
#
#     try:
#         sdf_thin = skfmm.distance(1 - mask_thin) - skfmm.distance(mask_thin)
#         sdf_thin = sdf_thin - sdf_thin.min() + 1
#     except:
#         logging.debug("SDF thinning failed. Skipping")
#         return None, None, None
#
#     kernel = cv2.circle(np.zeros((20, 20), np.uint8), (10, 10), 10, 1, -1)
#
#     path_imgs = []
#
#     for pos_end in endpoints_centroids:
#
#         path, cost = skimage.graph.route_through_array(sdf_thin.T,
#                                                        start=tuple(q.astype(np.int32)),
#                                                        end=tuple(pos_end.astype(np.int32)),
#                                                        fully_connected=True)
#         path = np.stack(path, axis=-1)
#         path_img = np.zeros(sat_image_viz.shape[0:2], dtype=np.uint8)
#         path_img[path[1], path[0]] = 1
#         path_imgs.append(cv2.dilate(path_img.astype(np.uint8), kernel, iterations=1))
#
#     mask_succ_sparse = np.max(np.array(path_imgs), axis=0)
#
#     return sat_image_viz, mask_succ_sparse, mask_thin
#




# def get_succ_graph(q, succ_traj, sat_image_viz, r_min=10, crop_size=256):
#
#     endpoints = []
#
#     for t in succ_traj:
#         if np.any(np.isclose(t[-1], np.array([crop_size-1, crop_size-1]), atol=10.0)) or np.any(np.isclose(t[-1], np.array([0, 0]), atol=10.0)):
#             coords = (int(t[-1, 0]), int(t[-1, 1]))
#             cv2.circle(sat_image_viz, coords, 5, (255, 255, 255), -1)
#             endpoints.append(coords)
#     endpoints = np.array(endpoints)
#
#     # sample halting points everywhere
#     points = poisson_disk_sampling(r_min=r_min,
#                                    width=crop_size,
#                                    height=crop_size)
#     edges = get_random_edges(points,
#                              min_point_dist=r_min,
#                              max_point_dist=2*r_min)
#
#     points = [np.array([p[0], p[1]]) for p in points]
#
#     G = nx.DiGraph()
#     for i in range(len(points)):
#         G.add_node(i, pos=points[i], log_odds_dijkstra=0.0, p=0)
#     for i in range(len(edges)):
#         G.add_edge(edges[i][0], edges[i][1], log_odds_dijkstra=0.0, p=0)
#
#     # Assign traversal cost according to tracklets in succ_traj
#     G, mask, mask_thin, sdf, angles = assign_graph_traversals(G, succ_traj, imsize=sat_image_viz.shape[0:2])
#     mask = (mask / 255.).astype(np.float32)
#
#     # Cluster endpoints
#     try:
#         clustering = DBSCAN(eps=15, min_samples=DBSCAN_MIN_N_SAMPLES).fit(endpoints)
#     except:
#         logging.error("DBSCAN endpoint clustering failed. Skipping")
#         return None, None, None, None, None, None, None
#
#     endpoints_centroids = []
#     for c in np.unique(clustering.labels_):
#         endpoints_centroids.append(np.mean(endpoints[clustering.labels_ == c], axis=0))
#     endpoints_centroids = np.array(endpoints_centroids)
#
#     # If at least 2 endpoints are found
#     num_endpoints = len(endpoints_centroids)
#     if num_endpoints < 2:
#         if np.random.rand() < 0.9:  # 90% chance to skip (cause we do not want to skip all)
#             logging.info("Too few endpoints found. Skipping")
#             return None, None, None, None, None, None, None
#
#     # Get planning from query point to end points
#     G_start = np.argmin(np.linalg.norm(np.array([G.nodes[i]["pos"] for i in G.nodes]) - q, axis=1))
#     G_ends = [np.argmin(np.linalg.norm(np.array([G.nodes[i]["pos"] for i in G.nodes]) - endnode, axis=1)) for endnode in endpoints_centroids]
#
#     for G_end in G_ends:
#         path = nx.dijkstra_path(G, G_start, G_end, weight="cost")
#         for i in range(len(path) - 1):
#             G.edges[path[i], path[i + 1]]["p"] = 1
#         for i in range(len(path)):
#             G.nodes[path[i]]["p"] = 1
#
#     # plot start node and end nodes
#     pos_start = G.nodes[G_start]["pos"]
#     pos_ends = [G.nodes[G_end]["pos"] for G_end in G_ends]
#
#     cv2.circle(sat_image_viz, (int(pos_start[0]), int(pos_start[1])), 4, (0, 255, 0), -1)
#     [cv2.circle(sat_image_viz, (int(p[0]), int(p[1])), 4, (0, 0, 0), -1) for p in pos_ends]
#
#     angles_viz = visualize_angles(np.cos(angles),
#                                   np.sin(angles),
#                                   mask=mask[:, :, 0])
#
#     try:
#         sdf_thin = skfmm.distance(1 - mask_thin) - skfmm.distance(mask_thin)
#         sdf_thin = sdf_thin - sdf_thin.min() + 1
#     except:
#         logging.error("SDF thinning failed. Skipping")
#         return None, None, None, None, None, None, None
#
#     kernel = np.zeros((20, 20), np.uint8)
#     kernel = cv2.circle(kernel, (10, 10), 10, 1, -1)
#
#     path_imgs = []
#
#     # if len(pos_ends) < 1:
#     #     logging.error("Too few endpoints found ({}). Skipping".format(len(pos_ends)))
#     #     return None, None, None, None, None, None
#
#     for pos_end in pos_ends:
#
#         path, cost = skimage.graph.route_through_array(sdf_thin.T,
#                                                        start=tuple(pos_start.astype(np.int32)),
#                                                        end=tuple(pos_end.astype(np.int32)),
#                                                        fully_connected=True)
#         path = np.stack(path, axis=-1)
#         path_img = np.zeros_like(sdf, dtype=np.int32)
#         path_img[path[1], path[0]] = 1
#         path_imgs.append(cv2.dilate(path_img.astype(np.uint8), kernel, iterations=1))
#
#     mask_succ_sparse = np.max(np.array(path_imgs), axis=0)
#
#
#     return G, sat_image_viz, mask, angles, angles_viz, mask_succ_sparse, num_endpoints
#
#


# def process_chunk_successors(source, roi_xxyy_list, export_final, trajectories_, trajectories_ped_, lanes_, sat_image_, out_path_root, centerline_image_=None, city_name=None):
#
#     if "tracklets" in source:
#         annotations_centers = [np.mean(t, axis=0) for t in trajectories_]
#         annotations_ped_centers = [np.mean(t, axis=0) for t in trajectories_ped_]
#         annot_ = trajectories_
#         annot_ped_ = trajectories_ped_
#     elif "lanes" in source:
#         annotations_centers = [np.mean(t, axis=0) for t in lanes_]
#         annot_ = lanes_
#     else:
#         raise ValueError("Invalid annotation source")
#
#     for roi_num, roi_xxyy in tqdm(enumerate(roi_xxyy_list), total=len(roi_xxyy_list)):
#
#         if centerline_image_ is not None:
#             if np.sum(centerline_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3]]) == 0:
#                 continue
#
#         if np.random.rand() < 0.1:
#             out_path = os.path.join(out_path_root, "val")
#         else:
#             out_path = os.path.join(out_path_root, "train")
#
#         sample_id = "{}-{}-{}".format(city_name, roi_xxyy[0], roi_xxyy[2])
#
#         if os.path.exists(os.path.join(out_path, "{}.pth".format(sample_id))):
#             continue
#
#         sat_image = sat_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3], :].copy()
#
#         # Filter non-square sat_images:
#         if sat_image.shape[0] != sat_image.shape[1]:
#             continue
#
#         annot_candidates = []
#         annot_ped_candidates = []
#         for i in range(len(annot_)):
#             if np.linalg.norm(annotations_centers[i] - [roi_xxyy[2], roi_xxyy[0]]) < 500:
#                 annot_candidates.append(annot_[i])
#         for i in range(len(annot_ped_)):
#             if np.linalg.norm(annotations_ped_centers[i] - [roi_xxyy[2], roi_xxyy[0]]) < 500:
#                 annot_ped_candidates.append(annot_ped_[i])
#
#         annots = []
#         for annot in annot_candidates:
#             annot = annot - np.array([roi_xxyy[2], roi_xxyy[0]])
#             # filter trajectories according to current roi_xxyy
#             is_in_roi = np.logical_and(annot[:, 0] > 0, annot[:, 0] < sat_image.shape[1])
#             if not np.any(is_in_roi):
#                 continue
#             is_in_roi = np.logical_and(is_in_roi, annot[:, 1] > 0)
#             if not np.any(is_in_roi):
#                 continue
#             is_in_roi = np.logical_and(is_in_roi, annot[:, 1] < sat_image.shape[0])
#             if not np.any(is_in_roi):
#                 continue
#             annot = annot[is_in_roi]
#
#             # resample trajectory to have equally distant points
#             annot = resample_trajectory(annot, dist=5)
#
#             annots.append(annot)
#
#         annots_ped = []
#         for annot in annot_ped_candidates:
#             annot = annot - np.array([roi_xxyy[2], roi_xxyy[0]])
#
#             # filter trajectories according to current roi_xxyy
#             is_in_roi = np.logical_and(annot[:, 0] > 0, annot[:, 0] < sat_image.shape[1])
#             if not np.any(is_in_roi):
#                 continue
#             is_in_roi = np.logical_and(is_in_roi, annot[:, 1] > 0)
#             if not np.any(is_in_roi):
#                 continue
#             is_in_roi = np.logical_and(is_in_roi, annot[:, 1] < sat_image.shape[0])
#             if not np.any(is_in_roi):
#                 continue
#             annot = annot[is_in_roi]
#
#             # resample trajectory to have equally distant points
#             annot = resample_trajectory(annot, dist=5)
#
#             annots_ped.append(annot)
#
#
#         # filter out too short annots
#         annots = [a for a in annots if len(a) > 5]
#         annots_ped = [a for a in annots_ped if len(a) > 5]
#
#         if len(annots) < 1:
#             continue
#
#         min_distance_from_center = min([np.min(np.linalg.norm(trajectory - np.array(sat_image.shape[:2]) / 2, axis=1)) for trajectory in annots])
#         if min_distance_from_center > 30:
#             continue
#
#         query_points = np.random.randint(0, sat_image.shape[0], size=(200, 2))
#
#         # Whether to export sparse or dense trajectory annotations
#         if source == "tracklets_sparse":
#             trajectories = annots  # not filtered
#             output_name = "sparse"
#         elif source == "tracklets_dense":
#             output_name = "dense"
#             trajectories = annots
#         elif source == "lanes":
#             output_name = "lanes"
#             trajectories = annots
#         else:
#             raise ValueError("Invalid source")
#
#         tracklets_im_list = []
#
#         for i_query, q in enumerate(query_points):
#
#             succ_traj, mask_total, sat_image_viz = merge_successor_trajectories(q, trajectories, annots_ped, sat_image)
#
#             if len(succ_traj) < 3:
#                 logging.info("Too few successor trajectories")
#                 continue
#
#             G, sat_image_viz, sdf, angles, angles_viz, mask_succ_sparse = get_succ_graph(q, succ_traj,  sat_image_viz, r_min=r_min)
#             if G is None:
#                 logging.info("Successor graph is None. Skipping")
#                 continue
#
#             # Minimum of X percent of pixels must be covered by the trajectory
#             if np.sum(sdf > 0.01) < 0.02 * np.prod(sdf.shape):
#                 logging.info("Not enough pixels covered with successor graph. Skipping")
#                 continue
#
#             # Must be sufficiently dissimilar from any previous sample
#             max_iou = max([iou_mask(mask_succ_sparse, m) for m in tracklets_im_list]) if len(tracklets_im_list) > 0 else 0
#             if max_iou > 0.7:
#                 logging.info("Sample too similar to previous samples. Skipping")
#                 continue
#             tracklets_im_list.append(mask_succ_sparse)
#
#
#             # Build 3 channel mask
#             mask_all = np.zeros(sat_image.shape, dtype=np.uint8)
#             mask_all[:, :, 0] = (mask_succ_sparse * 255.).astype(np.uint8)
#             mask_all[:, :, 1] = mask_total[:, :, 1]
#             mask_all[:, :, 2] = mask_total[:, :, 2]
#
#
#             pos_encoding = np.zeros(sat_image.shape, dtype=np.float32)
#             x, y = np.meshgrid(np.arange(sat_image.shape[1]), np.arange(sat_image.shape[0]))
#             pos_encoding[q[1], q[0], 0] = 1
#             pos_encoding[..., 1] = np.abs((x - q[0])) / sat_image.shape[1]
#             pos_encoding[..., 2] = np.abs((y - q[1])) / sat_image.shape[0]
#             pos_encoding = (pos_encoding * 255).astype(np.uint8)
#
#             print("-------------\nSaving to {}-{}-{}-{}".format(out_path, sample_id, i_query, output_name))
#
#             Image.fromarray(pos_encoding).save("{}/{}-{}-pos-encoding-{}.png".format(out_path, sample_id, i_query, output_name))
#             Image.fromarray(sat_image).save("{}/{}-{}-rgb.png".format(out_path, sample_id, i_query))
#             Image.fromarray(sat_image_viz).save("{}/{}-{}-rgb-viz.png".format(out_path, sample_id, i_query))
#             #Image.fromarray(angles_viz).save("{}/{}-{}-angles-tracklets-{}.png".format(out_path, sample_id, i_query, output_name))
#             #Image.fromarray((sdf * 255.).astype(np.uint8)).convert("L").save("{}/{}-{}-sdf-tracklets-{}.png".format(out_path, sample_id, i_query, output_name))
#             #Image.fromarray((mask_succ_sparse * 255.).astype(np.uint8)).convert("L").save("{}/{}-{}-sdf-tracklets-{}-sparse.png".format(out_path, sample_id, i_query, output_name))
#             Image.fromarray((mask_all).astype(np.uint8)).save("{}/{}-{}-masks-{}.png".format(out_path, sample_id, i_query, output_name))
#
#
#             if export_final:
#                 sdf_regressor = cv2.imread(os.path.join(out_path.replace("-post", "-pre"), "{}-sdf-reg.png".format(sample_id)))[:,:,0] / 255.
#                 angles_regressor = cv2.imread(os.path.join(out_path.replace("-post", "-pre"), "{}-angles-reg.png".format(sample_id)))
#
#                 # Filter graph according to predicted sdf
#                 G_ = G.copy()
#                 for n in G_.nodes:
#                     pos = G.nodes[n]["pos"]
#                     if sdf_regressor[int(pos[1]), int(pos[0])] < 0.3:
#                         G.remove_node(n)
#
#                 # remap node ids and edges
#                 node_ids = list(G.nodes)
#                 node_id_map = {node_ids[i]: i for i in range(len(node_ids))}
#                 G = nx.relabel_nodes(G, node_id_map)
#
#                 node_probabilities = np.array([G.nodes[n]["p"] for n in G.nodes])
#                 edge_probabilities = np.array([G.edges[e]["p"] for e in G.edges])
#
#                 if np.any(np.isnan(node_probabilities)) or np.any(np.isnan(edge_probabilities)):
#                     continue
#
#                 print("Processing {}...".format(sample_id))
#
#                 cmap = plt.get_cmap('viridis')
#                 norm = plt.Normalize(vmin=0.0, vmax=1.0)
#                 node_colors = cmap(norm(node_probabilities))
#
#                 fig, ax = plt.subplots(figsize=(10, 10))
#                 plt.tight_layout()
#                 ax.set_aspect('equal')
#                 ax.imshow(sat_image)
#
#                 # draw edges
#                 for t in trajectories:
#                     for i in range(len(t) - 1):
#                         ax.arrow(t[i][0], t[i][1],
#                                  t[i + 1][0] - t[i][0],
#                                  t[i + 1][1] - t[i][1],
#                                  color="white", alpha=0.5, width=0.5, head_width=1, head_length=1)
#
#                 edge_colors = cmap(norm(edge_probabilities))
#                 edge_colors[:, -1] = edge_probabilities
#
#                 nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
#                                  edge_color=edge_colors,
#                                  node_color=node_colors,
#                                  with_labels=False,
#                                  node_size=10,
#                                  arrowsize=3.0,
#                                  width=1,
#                                  )
#
#                 plt.savefig("{}/{}-{}-{}.png".format(out_path, sample_id, i_query, output_name), dpi=400)
#                 plt.close(fig)
#
#                 #fig.close()
#
#                 # preprocess sample into pth file
#                 preprocess_sample(G,
#                                   sat_image_=sat_image_,
#                                   pos_encoding=pos_encoding,
#                                   sdf=sdf,
#                                   sdf_regressor=sdf_regressor,
#                                   angles=angles,
#                                   angles_viz=angles_viz,
#                                   angles_regressor=angles_regressor,
#                                   roi_xxyy=roi_xxyy,
#                                   sample_id=sample_id,
#                                   out_path=out_path,
#                                   i_query=i_query,
#                                   output_name=output_name)
#
#     #np.save("data/roi_usable_{}.npy".format(city_name), roi_usable)


# def process_chunk(source, roi_xxyy_list, export_final, trajectories_, lanes_, sat_image_, out_path_root, centerline_image_=None, city_name=None):
#
#
#     if "tracklets" in source:
#         annotations_centers = [np.mean(t, axis=0) for t in trajectories_]
#         annot_ = trajectories_
#     elif "lanes" in source:
#         annotations_centers = [np.mean(t, axis=0) for t in lanes_]
#         annot_ = lanes_
#     else:
#         raise ValueError("Invalid source")
#
#     for roi_num, roi_xxyy in tqdm(enumerate(roi_xxyy_list), total=len(roi_xxyy_list)):
#
#         if centerline_image_ is not None:
#             if np.sum(centerline_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3]]) == 0:
#                 # no centerlines (and therefore no tracklets to be evaluated) in this roi
#                 continue
#
#         if roi_xxyy[0] < 16000:
#             out_path = os.path.join(out_path_root, "val")
#         else:
#             out_path = os.path.join(out_path_root, "train")
#
#         sample_id = "{}-{}-{}".format(city_name, roi_xxyy[0], roi_xxyy[2])
#
#         if os.path.exists(os.path.join(out_path, "{}.pth".format(sample_id))):
#             continue
#
#         sat_image = sat_image_[roi_xxyy[0]:roi_xxyy[1], roi_xxyy[2]:roi_xxyy[3], :].copy()
#
#         # Filter non-square sat_images:
#         if sat_image.shape[0] != sat_image.shape[1]:
#             continue
#
#         annot_candidates = []
#         for i in range(len(annot_)):
#             if np.linalg.norm(annotations_centers[i] - [roi_xxyy[2], roi_xxyy[0]]) < 500:
#                 annot_candidates.append(annot_[i])
#
#         annots = []
#         for annot in annot_candidates:
#
#             annot = annot - np.array([roi_xxyy[2], roi_xxyy[0]])
#
#             # filter trajectories according to current roi_xxyy
#             is_in_roi = np.logical_and(annot[:, 0] > 0, annot[:, 0] < sat_image.shape[1])
#             if not np.any(is_in_roi):
#                 continue
#             is_in_roi = np.logical_and(is_in_roi, annot[:, 1] > 0)
#             if not np.any(is_in_roi):
#                 continue
#
#             is_in_roi = np.logical_and(is_in_roi, annot[:, 1] < sat_image.shape[0])
#             if not np.any(is_in_roi):
#                 continue
#
#
#             annot = annot[is_in_roi]
#
#             # resample trajectory to have equally distant points
#             annot = resample_trajectory(annot, dist=5)
#
#             # filter based on number of points
#             if "tracklets" in source:
#                 if len(annot) < 15:
#                     continue
#
#             # and on physical length of trajectory
#             if "tracklets" in source:
#                 total_length = np.sum(np.linalg.norm(annot[1:] - annot[:-1], axis=1))
#                 if total_length < 50:
#                     continue
#
#             annots.append(annot)
#
#         if len(annots) < 1:
#             continue
#
#         min_distance_from_center = min([np.min(np.linalg.norm(trajectory - np.array(sat_image.shape[:2]) / 2, axis=1)) for trajectory in annots])
#         if min_distance_from_center > 30:
#             continue
#
#         def get_sdf(t):
#             sdf = np.zeros(sat_image.shape[0:2], dtype=np.float32)
#             for i in range(len(t) - 1):
#                 x1 = int(t[i][0])
#                 y1 = int(t[i][1])
#                 x2 = int(t[i + 1][0])
#                 y2 = int(t[i + 1][1])
#                 cv2.line(sdf, (x1, y1), (x2, y2), (1, 1, 1), thickness=5)
#             f = 15  # distance function scale
#             sdf = skfmm.distance(1 - sdf)
#             sdf[sdf > f] = f
#             sdf = sdf / f
#             sdf = 1 - sdf
#
#             return sdf
#
#
#         if "tracklets" in source:
#             # Filter out redundant trajectories:
#             filtered_annots = []
#             global_sdf = np.zeros(sat_image.shape[0:2], dtype=np.float32)
#
#             for t in annots:
#                 if len(filtered_annots) == 0:
#                     filtered_annots.append(t)
#                 t_sdf = get_sdf(t)
#
#                 # get overlap between t_sdf and global_sdf
#                 overlap = np.sum(np.logical_and(t_sdf > 0.1, global_sdf > 0.1)) + 1
#                 t_sdf_sum = np.sum(t_sdf > 0.1)
#
#                 if t_sdf_sum / overlap > 2:
#                     filtered_annots.append(t)
#                     global_sdf += t_sdf
#                 else:
#                     continue
#
#         # Whether to export sparse or dense trajectory annotations
#         if source == "tracklets_sparse":
#             trajectories = filtered_annots
#             output_name = "sparse"
#         elif source == "tracklets_dense":
#             output_name = "dense"
#             trajectories = annots
#         elif source == "lanes":
#             output_name = "lanes"
#             trajectories = annots
#         else:
#             raise ValueError("Invalid source")
#
#
#         G = initialize_graph(roi_xxyy, r_min=r_min)
#
#         for trajectory in trajectories:
#
#             # check length of trajectory
#             if "tracklets" in source:
#                 if np.linalg.norm(trajectory[0] - trajectory[-1]) < 50:
#                     print("skipping trajectory with length < 50")
#                     continue
#
#             # Now we update the angular gridmap
#             for i in range(len(trajectory) - 1):
#                 pos = trajectory[i]
#                 next_pos = trajectory[i + 1]
#                 angle = np.arctan2(next_pos[1] - pos[1], next_pos[0] - pos[0])
#                 G = bayes_update_graph(G, angle, x=pos[0], y=pos[1], p=0.9, r_min=r_min)
#
#         # perform angle kernel density estimation and peak detection
#         G = angle_kde(G)
#
#         # assign edge probabilities according to dijstra-approximated trajectories
#         G, sdf, angles, angles_viz = dijkstra_trajectories(G, trajectories, imsize=sat_image.shape[:2])
#
#         log_odds_e = np.array([G.edges[e]["log_odds_dijkstra"] for e in G.edges])
#         log_odds_n = np.array([G.nodes[n]["log_odds_dijkstra"] for n in G.nodes])
#
#         node_probabilities = np.exp(log_odds_n) / (1 + np.exp(log_odds_n))
#         edge_probabilities = np.exp(log_odds_e) / (1 + np.exp(log_odds_e))
#
#         if np.count_nonzero(edge_probabilities[edge_probabilities > 0.5]) < 10:
#             #print("Only {} edges with probability > 0.5".format(np.count_nonzero(edge_probabilities[edge_probabilities > 0.5])))
#             continue
#
#         #roi_usable[roi_num] = True
#         #continue
#
#         # rescale probabilities
#         node_probabilities = (node_probabilities - np.min(node_probabilities)) / (np.max(node_probabilities) - np.min(node_probabilities))
#         edge_probabilities = (edge_probabilities - np.min(edge_probabilities)) / (np.max(edge_probabilities) - np.min(edge_probabilities))
#
#         node_probabilities = (node_probabilities > 0.1).astype(np.float32)
#         edge_probabilities = (edge_probabilities > 0.1).astype(np.float32)
#
#         # assign probabilities to edges
#         for i, e in enumerate(G.edges):
#             G.edges[e]["p"] = edge_probabilities[i]
#         # assign probabilities to nodes
#         for i, n in enumerate(G.nodes):
#             G.nodes[n]["p"] = node_probabilities[i]
#
#         print("-------------")
#         print("Saving to {}-{}".format(sample_id, output_name))
#
#         Image.fromarray(sat_image).save("{}/{}-rgb.png".format(out_path, sample_id))
#         Image.fromarray(angles_viz).save("{}/{}-angles-tracklets-{}.png".format(out_path, sample_id, output_name))
#         Image.fromarray(sdf * 255.).convert("L").save("{}/{}-sdf-tracklets-{}.png".format(out_path, sample_id, output_name))
#
#         if export_final:
#
#             sdf_regressor = cv2.imread(
#                 os.path.join(out_path.replace("-post", "-pre"), "{}-sdf-reg.png".format(sample_id)))[:,:,0] / 255.
#             angles_regressor = cv2.imread(
#                 os.path.join(out_path.replace("-post", "-pre"), "{}-angles-reg.png".format(sample_id)))
#
#             # Filter graph according to predicted sdf
#             G_ = G.copy()
#             for n in G_.nodes:
#                 pos = G.nodes[n]["pos"]
#                 if sdf_regressor[int(pos[1]), int(pos[0])] < 0.3:
#                     G.remove_node(n)
#
#             # remap node ids and edges
#             node_ids = list(G.nodes)
#             node_id_map = {node_ids[i]: i for i in range(len(node_ids))}
#             G = nx.relabel_nodes(G, node_id_map)
#
#             node_probabilities = np.array([G.nodes[n]["p"] for n in G.nodes])
#             edge_probabilities = np.array([G.edges[e]["p"] for e in G.edges])
#
#             if np.any(np.isnan(node_probabilities)) or np.any(np.isnan(edge_probabilities)):
#                 continue
#
#             print("Processing {}...".format(sample_id))
#
#             cmap = plt.get_cmap('viridis')
#             norm = plt.Normalize(vmin=0.0, vmax=1.0)
#             node_colors = cmap(norm(node_probabilities))
#
#             fig, ax = plt.subplots(figsize=(10, 10))
#             plt.tight_layout()
#             ax.set_aspect('equal')
#             ax.imshow(sat_image)
#
#             # draw edges
#             for t in trajectories:
#                 for i in range(len(t) - 1):
#                     ax.arrow(t[i][0], t[i][1],
#                              t[i + 1][0] - t[i][0],
#                              t[i + 1][1] - t[i][1],
#                              color="white", alpha=0.5, width=0.5, head_width=1, head_length=1)
#
#             for n in G.nodes:
#                 angle_peaks = G.nodes[n]["angle_peaks"]
#                 pos = G.nodes[n]["pos"]
#                 for peak in angle_peaks:
#                     ax.arrow(pos[0], pos[1], np.cos(peak) * 3, np.sin(peak) * 3, color='r', width=0.3)
#
#             edge_colors = cmap(norm(edge_probabilities))
#             edge_colors[:, -1] = edge_probabilities
#
#             nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
#                              edge_color=edge_colors,
#                              node_color=node_colors,
#                              with_labels=False,
#                              node_size=10,
#                              arrowsize=3.0,
#                              width=1,
#                              )
#
#             plt.savefig("{}/{}-{}.png".format(out_path, sample_id, output_name), dpi=400)
#
#             # preprocess sample into pth file
#             preprocess_sample(G,
#                               sat_image_=sat_image_,
#                               sdf=sdf,
#                               sdf_regressor=sdf_regressor,
#                               angles=angles,
#                               angles_viz=angles_viz,
#                               angles_regressor=angles_regressor,
#                               roi_xxyy=roi_xxyy,
#                               sample_id=sample_id,
#                               out_path=out_path,
#                               output_name=output_name)
#
#     np.save("data/roi_usable_{}.npy".format(city_name), roi_usable)


def random_cropping(sat_image, tracklet_image, drivable_gt, trajectories_, crop_size):

    while True:
        randint = np.random.randint(0, len(trajectories_))
        rand_traj = trajectories_[randint]
        rand_point_in_traj = rand_traj[np.random.randint(0, len(rand_traj)-1)]

        pos_x = int(rand_point_in_traj[0])
        pos_y = int(rand_point_in_traj[1])

        # get angle from annotation
        next_x = int(trajectories_[randint][1, 0])
        next_y = int(trajectories_[randint][1, 1])
        angle = np.arctan2(next_y - pos_y, next_x - pos_x)
        angle = angle - np.pi / 2

        # random alteration of angle uniform between -pi/3 and pi/3
        angle = angle + np.random.uniform(-np.pi/3, np.pi/3)

        sat_image_crop = crop_img_at_pose(sat_image, [pos_x, pos_y, angle], crop_size)
        tracklet_image_crop = crop_img_at_pose(tracklet_image, [pos_x, pos_y, angle], crop_size)
        drivable_gt_crop = crop_img_at_pose(drivable_gt, [pos_x, pos_y, angle], crop_size)

        if sat_image_crop is not None:
            break

    crop_center_x = int(pos_x + np.sin(angle) * crop_size / 2)
    crop_center_y = int(pos_y - np.cos(angle) * crop_size / 2)

    return sat_image_crop, tracklet_image_crop, drivable_gt_crop, [crop_center_x, crop_center_y, angle]


def process_samples(args, city_name, trajectories_vehicles_, trajectories_ped_, G_annot,
                        sat_image_, tracklets_image, drivable_gt, out_path_root, max_num_samples=100, crop_size=256, y_min_cut=0):

    if args.source == "lanegraph":

        sample_num = 0
        while sample_num < max_num_samples:

            #print("Processing sample {}...".format(sample_num))

            start_node = np.random.choice(G_annot.nodes)

            # Generate Agent Trajectory
            agent_trajectory = [start_node]
            curr_node = start_node
            for i in range(0, 1000):
                successors = [n for n in G_annot.successors(curr_node)]
                if len(successors) == 0:
                    break
                curr_node = successors[np.random.randint(0, len(successors))]
                agent_trajectory.append(curr_node)


            # leave out the last nodes cause otherwise future trajectory is ending in image
            agent_trajectory = agent_trajectory[0:-10]
            agent_trajectory = agent_trajectory[::5]
            if len(agent_trajectory) == 0:
                continue

            # Iterate over agent trajectory:
            for t in range(0, len(agent_trajectory)-1, 2):

                curr_node = agent_trajectory[t]
                next_node = agent_trajectory[t+1]

                # Get the lane segment between the current and next node
                pos = G_annot.nodes[curr_node]["pos"]
                next_pos = G_annot.nodes[next_node]["pos"]
                curr_lane_segment = next_pos - pos

                # Get angle
                yaw = np.arctan2(curr_lane_segment[1], curr_lane_segment[0]) + np.pi / 2

                ego_x_y_yaw = np.array([pos[0], pos[1], yaw])

                x_true = ego_x_y_yaw[0]
                y_true = ego_x_y_yaw[1]
                yaw_true = ego_x_y_yaw[2]

                x_noise = int(np.random.default_rng().normal(loc=x_true, scale=5, size=1)[0])
                y_noise = int(np.random.default_rng().normal(loc=y_true, scale=5, size=1)[0])
                yaw_noise = np.random.default_rng().normal(loc=yaw_true, scale=0.3, size=1)[0]

                dataset_split = get_dataset_split(city_name, x_noise, y_noise)
                if dataset_split == None:
                    continue
                out_path = os.path.join(out_path_root, dataset_split)
                sample_id = "{}-{}-{}".format(city_name, x_noise, y_noise)

                if os.path.exists(os.path.join(out_path, "{}.pth".format(sample_id))):
                    continue

                # Source points are around center in satellite image crop
                center = np.array([512, 512])

                pos_noise = np.array([x_noise, y_noise])
                R_noise = np.array([[np.cos(yaw_noise), -np.sin(yaw_noise)],
                                    [np.sin(yaw_noise),  np.cos(yaw_noise)]])

                #subgraph_visible = filter_subgraph(G_annot, successor_subgraph, curr_node, max_distance=300)
                subgraph_visible = crop_graph(G_annot, x_noise-500, x_noise+500, y_noise-500, y_noise+500)

                if subgraph_visible.number_of_edges() < 10:
                    continue

                if subgraph_visible.number_of_nodes() < 10:
                    continue

                src_pts = np.array([[-128,  0],
                                    [-128, -255],
                                    [ 127, -255],
                                    [ 127,  0]])
                src_pts = (np.matmul(R_noise, src_pts.T).T + center).astype(np.float32)
                dst_pts = np.array([[0, crop_size - 1],
                                    [0, 0],
                                    [crop_size - 1, 0],
                                    [crop_size - 1, crop_size - 1]],
                                   dtype="float32")

                # the perspective transformation matrix
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                for n in subgraph_visible.nodes():
                    node_pos = G_annot.nodes[n]["pos"]
                    node_pos = node_pos - pos_noise + center
                    node_pos = cv2.perspectiveTransform(node_pos[None, None, :], M)
                    node_pos = node_pos[0, 0, :].astype(np.int32)
                    subgraph_visible.nodes[n]["pos"] = node_pos

                # remove subgraph_visible nodes outside of crop
                for n in list(subgraph_visible.nodes()):
                    node_pos = subgraph_visible.nodes[n]["pos"]
                    if node_pos[0] < 0 or node_pos[0] >= crop_size or node_pos[1] < 0 or node_pos[1] >= crop_size:
                        subgraph_visible.remove_node(n)

                # Skip if no nodes left
                if subgraph_visible.number_of_nodes() == 0:
                    logging.log(logging.INFO, "No nodes left after subgraph_visible cropping")
                    continue

                sat_image_crop_ = sat_image_[int(y_noise - crop_size_large):int(y_noise + crop_size_large),
                                             int(x_noise - crop_size_large):int(x_noise + crop_size_large)].copy()
                sat_image_crop = cv2.warpPerspective(sat_image_crop_, M, (crop_size, crop_size), cv2.INTER_LINEAR)

                drivable_gt_crop_ = drivable_gt[int(y_noise - crop_size_large):int(y_noise + crop_size_large),
                                                       int(x_noise - crop_size_large):int(x_noise + crop_size_large)].copy()
                drivable_gt_crop = cv2.warpPerspective(drivable_gt_crop_, M, (crop_size, crop_size), cv2.INTER_NEAREST)


                # make list of annots out of edges of subgraph
                roots = [n for (n, d) in subgraph_visible.in_degree if d == 0]
                leafs = [n for (n, d) in subgraph_visible.out_degree if d == 0]

                # print("roots: ", roots)
                # print("leafs: ", leafs)

                branches = []
                for root in roots:
                    for path in nx.all_simple_paths(subgraph_visible, root, leafs):
                        if len(path) > 2:
                            branches.append(path)


                # print("branches: ", branches)


                annots_ = []
                for branch in branches:
                    coordinates = [subgraph_visible.nodes[n]["pos"] for n in branch]
                    if len(coordinates) > 1:
                        coordinates = np.array(coordinates)
                        annots_.append(coordinates)

                # print("annots_: ", annots_)


                # split annotation that has a sharp turn
                annots = []
                for annot in annots_:
                    angles = np.arctan2(annot[1:, 1] - annot[:-1, 1], annot[1:, 0] - annot[:-1, 0])
                    # sudden changes in angle
                    change_indices = np.where(np.logical_and(np.abs(np.diff(angles)) > 1 * np.pi / 4,
                                                             np.abs(np.diff(angles)) < 7 * np.pi / 4))[0]
                    if len(change_indices) == 0:
                        annots.append(annot)
                    else:
                        for index in change_indices:
                            annots.append(annot[:index+2])
                            annots.append(annot[index+1:])

                # print("annots: ", annots)

                query_distance_threshold = 10
                joining_distance_threshold = 4
                joining_angle_threshold = np.pi / 8

                query_points = np.array([[crop_size // 2, crop_size - 1]])

                tracklets_im_list = []

                for i_query, q in enumerate(query_points):

                    succ_traj, mask_total, mask_angle_colorized, sat_image_viz = \
                        merge_successor_trajectories(q, annots, sat_image_crop,
                                                     trajectories_ped=[],
                                                     query_distance_threshold=query_distance_threshold,
                                                     joining_distance_threshold=joining_distance_threshold,
                                                     joining_angle_threshold=joining_angle_threshold)

                    num_clusters, endpoints = get_endpoints(succ_traj, crop_size)

                    # print("num_clusters", num_clusters)

                    if num_clusters > 1:
                        sample_type = "branching"
                    else:
                        sample_type = "straight"

                    do_debugging = False
                    # if num_clusters > 1:
                    #     do_debugging = True
                    if do_debugging:
                        sat_image_crop_viz = cv2.cvtColor(sat_image_crop, cv2.COLOR_BGR2RGB)
                        cv2.imshow('sat_image_viz', sat_image_crop_viz)
                        cv2.waitKey(10)

                        def viz(event, mouseX, mouseY, flags, param):
                            if event == cv2.EVENT_LBUTTONDOWN:
                                q = np.array([mouseX, mouseY])
                                print(q)

                                succ_traj, mask_total, mask_angle_colorized, sat_image_viz = \
                                    merge_successor_trajectories(q, annots, sat_image_crop_viz,
                                                                 trajectories_ped=[],
                                                                 query_distance_threshold=query_distance_threshold,
                                                                 joining_distance_threshold=joining_distance_threshold,
                                                                 joining_angle_threshold=joining_angle_threshold)

                                if sat_image_viz is not None:
                                    sat_image_viz = cv2.circle(sat_image_viz, (mouseX, mouseY), 5, (0, 0, 0), -1)
                                    cv2.imshow('sat_image_viz', sat_image_viz)

                        cv2.namedWindow('sat_image_viz')
                        cv2.setMouseCallback('sat_image_viz', viz)
                        cv2.waitKey(1)
                        cv2.waitKey(0)

                    # Filter out all samples that do not fit in quality criteria
                    if len(succ_traj) < 1:
                        logging.debug("Too few successor trajectories")
                        continue

                    # Minimum of X percent of pixels must be covered by the trajectory
                    mask_succ_sparse = mask_total[0].copy()
                    if np.sum(mask_succ_sparse > 128) < FRAC_SUCC_GRAPH_PIXELS * np.prod(
                            mask_succ_sparse.shape) and args.source == "tracklets_joint":
                        logging.debug("Not enough pixels covered with successor graph. Skipping")
                        continue

                    # Must be sufficiently dissimilar from any previous sample
                    max_iou = max([iou_mask(mask_succ_sparse, m) for m in tracklets_im_list]) if len(
                        tracklets_im_list) > 0 else 0
                    if max_iou > IOU_SIMILARITY_THRESHOLD:
                        logging.info("Sample too similar to previous samples. Skipping")
                        continue

                    tracklets_im_list.append(mask_succ_sparse)

                    [cv2.circle(sat_image_viz, (qq[0], qq[1]), 2, (0, 150, 255), -1) for qq in query_points]

                    pos_encoding = np.zeros(sat_image_crop.shape, dtype=np.float32)
                    x, y = np.meshgrid(np.arange(sat_image_crop.shape[1]), np.arange(sat_image_crop.shape[0]))
                    pos_encoding[q[1], q[0], 0] = 1
                    pos_encoding[..., 1] = np.abs((x - q[0])) / sat_image_crop.shape[1]
                    pos_encoding[..., 2] = np.abs((y - q[1])) / sat_image_crop.shape[0]
                    pos_encoding = (pos_encoding * 255).astype(np.uint8)

                    sample_num += 1
                    print("---- TID: {}/{}: Sample {}/{}/{}/{} ({}/{})".format(args.thread_id, args.num_parallel,
                                                                               out_path, sample_type, sample_id,
                                                                               i_query, sample_num, max_num_samples))

                    Image.fromarray(pos_encoding).save(
                        "{}/{}/{}-{}-pos-encoding.png".format(out_path, sample_type, sample_id, i_query))
                    Image.fromarray(sat_image_crop).save(
                        "{}/{}/{}-{}-rgb.png".format(out_path, sample_type, sample_id, i_query))
                    Image.fromarray(sat_image_viz).save(
                        "{}/{}/{}-{}-rgb-viz.png".format(out_path, sample_type, sample_id, i_query))
                    Image.fromarray(mask_total).save(
                        "{}/{}/{}-{}-masks.png".format(out_path, sample_type, sample_id, i_query))
                    Image.fromarray(drivable_gt_crop.astype(np.uint8)).save(
                        "{}/{}/{}-{}-drivable-gt.png".format(out_path, sample_type, sample_id, i_query))
                    Image.fromarray(mask_angle_colorized).save(
                        "{}/{}/{}-{}-angles.png".format(out_path, sample_type, sample_id, i_query))


    elif "tracklets" in args.source:
        annot_veh_ = trajectories_vehicles_
        annot_ped_ = trajectories_ped_

        centers = [np.mean(t, axis=0) for t in annot_veh_]
        # centers_ped = [np.mean(t, axis=0) for t in annot_ped_]

        sample_num = 0
        while sample_num < max_num_samples:

            # this is not guaranteed to give valid crops
            sat_image_crop, tracklet_image_crop, drivable_gt_crop, crop_center = \
                random_cropping(sat_image_, tracklets_image, drivable_gt, annot_veh_, crop_size=crop_size)

            angle = crop_center[2]
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])

            dataset_split = get_dataset_split(city_name, crop_center[0], crop_center[1])

            if dataset_split == None:
                continue

            out_path = os.path.join(out_path_root, dataset_split)

            sample_id = "{}-{}-{}".format(city_name, crop_center[0], crop_center[1])

            if os.path.exists(os.path.join(out_path, "{}.pth".format(sample_id))):
                continue

            annot_candidates = []
            for i in range(len(annot_veh_)):
                if np.linalg.norm(centers[i] - [crop_center[0], crop_center[1]]) < 512:
                    annot_candidates.append(annot_veh_[i])

            # annot_ped_candidates = []
            # for i in range(len(annot_ped_)):
            #     if np.linalg.norm(tracklets_ped_centers[i] - [crop_center[0], crop_center[1]]) < 512:
            #         annot_ped_candidates.append(annot_ped_[i])

            annots = []
            for annot in annot_candidates:
                # transform to crop coordinates
                annot = np.array(annot)
                annot = np.dot(annot - [crop_center[0], crop_center[1]], R) + [crop_size//2, crop_size//2]

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

            # annots_ped = []
            # for annot in annot_ped_candidates:
            #     annot = np.array(annot)
            #     annot = np.dot(annot - [crop_center[0], crop_center[1]], R) + [crop_size//2, crop_size//2]
            #
            #     # filter trajectories according to current roi_xxyy
            #     is_in_roi = np.logical_and(annot[:, 0] > 0, annot[:, 0] < sat_image_crop.shape[1])
            #     if not np.any(is_in_roi):
            #         continue
            #     is_in_roi = np.logical_and(is_in_roi, annot[:, 1] > 0)
            #     if not np.any(is_in_roi):
            #         continue
            #     is_in_roi = np.logical_and(is_in_roi, annot[:, 1] < sat_image_crop.shape[0])
            #     if not np.any(is_in_roi):
            #         continue
            #     annot = annot[is_in_roi]
            #
            #     # resample trajectory to have equally distant points
            #     annot = resample_trajectory(annot, dist=5)
            #
            #     annots_ped.append(annot)

            # filter out too short annots
            annots = [a for a in annots if len(a) > 5]
            # annots_ped = [a for a in annots_ped if len(a) > 5]

            if len(annots) < 1:
                continue

            if args.source == "tracklets_raw":
                query_distance_threshold = 4
                joining_distance_threshold = -1
                joining_angle_threshold = -1
            elif args.source == "tracklets_joint":
                query_distance_threshold = 4
                joining_distance_threshold = 4
                joining_angle_threshold = np.pi/4
            else:
                raise ValueError("Invalid source")

            # Get query points
            if args.query_points == "random":
                query_points = np.argwhere(tracklet_image_crop[:, :, 0] > 0)
                query_points = np.fliplr(query_points)
                np.random.shuffle(query_points)
                query_points = query_points[:NUM_QUERY_POINTS]
            elif args.query_points == "ego":
                query_points = np.array([[crop_size//2, crop_size-1]])
            else:
                raise ValueError("Invalid query_points mode")

            do_debugging = False
            if do_debugging:
                sat_image_crop_viz = cv2.cvtColor(sat_image_crop, cv2.COLOR_BGR2RGB)
                cv2.imshow('sat_image_viz', sat_image_crop_viz)
                cv2.waitKey(10)

                def viz(event, mouseX, mouseY, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        q = np.array([mouseX, mouseY])
                        print(q)

                        succ_traj, mask_total, mask_angle_colorized, sat_image_viz = \
                            merge_successor_trajectories(q, annots, sat_image_crop_viz,
                                                         trajectories_ped=[],
                                                         query_distance_threshold=query_distance_threshold,
                                                         joining_distance_threshold=joining_distance_threshold,
                                                         joining_angle_threshold=joining_angle_threshold)

                        if sat_image_viz is not None:
                            sat_image_viz = cv2.circle(sat_image_viz, (mouseX, mouseY), 5, (0, 0, 0), -1)
                            cv2.imshow('sat_image_viz', sat_image_viz)

                cv2.namedWindow('sat_image_viz')
                cv2.setMouseCallback('sat_image_viz', viz)
                cv2.waitKey(1)
                cv2.waitKey(0)



            tracklets_im_list = []

            for i_query, q in enumerate(query_points):

                succ_traj, mask_total, mask_angle_colorized, sat_image_viz = \
                    merge_successor_trajectories(q, annots, sat_image_crop,
                                                  trajectories_ped=[],
                                                 query_distance_threshold=query_distance_threshold,
                                                 joining_distance_threshold=joining_distance_threshold,
                                                 joining_angle_threshold=joining_angle_threshold)

                num_clusters, _ = get_endpoints(succ_traj, crop_size)

                if num_clusters > 1:
                    sample_type = "branching"
                else:
                    sample_type = "straight"
                # Filter out all samples that do not fit in quality criteria
                if len(succ_traj) < N_MIN_SUCC_TRAJECTORIES:
                    logging.debug("Too few successor trajectories")
                    continue

                # Minimum of X percent of pixels must be covered by the trajectory
                mask_succ_sparse = mask_total[0].copy()
                if np.sum(mask_succ_sparse > 128) < FRAC_SUCC_GRAPH_PIXELS * np.prod(mask_succ_sparse.shape) and args.source == "tracklets_joint":
                    logging.debug("Not enough pixels covered with successor graph. Skipping")
                    continue

                # Must be sufficiently dissimilar from any previous sample
                max_iou = max([iou_mask(mask_succ_sparse, m) for m in tracklets_im_list]) if len(tracklets_im_list) > 0 else 0
                if max_iou > IOU_SIMILARITY_THRESHOLD:
                    logging.info("Sample too similar to previous samples. Skipping")
                    continue

                tracklets_im_list.append(mask_succ_sparse)

                [cv2.circle(sat_image_viz, (qq[0], qq[1]), 2, (0, 150, 255), -1) for qq in query_points]

                pos_encoding = np.zeros(sat_image_crop.shape, dtype=np.float32)
                x, y = np.meshgrid(np.arange(sat_image_crop.shape[1]), np.arange(sat_image_crop.shape[0]))
                pos_encoding[q[1], q[0], 0] = 1
                pos_encoding[..., 1] = np.abs((x - q[0])) / sat_image_crop.shape[1]
                pos_encoding[..., 2] = np.abs((y - q[1])) / sat_image_crop.shape[0]
                pos_encoding = (pos_encoding * 255).astype(np.uint8)

                sample_num += 1
                print("---- TID: {}/{}: Sample {}/{}/{}/{} ({}/{})".format(args.thread_id, args.num_parallel, out_path,
                                                                           sample_type, sample_id, i_query, sample_num,
                                                                           max_num_samples))

                Image.fromarray(pos_encoding).save("{}/{}/{}-{}-pos-encoding.png".format(out_path, sample_type, sample_id, i_query))
                Image.fromarray(sat_image_crop).save("{}/{}/{}-{}-rgb.png".format(out_path, sample_type, sample_id, i_query))
                Image.fromarray(sat_image_viz).save("{}/{}/{}-{}-rgb-viz.png".format(out_path, sample_type, sample_id, i_query))
                Image.fromarray(mask_total).save("{}/{}/{}-{}-masks.png".format(out_path, sample_type, sample_id, i_query))
                Image.fromarray(drivable_gt_crop.astype(np.uint8)).save("{}/{}/{}-{}-drivable-gt.png".format(out_path, sample_type, sample_id, i_query))
                Image.fromarray(mask_angle_colorized).save("{}/{}/{}-{}-angles.png".format(out_path, sample_type, sample_id, i_query))
    else:
        raise ValueError("Invalid source")

if __name__ == "__main__":

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_final", action="store_true")
    parser.add_argument("--city_name", type=str, default="data")
    parser.add_argument("--sat_image_root", type=str, default="/data/lanegraph/woven-data/")
    parser.add_argument("--urbanlanegraph_root", type=str, default="/data/lanegraph/urbanlanegraph-dataset-dev/")
    parser.add_argument("--out_path_root", type=str, default="data")
    parser.add_argument("--num_cpus", type=int, default=1)
    parser.add_argument("--max_num_samples", type=int, default=100, help="Number of samples to generate per city")
    parser.add_argument("--crop_size", type=int, default=256, help="Size of the BEV image crop")
    parser.add_argument("--query_points", type=str, default="ego", choices=[None, "ego", "random"])
    parser.add_argument("--thread_id", type=int, default=0, help="ID of thread from 1 to num_parallel")
    parser.add_argument("--num_parallel", type=int, default=1, help="Number of parallel parsing processes")

    parser.add_argument("--source", type=str, default="tracklets_joint", choices=["tracklets_joint", "tracklets_raw", "lanegraph"])

    args = parser.parse_args()

    print("Parsing config:", args)

    logging.basicConfig(level=logging.DEBUG, filename='aggregate_av2.log', filemode='w')

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

    os.makedirs(os.path.join(out_path_root, 'train', 'branching'), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'train', 'straight'), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'eval', 'branching'), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'eval', 'straight'), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'test', 'branching'), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'test', 'straight'), exist_ok=True)

    sat_image_ = np.asarray(Image.open(os.path.join(args.urbanlanegraph_root, "{}/{}.png".format(city_name, city_name)))).astype(np.uint8)
    drivable_gt = np.asarray(Image.open(os.path.join(args.urbanlanegraph_root, "{}/{}_drivable.png".format(city_name, city_name)))).astype(np.uint8)
    drivable_gt[drivable_gt > 1] = 255

    print("Satellite resolution: {}x{}".format(sat_image_.shape[1], sat_image_.shape[0]))
    print("Exporting {} tracklet annotations!".format(args.source))

    [transform_R, transform_c, transform_t] = get_transform_params(city_name.lower())
    all_tracking_files = glob('/data/argoverse2-full/*_tracking.pickle')

    if not os.path.exists("data/trajectories_gt_{}.npy".format(city_name)):
        print("Generating trajectories")
        trajectories_ = []

        for scenario_path in tqdm(all_tracking_files):
            scenario_path = Path(scenario_path)
            scenario_id = scenario_path.stem.split("_")[-1]
            static_map_path = scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"

            try:
                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            except:
                print("Error loading scenario: {}".format(scenario_path))
                continue

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

        trajectories_ = np.array(trajectories_)

        # save trajectories
        np.save("data/trajectories_gt_{}.npy".format(city_name), trajectories_)
    else:
        trajectories_gt_ = np.load("data/trajectories_gt_{}.npy".format(city_name), allow_pickle=True)  # GT

    if not os.path.exists("data/trajectories_pred_{}.npy".format(city_name)):
        trajectories_pred_ = []
        trajectories_ped_pred_ = []
        num_removed_trajectories = 0; num_kept_trajectories = 0

        for fcounter, p in tqdm(enumerate(all_tracking_files), total=len(all_tracking_files)):
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
                    if tracklet is None:
                        num_removed_trajectories += 1
                    if tracklet is not None:
                        num_kept_trajectories += 1
                        if tracklet.label == 1: # vehicle
                            trajectories_pred_.append(tracklet.path)
                        else:
                            trajectories_ped_pred_.append(tracklet.path)

        print("num_removed_trajectories: ", num_removed_trajectories)
        print("num_kept_trajectories: ", num_kept_trajectories)

        trajectories_pred_ = np.array(trajectories_pred_)
        trajectories_ped_pred_ = np.array(trajectories_ped_pred_)

        np.save("data/trajectories_pred_{}.npy".format(city_name), trajectories_pred_)
        np.save("data/trajectories_ped_pred_{}.npy".format(city_name), trajectories_ped_pred_)
    else:
        trajectories_pred_ = np.load("data/trajectories_pred_{}.npy".format(city_name), allow_pickle=True)  # PRED VEHICLES
        trajectories_ped_pred_ = np.load("data/trajectories_ped_pred_{}.npy".format(city_name), allow_pickle=True)  # PRED PEDESTRIANS

    # get lanes from graph file
    graph_files = glob(os.path.join(args.urbanlanegraph_root, "{}/tiles/*/*.gpickle".format(city_name)))
    G_tiles = []
    for graph_file in graph_files:
        G_tiles.append(nx.read_gpickle(graph_file))

    # join all tiles
    G_annot = nx.DiGraph()
    for G_tile in G_tiles:
        G_annot = nx.union(G_annot, G_tile, rename=("G", "H"))

    G_annot = crop_graph(G_annot, x_min=0, x_max=sat_image_.shape[1], y_min=0, y_max=sat_image_.shape[0])

    # use predicted trajectories
    trajectories_ = trajectories_pred_
    trajectories_ped_ = trajectories_ped_pred_

    trajectories_ = np.array([smooth_trajectory(t, window_size=6) for t in trajectories_])
    trajectories_ped_ = np.array([smooth_trajectory(t, window_size=4) for t in trajectories_ped_])

    y_min_cut = 0

    if args.thread_id > 0:  # if we are parallel
        num_y_pixels = sat_image_.shape[0]
        y_min_cut = int(num_y_pixels * float(args.thread_id - 1) / args.num_parallel)
        y_max_cut = int(num_y_pixels * float(args.thread_id) / args.num_parallel)

        sat_image_ = np.ascontiguousarray(sat_image_[y_min_cut:y_max_cut, :, :])
        drivable_gt = np.ascontiguousarray(drivable_gt[y_min_cut:y_max_cut, :])

        trajectories_ = [t - np.array([0, y_min_cut]) for t in trajectories_]
        trajectories_ped_ = [t - np.array([0, y_min_cut]) for t in trajectories_ped_]

        # delete trajectories that are outside of the image
        trajectories_ = [t for t in trajectories_ if np.all(t[:, 1] >= 0) and np.all(t[:, 1] < sat_image_.shape[0])]
        trajectories_ped_ = [t for t in trajectories_ped_ if np.all(t[:, 1] >= 0) and np.all(t[:, 1] < sat_image_.shape[0])]

        for node in G_annot.nodes:
            G_annot.nodes[node]["pos"][1] = G_annot.nodes[node]["pos"][1] - y_min_cut

        # delete nodes outside of image
        nodes_to_delete = []
        for node in G_annot.nodes:
            if G_annot.nodes[node]["pos"][1] < 0 or G_annot.nodes[node]["pos"][1] >= sat_image_.shape[0]:
                nodes_to_delete.append(node)
        G_annot.remove_nodes_from(nodes_to_delete)

        print("Thread: {}, img shape: {}, len(traj): {}, len(traj_ped): {}, G_annot.number_of_nodes(): {}".
              format(args.thread_id, sat_image_.shape, len(trajectories_), len(trajectories_ped_), G_annot.number_of_nodes()))

        if len(trajectories_) == 0:
            print("No trajectories in this thread. Exiting...")
            exit(0)

    # print("     Number of vehicle trajectories: ",  len(trajectories_))
    # print("     Number of pedestrian trajectories: ",  len(trajectories_ped_))
    #
    # # get summed length of trajectories
    # traj_length = 0
    # for t in trajectories_:
    #     for i in range(len(t)-1):
    #         traj_length += np.linalg.norm(t[i+1] - t[i])
    # print("     Summed length of vehicle trajectories: ", traj_length)
    #
    # traj_length = 0
    # for t in trajectories_ped_:
    #     for i in range(len(t)-1):
    #         traj_length += np.linalg.norm(t[i+1] - t[i])
    # print("     Summed length of pedestrian trajectories: ", traj_length)

    viz_file = os.path.join(args.urbanlanegraph_root, "{}/{}-viz-tracklets.png".format(city_name, city_name))
    tracklet_file = os.path.join(args.urbanlanegraph_root, "{}/{}-tracklets.png".format(city_name, city_name))

    # # Visualize tracklets
    # sat_image_viz = sat_image_.copy()
    #
    # tracklets_image = np.zeros_like(sat_image_viz).astype(np.uint8)
    #
    # for t in tqdm(trajectories_):
    #     rc = (np.array(plt.get_cmap('viridis')(np.random.rand())) * 255)[0:3]
    #     rc = (int(rc[0]), int(rc[1]), int(rc[2]))
    #     for i in range(len(t)-1):
    #         cv2.line(sat_image_viz, (int(t[i, 0]), int(t[i, 1])), (int(t[i+1, 0]), int(t[i+1, 1])), rc, 1, cv2.LINE_AA)
    #         cv2.line(tracklets_image, (int(t[i, 0]), int(t[i, 1])), (int(t[i+1, 0]), int(t[i+1, 1])), (255, 0, 0), 7)
    # for t in tqdm(trajectories_ped_):
    #     rc = (np.array(plt.get_cmap('magma')(np.random.rand())) * 255)[0:3]
    #     rc = (int(rc[0]), int(rc[1]), int(rc[2]))
    #     for i in range(len(t)-1):
    #         cv2.line(sat_image_viz, (int(t[i, 0]), int(t[i, 1])), (int(t[i+1, 0]), int(t[i+1, 1])), rc, 1, cv2.LINE_AA)
    #         cv2.line(tracklets_image, (int(t[i, 0]), int(t[i, 1])), (int(t[i+1, 0]), int(t[i+1, 1])), (0, 255, 0), 3)
    #
    # cv2.imwrite(viz_file, cv2.cvtColor(sat_image_viz, cv2.COLOR_RGB2BGR))
    # print("Saved tracklet visualization to {}".format(viz_file))
    # cv2.imwrite(tracklet_file, tracklets_image)
    # print("Saved tracklet visualization to {}".format(tracklet_file))
    # del sat_image_viz

    tracklets_image = np.asarray(Image.open(tracklet_file)).astype(np.uint8)
    tracklets_image = cv2.cvtColor(tracklets_image, cv2.COLOR_BGR2RGB)

    # single core
    if num_cpus <= 1:
        process_samples(args,
                            city_name,
                            trajectories_,
                            trajectories_ped_,
                            G_annot,
                            sat_image_,
                            tracklets_image,
                            drivable_gt,
                            out_path_root,
                            args.max_num_samples,
                            crop_size=args.crop_size,
                            y_min_cut=y_min_cut)