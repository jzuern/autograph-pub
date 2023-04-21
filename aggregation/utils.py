import numpy as np
import networkx as nx
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import cv2
import skfmm
from pathlib import Path
from scipy.spatial.distance import cdist
import torch
import os
from sklearn.cluster import DBSCAN
from collections import defaultdict
from shapely.geometry import LineString
import time

from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap

import matplotlib.pyplot as plt
from lanegnn.utils import poisson_disk_sampling, get_random_edges, visualize_angles,  get_oriented_crop, transform2vgg


class AngleColorizer:
    def __init__(self):
        pass

    def safety_check(self, angle):
        if angle.min() < 0 or angle.max() > 2 * np.pi:
            min_diff = np.abs(angle.min() - 0)
            max_diff = np.abs(angle.max() - 2 * np.pi)
            if min_diff < 0.1 and max_diff < 0.1:
                angle = np.clip(angle, 0, 2 * np.pi)
            else:
                print(angle.min(), angle.max())
                raise ValueError("Angle must be in the range [0, 2*pi]")
        return angle



    def angle_to_color(self, angle, mask=None):
        """
        Converts an angle to a color. Angle is in radians and is in the range [0, 2*pi].
        :param angle:
        :return:  A 3-tuple representing the RGB color.
        """
        angle = self.safety_check(angle)

        cmap = plt.get_cmap("hsv")
        angle_hsv = (cmap(angle / (2 * np.pi))[:, :, :3] * 255).astype(np.uint8)

        if mask is not None:
            angle_hsv = angle_hsv * mask[:, :, None]

        return angle_hsv

    def angles_to_colors(self, angles):
        """
        Converts a list of angles to a list of colors. Angle is in radians and is in the range [0, 2*pi].
        :param angles:
        :return:  A list of 3-tuples representing the RGB colors.
        """
        return [self.angle_to_color(angle) for angle in angles]

    def color_to_angle(self, color_image):
        """
        Converts a color to an angle. Angle is in radians and is in the range [0, 2*pi].
        :param color: A 3-tuple representing the RGB color.
        :return:
        """
        color_hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        return color_hsv[:, :, 0] / 255 * 2 * np.pi * 1.4410


    def colors_to_angles(self, colors):
        """
        Converts a list of colors to a list of angles. Angle is in radians and is in the range [0, 2*pi].
        :param colors:
        :return:
        """
        return [self.color_to_angle(color) for color in colors]

    def angle_to_xy(self, angle):
        """
        Converts an angle to a 2D vector. Angle is in radians and is in the range [0, 2*pi].
        :param angle:
        :return:
        """
        angle = self.safety_check(angle)

        return np.array([np.cos(angle), np.sin(angle)])

    def angles_to_xys(self, angles):
        """
        Converts a list of angles to a list of 2D vectors. Angle is in radians and is in the range [0, 2*pi].
        :param angles:
        :return:
        """
        return [self.angle_to_xy(angle) for angle in angles]

    def xy_to_angle(self, xy):
        """
        Converts a 2D vector to an angle. Angle is in radians and is in the range [0, 2*pi].
        :param xy:
        :return:
        """
        return np.arctan2(xy[1], xy[0]) % (2 * np.pi)



class Cropper:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def crop(self, image, center, angle):
        pass

    def crop_batch(self, images, centers, angles):
        pass

    def filter_roi(self, trajectories, center_x, center_y, angle):
        pass


# Smooth trajectories with a median filter
def smooth_trajectory(traj, window_size=5):
    """
    Smooths a 2D trajectory using a moving average filter.

    Parameters:
    traj (ndarray): A 2D numpy array of shape (N, 2), where N is the number of points in the trajectory.
    window_size (int): The size of the moving window used to compute the moving average. Default is 5.

    Returns:
    smoothed_traj (ndarray): A 2D numpy array of shape (N, 2), where N is the number of points in the trajectory,
    with the smoothed x and y coordinates.
    """

    # Pad the trajectory with zeros at the beginning and end to handle edge cases
    padded_traj = np.pad(traj, ((window_size // 2, window_size // 2), (0, 0)), mode='edge')

    # Compute the moving average of the x and y coordinates separately
    smoothed_x = np.convolve(padded_traj[:, 0], np.ones(window_size) / window_size, mode='valid')
    smoothed_y = np.convolve(padded_traj[:, 1], np.ones(window_size) / window_size, mode='valid')

    # Combine the smoothed x and y coordinates into a single trajectory
    smoothed_traj = np.column_stack((smoothed_x, smoothed_y))

    return smoothed_traj


def get_endpoints(succ_traj, crop_size):
    endpoints = []

    # find endpoints by checking if they are close to the image border
    for t in succ_traj:
        if np.any(np.isclose(t[-1], np.array([crop_size - 1, crop_size - 1]), atol=10.0)) or np.any(
                np.isclose(t[-1], np.array([0, 0]), atol=10.0)):
            coords = (int(t[-1, 0]), int(t[-1, 1]))
            endpoints.append(coords)
    endpoints = np.array(endpoints)

    # Cluster endpoints
    try:
        clustering = DBSCAN(eps=40, min_samples=1).fit(endpoints)
    except:
        return 0, endpoints

    endpoints_centroids = []
    for c in np.unique(clustering.labels_):
        endpoints_centroids.append(np.mean(endpoints[clustering.labels_ == c], axis=0))
    endpoints_centroids = np.array(endpoints_centroids)

    return len(endpoints_centroids), endpoints




def iou_mask(mask1, mask2):
    # Calculates IoU between two binary masks
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero( np.logical_and(mask1, mask2) )
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou



# merge function to  merge all sublist having common elements.
def merge_common(lists):
    neigh = defaultdict(set)
    visited = set()
    for each in lists:
        for item in each:
            neigh[item].update(each)

    def comp(node, neigh=neigh, visited=visited, vis=visited.add):
        nodes = set([node])
        next_node = nodes.pop
        while nodes:
            node = next_node()
            vis(node)
            nodes |= neigh[node] - visited
            yield node

    for node in neigh:
        if node not in visited:
            yield sorted(comp(node))

def connected_component_subgraphs(G):
    # make an undirected copy of the digraph
    UG = G.to_undirected()
    for c in nx.connected_components(UG):
        yield UG.subgraph(c)



def get_supernodes(G, max_distance=0.1):

    # Max distance is in unit pixel
    print("Getting supernodes for graph with {} nodes".format(G.number_of_nodes()))

    waypoints = nx.get_node_attributes(G, 'pos')
    nodes = np.array([waypoints[i] for i in G.nodes()])
    node_name_dict = {i: node_name for i, node_name in enumerate(G.nodes())}

    distance_matrix = cdist(nodes, nodes, metric='euclidean')
    close_indices = np.argwhere(distance_matrix < max_distance)

    # convert according to node_name_dict
    close_indices_ = []
    for i, j in close_indices:
        close_indices_.append([node_name_dict[i], node_name_dict[j]])
    close_indices = close_indices_

    close_indices_sets = list(merge_common(close_indices))

    print("Getting supernodes for graph with {} nodes... done! Found {} supernodes.".format(G.number_of_nodes(), len(close_indices_sets)))

    return close_indices_sets


def crop_graph(g, edge_pos, x_min, x_max, y_min, y_max):

    t0 = time.time()

    edge_mask = np.vstack((edge_pos[0, :, 0] > x_min, edge_pos[0, :, 0] < x_max, edge_pos[0, :, 1] > y_min, edge_pos[0, :, 1] < y_max,
                           edge_pos[1, :, 0] > x_min, edge_pos[1, :, 0] < x_max, edge_pos[1, :, 1] > y_min, edge_pos[1, :, 1] < y_max))
    edge_mask = np.all(edge_mask, axis=0)

    edges_to_keep = np.array(g.edges())[edge_mask]


    # # build new graph
    g_copy = nx.DiGraph()
    for edge in edges_to_keep:
        g_copy.add_edge(edge[0], edge[1])

    print("     Cropping graph with {} nodes to {} nodes... done! Took {:.2f}s".format(g.number_of_nodes(), g_copy.number_of_nodes(), time.time()-t0))

    return g_copy


def remove_redundant_nodes(obj_boxes, obj_relation_triplets, redundant_distance_threshold_px=10.0):

    # remove "1" column
    obj_relation_triplets = obj_relation_triplets[:, 0:2]

    # find close indices
    distance_matrix = cdist(obj_boxes, obj_boxes, metric='euclidean')
    close_indices = np.argwhere(distance_matrix < redundant_distance_threshold_px)

    # remove rows with same values
    close_indices = np.delete(close_indices, np.where((close_indices[:, 0] == close_indices[:, 1])), axis=0)

    close_indices_sets = list(merge_common(close_indices))

    # make dict
    map_to_from = {}
    for c in close_indices_sets:
        c = list(c)  # convert set to list
        idx_to = c[0]
        indices_from = c[1:]
        map_to_from[idx_to] = indices_from

    # replace redundant nodes
    obj_relation_triplets_ = obj_relation_triplets.copy()
    to_remove_indices = set()

    for index_to, indices_from in map_to_from.items():
        for index_from in indices_from:
            # print('Replacing node ', index_from, ' with ', index_to)
            obj_relation_triplets_[obj_relation_triplets == index_from] = index_to
            to_remove_indices.add(index_from)
    obj_relation_triplets = obj_relation_triplets_.copy()
    to_remove_indices = list(to_remove_indices)

    # Now we remove the non-connected nodes from the graph
    curr_indices = range(len(obj_boxes))
    new_indices = []
    counter = 0
    for i in curr_indices:
        new_indices.append(counter)
        if i not in to_remove_indices:
            counter += 1

    index_map = dict(zip(curr_indices, new_indices))

    # perform remapping in obj_relation_triplets
    obj_relation_triplets_ = obj_relation_triplets.copy()
    for i in range(len(obj_relation_triplets)):
        obj_relation_triplets_[i, 0] = index_map[obj_relation_triplets[i, 0]]
        obj_relation_triplets_[i, 1] = index_map[obj_relation_triplets[i, 1]]
    obj_relation_triplets = obj_relation_triplets_.copy()

    # Make unique
    obj_relation_triplets = np.unique(obj_relation_triplets, axis=0)

    valid_indices = np.ones(len(obj_boxes), dtype=np.uint8)
    valid_indices[to_remove_indices] = 0
    valid_indices = valid_indices.astype(np.bool)

    obj_boxes = obj_boxes[valid_indices]

    # Generate new obj_relations
    obj_relations = np.zeros([len(obj_boxes), len(obj_boxes)], dtype=np.uint8)
    for triplet in obj_relation_triplets:
        obj_relations[triplet[0], triplet[1]] = 1

    # add "1" column
    obj_relation_triplets = np.hstack([obj_relation_triplets, np.ones([len(obj_relation_triplets), 1], dtype=np.uint8)])

    return obj_boxes, obj_relation_triplets, obj_relations, distance_matrix


def filter_subgraph(G, subgraph, ego_node, max_distance=256):
    """
    Filters subgraph based on spatial distance to ego-node. If a node is removed, all downstream nodes are also removed.
    :param subgraph:
    :param ego_node:
    :return:
    """

    subgraph_from_ego = nx.bfs_tree(subgraph, ego_node)

    # Remove all nodes that are too far away from the ego-node
    subgraph_from_ego_ = subgraph_from_ego.copy()
    for n in subgraph_from_ego_.nodes():
        if np.linalg.norm(G.nodes[n]["pos"] - G.nodes[ego_node]["pos"]) > max_distance:
            subgraph_from_ego.remove_node(n)

    subgraph_from_ego_ = subgraph_from_ego.copy()
    # Remove all nodes without connection to ego_node
    for n in subgraph_from_ego_.nodes():
        if not nx.has_path(subgraph_from_ego, ego_node, n):
            subgraph_from_ego.remove_node(n)

    return subgraph_from_ego


def merge_close_graph_nodes(graph):

    node_positions = np.array([graph.nodes[n]["pos"] for n in graph.nodes()])

    edges = np.array(list(graph.edges()))
    edge_triplets = np.vstack((edges[:, 0], edges[:, 1], np.ones(len(edges), dtype=np.uint8))).T

    print("1-Number of edges before: ", len(edge_triplets))

    # # remove overlapping nodes with very similar coordinates and connect
    node_positions, edge_triplets, _, _ = remove_redundant_nodes(node_positions, edge_triplets, redundant_distance_threshold_px=10.0)
    print("2-Number of edges after removing redundant nodes: ", len(edge_triplets))

    # Convert to networkx directional graph
    G = nx.DiGraph()

    for ii, node_pos in enumerate(node_positions):
        G.add_node(ii, pos=node_pos[0:2])
    for e in edge_triplets:
        G.add_edge(e[0], e[1])

    return G


def find_successor_annotations(annot):

    # convert to shapely linestrings
    lines = [LineString([(l[0, 0], l[0, 1]), (l[1, 0], l[1, 1])]) for l in annot if l.shape[0] == 2]

    # cluster lines into connected components
    clusters = []
    for line in lines:
        if len(clusters) == 0:
            clusters.append([line])
        else:
            found = False
            for cluster in clusters:
                for cline in cluster:
                    if line.distance(cline) < 1e-3:
                        cluster.append(line)
                        found = True
                        break
                if found:
                    break
            if not found:
                clusters.append([line])

    # convert back to annot
    annot_successor = []
    for cluster in clusters:
        annot_c = []
        for line in cluster:
            annot_c.append(np.array(line.coords))

        annot_c = np.vstack(annot_c)

        # # take first element and then every second element
        # annot_c0 = annot_c[0, :]
        # annot_c1 = annot_c[1::2, :]
        # annot_c = np.vstack([annot_c0, annot_c1])

        annot_successor.append(annot_c)

    return annot_successor


def merge_successor_trajectories(q, trajectories_all,
                                 trajectories_ped=[],
                                 query_distance_threshold=4,  # in px
                                 joining_distance_threshold=4,  # in px
                                 joining_angle_threshold=np.pi/4  # in rad
     ):

    # Get all trajectories that go through query point

    tracklets_viz = np.zeros([256, 256, 3], dtype=np.uint8)

    trajectories_close_q = []
    for t in trajectories_all:
        min_distance_from_q = np.min(np.linalg.norm(t - q, axis=1))
        if min_distance_from_q < query_distance_threshold:
            closest_index = np.argmin(np.linalg.norm(t - q, axis=1))
            t = t[closest_index:]
            if len(t) > 2:
                trajectories_close_q.append(t)

    for t in trajectories_all:
        for i in range(len(t)-1):
            cv2.line(tracklets_viz, tuple(t[i].astype(int)), tuple(t[i+1].astype(int)), (0, 0, 255), 1, cv2.LINE_AA)


    # then get all trajectories that are close to any of the trajectories
    trajectories_2 = []
    for t0 in trajectories_close_q:
        for t1 in trajectories_all:

            angles0 = np.arctan2(t0[1:, 1] - t0[:-1, 1],
                                 t0[1:, 0] - t0[:-1, 0]) + np.pi
            angles0 = np.concatenate([angles0, [angles0[-1]]])

            angles1 = np.arctan2(t1[1:, 1] - t1[:-1, 1],
                                 t1[1:, 0] - t1[:-1, 0]) + np.pi
            angles1 = np.concatenate([angles1, [angles1[-1]]])

            # check if t1 is close to t0 at any point
            min_dist = np.amin(cdist(t0, t1), axis=0)
            min_angle = np.amin(cdist(angles0[:, np.newaxis],
                                      angles1[:, np.newaxis]), axis=0)

            crit_angle = min_angle < joining_angle_threshold
            crit_dist = min_dist < joining_distance_threshold

            # logical AND
            crit = crit_angle * crit_dist

            if np.any(crit):
                # if so, merge the trajectories
                # find the first point where the criteria is met
                first_crit = np.where(crit)[0][0]

                trajectories_2.append(t1[first_crit:])

                # fig, axarr = plt.subplots(1, 4, figsize=(10, 5))
                # axarr[0].imshow(sat_image_viz, alpha=0.5)
                # axarr[0].plot(t0[:, 0], t0[:, 1], 'rx')
                # axarr[0].plot(t1[:, 0], t1[:, 1], 'b.')
                #
                # axarr[1].plot(angles0, 'r')
                # axarr[1].plot(angles1, 'b')
                # axarr[1].title.set_text("angles")
                #
                # axarr[2].plot(min_dist, 'k')
                # axarr[2].title.set_text("min_dist")
                # axarr[3].plot(min_angle, 'k')
                # axarr[3].title.set_text("min_angle")
                #
                # plt.show()

                cv2.circle(tracklets_viz, tuple(t1[first_crit].astype(int)), 2, (0, 255, 0), -1)



    for t2 in trajectories_2:
        for i in range(len(t2)-1):
            cv2.line(tracklets_viz, tuple(t2[i].astype(int)), tuple(t2[i+1].astype(int)), (255, 0, 0), 1, cv2.LINE_AA)

    mask_ped = np.zeros(tracklets_viz.shape[0:2], dtype=np.uint8)
    mask_veh = np.zeros(tracklets_viz.shape[0:2], dtype=np.uint8)

    # Paint into vehicle and pedestrian masks
    for t in trajectories_all:
        for i in range(len(t)-1):
            cv2.line(mask_veh, tuple(t[i].astype(int)), tuple(t[i+1].astype(int)), (255), 7)
    for t in trajectories_ped:
        for i in range(len(t)-1):
            cv2.line(mask_ped, tuple(t[i].astype(int)), tuple(t[i+1].astype(int)), (255), 7)
            cv2.line(tracklets_viz, tuple(t[i].astype(int)), tuple(t[i+1].astype(int)), (0, 255, 0), 1, cv2.LINE_AA)

    # paint into angle mask
    mask_angle = np.zeros(tracklets_viz.shape[0:2], dtype=np.float32)
    for t in trajectories_all:
        for i in range(len(t) - 1):
            start = tuple(t[i].astype(int))
            end = tuple(t[i + 1].astype(int))
            angle = np.arctan2(end[1] - start[1], end[0] - start[0]) + np.pi
            cv2.line(mask_angle, start, end, angle, 7)

    # convert using color map
    mask_angle_colorized = AngleColorizer().angle_to_color(mask_angle)
    mask_angle_colorized[mask_angle == 0] = 0

    succ_traj = trajectories_2 + trajectories_close_q

    # visualize succ traj in sat image viz and in mask_total
    mask_succ = np.zeros(tracklets_viz.shape[0:2], dtype=np.uint8)

    for t in succ_traj:
        for i in range(len(t)-1):
            cv2.line(mask_succ, tuple(t[i].astype(int)), tuple(t[i+1].astype(int)), (255), 7)
            cv2.line(tracklets_viz, tuple(t[i].astype(int)), tuple(t[i+1].astype(int)), (255, 0, 0), 1, cv2.LINE_AA)

    mask_total = np.zeros(tracklets_viz.shape, dtype=np.uint8)
    mask_total[:, :, 0] = mask_succ
    mask_total[:, :, 2] = mask_ped
    mask_total[:, :, 1] = mask_veh

    return succ_traj,  mask_total, mask_angle_colorized, tracklets_viz



class Tracklet(object):
    def __init__(self, label):
        self.label = label
        self.path = []
        self.timesteps = []

    def transform(self, t, c, R):
        self.path = np.array(self.path)
        traj = self.path[:, 0:2]

        # Coordinate transformation
        bb = np.hstack([traj, np.zeros((len(traj), 1))])
        tmp = t[np.newaxis, :] + c * np.einsum('jk,ik', R, bb)
        traj = tmp[:, 0:2]

        self.path = traj


def filter_tracklet(tracklet):

    traj = np.array(tracklet.path)
    traj = traj[:, 0:2]

    # ALL UNITS IN PIX
    if tracklet.label == 1:  # vehicle
        MIN_TRAJ_LEN = 50
        MIN_START_END_DIFF = 50
        MIN_TIMESTEPS = 5
        MAX_JUMPSIZE = 20
    else:                   # pedestrian
        MIN_TRAJ_LEN = 5
        MIN_START_END_DIFF = 5
        MIN_TIMESTEPS = 5
        MAX_JUMPSIZE = 10

    # Based on overall length
    total_length = np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1))
    if total_length < MIN_TRAJ_LEN:
        return None

    # Based on start end diff
    start_end_diff = np.linalg.norm(traj[0, :] - traj[-1, :])
    if start_end_diff < MIN_START_END_DIFF:
        return None

    # Based on number of timesteps
    if len(traj) < MIN_TIMESTEPS:
        return None

    # Remove big jumps in trajectory
    if np.max(np.linalg.norm(traj[1:] - traj[:-1], axis=1)) > MAX_JUMPSIZE:
        return None

    tracklet.path = traj

    return tracklet


def preprocess_sample(G_gt_nx, pos_encoding, sat_image_, sdf, sdf_regressor, angles, angles_viz, angles_regressor, roi_xxyy,
                      sample_id, out_path, i_query=0, output_name=''):

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
    pos_encoding = (np.asarray([np.pad(pos_encoding[..., i], margin, mode="constant", constant_values=0)
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
        pos_encoding_ = transform2vgg(pos_encoding).unsqueeze(0).numpy()

        feats = np.concatenate([crop_img_rgb_, crop_sdf_, crop_angles_viz_, pos_encoding_], axis=1)

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

    out_file = os.path.join(out_path, "{}-{}-{}.pth".format(sample_id, i_query, output_name))
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






def to_graph(trajectory):
    g = nx.DiGraph()
    for i in range(len(trajectory) - 1):
        angle = np.arctan2(trajectory[i+1][1] - trajectory[i][1], trajectory[i+1][0] - trajectory[i][0])
        g.add_edge(trajectory[i], trajectory[i + 1], angle=angle)
    return g




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





def assign_graph_traversals(G, trajectories, imsize):

    global_mask = np.zeros((imsize[0], imsize[1], 3), dtype=np.uint8)
    global_mask_thin = np.zeros((imsize[0], imsize[1]), dtype=np.uint8)
    global_angle = np.zeros((imsize[0], imsize[1]), dtype=np.float32)
    sdf = np.zeros((imsize[0], imsize[1]), dtype=np.float32)

    for t in trajectories:
        for i in range(len(t) - 1):
            x1 = int(t[i][0])
            y1 = int(t[i][1])
            x2 = int(t[i + 1][0])
            y2 = int(t[i + 1][1])
            angle = np.arctan2(y2 - y1, x2 - x1)
            cv2.line(global_angle, (x1, y1), (x2, y2), angle, thickness=4)
            cv2.line(global_mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=4)
            cv2.line(global_mask_thin, (x1, y1), (x2, y2), 1, thickness=2)
            cv2.line(sdf, (x1, y1), (x2, y2), 1, thickness=2)

    f = 15  # distance function scale
    try:
        sdf = skfmm.distance(1 - sdf)
    except:
        pass
    sdf[sdf > f] = f
    sdf = sdf / f

    # assign cost to nodes and edges
    for e in G.edges:
        start = G.nodes[e[0]]["pos"]
        end = G.nodes[e[1]]["pos"]
        midpoint = (start + end) / 2
        try:
            G.edges[e]["cost"] = sdf[int(midpoint[1]), int(midpoint[0])] ** 0.5 + \
                                 sdf[int(start[1]), int(start[0])] ** 0.5 +  \
                                 sdf[int(end[1]), int(end[0])] ** 0.5
        except Exception as e:
            print(e)
            continue

    return G, global_mask, global_mask_thin, sdf, global_angle


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


    angles_image_list = []

    for t in trajectories:

        # create sdf of trajectory
        sdf = np.zeros(imsize, dtype=np.float32)
        traj_angle = np.zeros(imsize, dtype=np.float32)

        for i in range(len(t) - 1):
            x1 = int(t[i][0])
            y1 = int(t[i][1])
            x2 = int(t[i + 1][0])
            y2 = int(t[i + 1][1])
            cv2.line(sdf, (x1, y1), (x2, y2), 1, 1)

            angle = np.arctan2(y2 - y1, x2 - x1)
            cv2.line(global_angle, (x1, y1), (x2, y2), angle, thickness=12)
            cv2.line(traj_angle, (x1, y1), (x2, y2), angle, thickness=12)
            cv2.line(global_mask, (x1, y1), (x2, y2), (1, 1, 1), thickness=5)
        angles_image_list.append(traj_angle)

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

    # now we build up the divergence map. for each angle image we compare with the other angle images and save the maximum angle difference
    # this is a measure of how much the angle changes at each pixel

    angle_diff_map = np.zeros_like(global_angle, dtype=np.float32)
    for i in range(len(angles_image_list)):
        for j in range(i+1, len(angles_image_list)):
            angle_diff = np.abs(angles_image_list[i] - angles_image_list[j])
            angle_diff[angles_image_list[i] * angles_image_list[j] == 0] = 0
            angle_diff[angle_diff < np.pi / 4] = 0
            angle_diff_map += angle_diff

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


def mean_angle_abs_diff(x, y):
    period = 2 * np.pi
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]
    return np.abs(diff)


def similarity_check(pose, pose_list, min_dist=10, min_angle=np.pi/4):
    if len(pose_list) == 0:
        return False
    poslist = np.array(pose_list)[:, :2]
    anglelist = np.array(pose_list)[:, 2:]

    euclidean_distances = cdist(pose[np.newaxis, 0:2], poslist, metric='euclidean')
    angle_distances = cdist(pose[np.newaxis, 2:], anglelist, lambda u, v: mean_angle_abs_diff(u, v))

    position_criterium = euclidean_distances < min_dist
    angle_criterium = angle_distances < min_angle
    criterium = position_criterium & angle_criterium

    # Check criterium sum
    if np.sum(criterium) > 0:
        return True
    else:
        return False


def out_of_bounds_check(pose, satellite_shape, oob_margin=400):
    if pose[0] < oob_margin or pose[0] > satellite_shape[1] - oob_margin or pose[1] < oob_margin or pose[1] > satellite_shape[0] - oob_margin:
        return True
    else:
        return False



def visualize_graph(G, ax, node_color=np.array([255, 0, 142]) / 255., edge_color=np.array([255, 0, 142]) / 255.):

    '''
    Visualize a lane graph on an axis
    Args:
        G: graph
        ax:  axis object
        node_color:  color of nodes
        edge_color:  color of edges

    Returns:
        None
    '''

    nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
                     edge_color=node_color,
                     node_color=edge_color,
                     with_labels=True,
                     node_size=2,
                     arrowsize=10.0,)
