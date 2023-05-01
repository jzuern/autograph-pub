import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString
import osmnx.distance
import math
from collections import defaultdict
import json
from PIL import Image
Image.MAX_IMAGE_PIXELS = pow(2, 35).__int__()
from scipy.spatial.distance import cdist
import cv2
import sknw
from aggregation.utils import smooth_trajectory, resample_trajectory
from skimage.morphology import skeletonize


def roundify_skeleton_graph(skeleton_graph: nx.DiGraph):

    '''
    The skeleton graph has edges defined with a point list. We add the new points and edges to the graph to roundify it
    :param skeleton_graph:
    :return:
    '''

    skeleton_graph_ = skeleton_graph.copy()

    for edge in skeleton_graph.edges:
        pointlist = skeleton_graph.edges[edge]['pts']
        pointlist = resample_trajectory(pointlist, dist=10)


        # check whether beginning or end of pointlist is closer to edge[0]
        edge_0_pos = skeleton_graph.nodes[edge[0]]['pos']
        if cdist(np.array([edge_0_pos]), np.array([pointlist[0]]))[0][0] > cdist(np.array([edge_0_pos]), np.array([pointlist[-1]]))[0][0]:
            pointlist = pointlist[::-1]

        if len(pointlist) < 3:
            continue


        # add new points and edges to the graph
        for i in range(0, len(pointlist) - 1):
            if i == 0:
                point = (int(pointlist[i][0]), int(pointlist[i][1]))
                skeleton_graph_.add_node(point, pos=pointlist[i])
                skeleton_graph_.add_edge(edge[0], point)
            if i == len(pointlist) - 2:
                point = (int(pointlist[i][0]), int(pointlist[i][1]))
                skeleton_graph_.add_node(point, pos=pointlist[i])
                skeleton_graph_.add_edge(point, edge[1])
            else:
                point0 = (int(pointlist[i][0]), int(pointlist[i][1]))
                skeleton_graph_.add_node(point0, pos=pointlist[i])
                point1 = (int(pointlist[i + 1][0]), int(pointlist[i + 1][1]))
                skeleton_graph_.add_node(point1, pos=pointlist[i + 1])

                skeleton_graph_.add_edge(point0, point1)

        skeleton_graph_.remove_edge(edge[0], edge[1])


    return skeleton_graph_






def skeletonize_prediction(pred_succ, threshold=0.5):
    # first, convert to binary
    pred_succ_thrshld = (pred_succ > threshold).astype(np.uint8)

    #cv2.imshow("pred_succ_thrshld", pred_succ_thrshld * 255)

    # then, skeletonize
    skeleton = skeletonize(pred_succ_thrshld)

    # cut away top and sides by N pixels
    N = 30
    skeleton[:N, :] = 0
    skeleton[:, :N] = 0
    skeleton[:, -N:] = 0

    return skeleton



def skeleton_to_graph(skeleton):
    """Convert skeleton to graph"""

    # build graph from skeleton
    graph = sknw.build_sknw(skeleton)

    # smooth edges
    for (s, e) in graph.edges():
        graph[s][e]['pts'] = smooth_trajectory(graph[s][e]['pts'])

    # add node positions
    node_positions = np.array([graph.nodes[n]['o'] for n in graph.nodes()])

    if len(node_positions) == 0:
        return graph

    # start_node = np.argmin(np.linalg.norm(node_positions - np.array([128, 255]), axis=1))
    start_node = np.argmin(np.linalg.norm(node_positions - np.array([255, 128]), axis=1))

    # remove all edges that are not connected to closest node
    connected = nx.node_connected_component(graph, start_node)  # nodes of component that contains node 0
    graph.remove_nodes_from([n for n in graph if n not in connected])

    graph = graph.to_directed()

    # adjust coordinates order
    # add node positions to graph
    for node in graph.nodes():
        graph.nodes[node]['pos'] = graph.nodes[node]['o'][::-1]
    for edge in graph.edges():
        graph.edges[edge]['pts'] = graph.edges[edge]['pts'][:, ::-1]

    # now let every edge face away from the closest node
    edge_order = nx.dfs_edges(graph, source=start_node, depth_limit=None)
    edge_order = list(edge_order)

    for i, (s, e) in enumerate(edge_order):
        if graph.has_edge(s, e):
            if graph.has_edge(e, s):
                graph[s][e]['pts'] = np.flip(graph[e][s]['pts'], axis=0)
                graph.remove_edge(e, s)

    return graph



def colorize(mask):

    # normalize mask
    mask = np.log(mask + 1e-8)
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)

    mask = (mask * 255.).astype(np.uint8)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_MAGMA)
    return mask




def transform_keypoint_to_world(keypoint, ego_x_y_yaw):
    """
    Transform keypoints from image coordinates to world coordinates.
    :param keypoints: list of keypoints in image coordinates
    :param ego_x_y_yaw: ego pose in world coordinates
    :return: list of keypoints in world coordinates
    """
    [x, y, yaw] = ego_x_y_yaw.squeeze().tolist()
    yaw = yaw + np.pi / 2

    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw), np.cos(yaw)]])

    T = np.eye(3)
    T[0:2, 2] = [x, y]
    T[0:2, 0:2] = R

    keypoint = np.array(keypoint).astype(np.float32)
    keypoint -= np.array([128, 256])
    keypoint = np.append(keypoint, 1)
    keypoint = np.dot(T, keypoint)  # Transform keypoint
    keypoint = keypoint[0:2]
    # keypoint -= np.array([roi_area_xxyy[0], roi_area_xxyy[2]]) # Offset x and y to account for to ROI

    return keypoint


def get_closest_agg_node_from_pred_start(G_incoming, G_existing, ego_agg_min_dist=30):
    """
    Queries current ego position from G_incoming and provides closest node in G_existing.
    """
    new_in_degree = dict(G_incoming.in_degree(list(G_incoming.nodes())))

    # Check if dict is empty
    if len(new_in_degree) == 0:
        print("G_incoming is empty. No splits to validate.")
        return None

    # Get key of new_in_degree dict with minimum value
    min_in_degree_key = min(new_in_degree, key=new_in_degree.get)
    assert new_in_degree[min_in_degree_key] == 0, "Minimum in degree key is not 0. Something is wrong."
    ego_pos = G_incoming.nodes[min_in_degree_key]['pos']

    # Get the closest node in G_existing with respect to ego_pos
    closest_node = None
    closest_node_dist = np.inf
    for n in G_existing.nodes():
        dist = np.linalg.norm(ego_pos - G_existing.nodes[n]['pos'])
        if dist < closest_node_dist:
            closest_node_dist = dist
            closest_node = n
    if closest_node_dist < ego_agg_min_dist:
        agg_ego_node = closest_node
    else:
        agg_ego_node = None

    return agg_ego_node

def get_parallel_paths(G, cutoff=6):
    return [list(nx.all_simple_paths(G, i, j, cutoff=cutoff)) for i in G.nodes() for j in G.nodes() if i != j and nx.has_path(G, i, j)]


def remove_parallel_paths(G_incoming, G_existing, visited_edges, ego_agg_min_dist=30, fixed_eval_lag=1):
    agg_ego_node = get_closest_agg_node_from_pred_start(G_incoming, G_existing, ego_agg_min_dist=ego_agg_min_dist)

    visited_nodes = list()
    for e in visited_edges:
        visited_nodes += [e[0], e[1]]
    visited_nodes = set(visited_nodes)

    agg_out_degree = dict(G_existing.out_degree(list(G_existing.nodes())))
    if agg_ego_node is None:
        return G_existing
    nodes_to_be_removed = list()
    parallel_paths = [connection for connection in get_parallel_paths(G_existing, cutoff=8) if len(connection) > 0]

    # sort parallel_paths by length of first element (shortest first: get just the loops without appendices)
    parallel_paths.sort(key=lambda x: len(x[0]))
    competing_paths = defaultdict(list)
    competing_weights = defaultdict(list)
    for paths in parallel_paths:
        start_node, end_node = paths[0][0], paths[0][-1]
        rev_G_existing = G_existing.reverse()
        try:
            length = nx.shortest_path_length(rev_G_existing, source=agg_ego_node, target=end_node)
            # eval-lag is measured wrt end_node
            if length > fixed_eval_lag:
                # check if path starting node and end node fulfill degree criteria
                branch_factor = G_existing.out_degree(start_node)
                merge_factor = G_existing.in_degree(end_node)
                if branch_factor > 1 and merge_factor > 1:
                    competing_paths[(start_node, end_node)] = paths
                    competing_weights[(start_node, end_node)] = [np.sum([G_existing.nodes[k]['weight'] for k in path]) for path in
                                                           paths]
                    max_weight_path = np.argmax(competing_weights[(start_node, end_node)])
                    for i, path in enumerate(competing_paths[(start_node, end_node)]):
                        if i != max_weight_path:
                            nodes_to_be_removed += path[1:-1]
        except nx.NetworkXNoPath:
            pass

    edges_to_be_removed = list()
    # Remove parallel edges of length 1
    for edge in G_existing.edges():
        if edge not in visited_edges:
            if edge[0] in visited_nodes and edge[1] in visited_nodes:
                edges_to_be_removed += [edge]

    for e in edges_to_be_removed:
        G_existing.remove_edge(e[0], e[1])


    for t in nodes_to_be_removed:
        if G_existing.has_node(t) and t not in visited_nodes:
            G_existing.remove_node(t)
    # Remove all remaining isolated nodes
    G_existing.remove_nodes_from(list(nx.isolates(G_existing)))
    return G_existing


def remove_unvalidated_splits_merges(G_incoming, G_existing, visited_edges, ego_agg_min_dist=30, fixed_eval_lag=1, split_weight_thresh=6):
    """
    Remove unvalidated splits and merges from G_existing
    Only the non-highest weighted predecessor trees are removed when deleting merges.
    """
    agg_ego_node = get_closest_agg_node_from_pred_start(G_incoming, G_existing, ego_agg_min_dist=ego_agg_min_dist)
    if agg_ego_node is None:
        return G_existing
    agg_out_degree = dict(G_existing.out_degree(list(G_existing.nodes())))

    nodes_to_be_removed = list()

    # Remove all splits that do not satisfy a certain length
    edges_to_be_removed_first = list()
    for n in G_existing.nodes():
        if agg_out_degree[n] >= 2:
            if G_existing.in_degree(n) == 0:
                edges_to_be_removed_first += [(n, k) for k in G_existing.successors(n)]
            # Gather all untraversed edges leaving this node
            edges_starting_at_n = [e for e in G_existing.out_edges(n)]
            if len(edges_starting_at_n) > 0:
                for unvisited_edge in edges_starting_at_n:
                    edge_tree = nx.dfs_tree(G_existing, source=unvisited_edge[1], depth_limit=5)
                    edge_tree_nodes = list(edge_tree)
                    edge_tree_weight = np.sum([1 for k in edge_tree_nodes])
                    if edge_tree_weight <= 1:
                        for e in edge_tree.edges():
                            edges_to_be_removed_first += [e]

    for e in edges_to_be_removed_first:
        G_existing.remove_edge(e[0], e[1])

    # Remove all merges that do not satisfy a certain weight
    for n in G_existing.nodes():
        if agg_out_degree[n] >= 2:
            rev_G_existing = G_existing.reverse()
            try:
                length = nx.shortest_path_length(rev_G_existing, source=agg_ego_node, target=n)
                if length > fixed_eval_lag:

                    # REMOVE UNVALIDATED SPLITS
                    successors = list(G_existing.successors(n))
                    successor_trees = [list(nx.dfs_tree(G_existing, source=s, depth_limit=10)) for s in successors]
                    successor_tree_weights = [np.sum([G_existing.nodes[k]['weight'] for k in t]) for t in successor_trees]

                    # Delete all successor trees with weights < 2
                    for i, s in enumerate(successors):
                        if successor_tree_weights[i] <= split_weight_thresh or len(successor_trees[i]) <= 2:
                            for t in successor_trees[i]:
                                nodes_to_be_removed.append(t)


                    # REMOVE UNVALIDATED MERGES EXCEPT HIGHEST WEIGHTED PREDECESSOR TREE
                    predecessors = list(G_existing.predecessors(n))
                    predecessor_trees = [
                        list(nx.dfs_tree(rev_G_existing, source=p, depth_limit=4)) for p in
                        predecessors]
                    predecessor_tree_weights = [np.sum([G_existing.nodes[k]['weight'] for k in t]) for t
                                              in predecessor_trees]
                    if len(predecessor_tree_weights):
                        predecessor_max_weight_tree_idx = np.argmax(predecessor_tree_weights)

                    # Delete all successor trees with weights < 2
                    for i, s in enumerate(predecessors):
                        if i == predecessor_max_weight_tree_idx:
                            continue
                        if predecessor_tree_weights[i] <= split_weight_thresh or len(
                                predecessor_trees[i]) <= 2:
                            for t in predecessor_trees[i]:
                                nodes_to_be_removed.append(t)

            except nx.NetworkXNoPath as e:
                pass


    for t in nodes_to_be_removed:
        if G_existing.has_node(t):
            G_existing.remove_node(t)

    return G_existing

def mean_angle_abs_diff(x, y):
    period = 2 * np.pi
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]
    return np.abs(diff)


# def naive_aggregate(G_agg, G_new, threshold_px=20, closest_node_dist_thresh=30):
#
#     # Maps from agg nodes to new nodes
#     merging_map = defaultdict(list)
#
#     # Add aggregation weight to new predictions
#     for n in G_new.nodes():
#         G_new.nodes[n]['weight'] = 1.0
#
#     # Add edge angles to new graph
#     for e in G_new.edges():
#         G_new.edges[e]['angle'] = np.arctan2(G_new.nodes[e[1]]['pos'][1] - G_new.nodes[e[0]]['pos'][1],
#                                              G_new.nodes[e[1]]['pos'][0] - G_new.nodes[e[0]]['pos'][0])
#
#     # Get mean of angles of edges connected to each node in G_new
#     for n in G_new.nodes():
#         edge_angles_pred = [nx.get_edge_attributes(G_new, 'angle')[(x, n)] for x in G_new.predecessors(n)]
#         edge_angles_succ = [nx.get_edge_attributes(G_new, 'angle')[(n, x)] for x in G_new.successors(n)]
#         edge_angles = edge_angles_pred + edge_angles_succ
#         edge_angles_sin = [np.sin(angle) for angle in edge_angles]
#         edge_angles_cos = [np.cos(angle) for angle in edge_angles]
#         mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
#         if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
#             mean_angle = 0
#         G_new.nodes[n]['mean_angle'] = mean_angle
#
#     # What if G_agg is empty? Then just return G_new, because it's the first graph and will be used as G_agg in next iteration
#     if len(G_agg.nodes) == 0:
#         return G_new.copy(), merging_map
#
#     # Assign angle attribute on edges of G_agg and G_new
#     for e in G_agg.edges():
#         G_agg.edges[e]['angle'] = np.arctan2(G_agg.nodes[e[1]]['pos'][1] - G_agg.nodes[e[0]]['pos'][1],
#                                              G_agg.nodes[e[1]]['pos'][0] - G_agg.nodes[e[0]]['pos'][0])
#
#     # Get mean of angles of edges connected to each node in G_agg
#     for n in G_agg.nodes():
#         edge_angles_pred = [nx.get_edge_attributes(G_agg, 'angle')[(x, n)] for x in G_agg.predecessors(n)]
#         edge_angles_succ = [nx.get_edge_attributes(G_agg, 'angle')[(n, x)] for x in G_agg.successors(n)]
#         edge_angles = edge_angles_pred + edge_angles_succ
#         edge_angles_sin = [np.sin(angle) for angle in edge_angles]
#         edge_angles_cos = [np.cos(angle) for angle in edge_angles]
#         mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
#         if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
#             mean_angle = 0
#         G_agg.nodes[n]['mean_angle'] = mean_angle
#
#     # Get node name map
#     node_names_agg = list(G_agg.nodes())
#     node_names_new = list(G_new.nodes())
#
#     # Get pairwise distance between nodes in G_agg and G_new
#     node_pos_agg = np.array([G_agg.nodes[n]['pos'] for n in G_agg.nodes]).reshape(-1, 2)
#     node_pos_new = np.array([G_new.nodes[n]['pos'] for n in G_new.nodes]).reshape(-1, 2)
#     node_distances = cdist(node_pos_agg, node_pos_new, metric='euclidean') # i: agg, j: new
#
#     # Produce a pairwise thresholding that allows the construction of ROIs in terms of Euclidean distance
#     position_criterium = node_distances < threshold_px
#
#     closest_agg_nodes = defaultdict()
#     # Loop through all new nodes (columns indexed with j)
#     for j in range(position_criterium.shape[1]):
#         # Loop through all close agg-nodes and construct the j-specific local agg graph
#         agg_j_multilines = list()
#
#         # Get all agg-nodes that are close to new node j
#         # Use orthogonal linear coordinates system to avoid problems arising from OSMnx distance calculation
#         G_agg_j = nx.MultiDiGraph(crs="EPSG:3857")
#         for i in range(position_criterium.shape[0]):
#             if position_criterium[i, j]: # check if agg node i is close enough to new node j
#                 for e in G_agg.edges(node_names_agg[i]):
#                     # Add edge to local agg graph
#
#                     G_agg_j.add_node(str(e[0]), x=G_agg.nodes[e[0]]['pos'][0], y=G_agg.nodes[e[0]]['pos'][1])
#                     G_agg_j.add_node(str(e[1]), x=G_agg.nodes[e[1]]['pos'][0], y=G_agg.nodes[e[1]]['pos'][1])
#                     G_agg_j.add_edge(str(e[0]), str(e[1]))
#                     agg_j_multilines.append(((G_agg.nodes[e[0]]['pos'][0], G_agg.nodes[e[0]]['pos'][1]),
#                                            (G_agg.nodes[e[1]]['pos'][0], G_agg.nodes[e[1]]['pos'][1])))
#         agg_j_shapely = MultiLineString(agg_j_multilines)
#         # Find closest edge and closest_node in agg-graph to new node j
#         if len(list(G_agg_j.edges)) > 0:
#             closest_node = osmnx.distance.nearest_nodes(G_agg_j,
#                                                            float(G_new.nodes[node_names_new[j]]['pos'][0]),
#                                                            float(G_new.nodes[node_names_new[j]]['pos'][1]),
#                                                            return_dist=False)
#             closest_node = eval(closest_node)
#             closest_node_dist = np.linalg.norm(np.array(G_agg.nodes[closest_node]['pos']) - G_new.nodes[node_names_new[j]]['pos'])
#
#             if closest_node_dist < closest_node_dist_thresh:
#                 closest_agg_nodes[node_names_new[j]] = closest_node
#                 updtd_closest_node_pos = (G_agg.nodes[closest_node]['weight'] * np.array(G_agg.nodes[closest_node]['pos']) \
#                                          + np.array(G_new.nodes[node_names_new[j]]['pos'])) / (G_agg.nodes[closest_node]['weight'] + 1)
#
#                 # Check if the updated node is not NaN
#                 if not math.isnan(updtd_closest_node_pos[0] * updtd_closest_node_pos[1]):
#                     G_agg.nodes[closest_node]['pos'][0], G_agg.nodes[closest_node]['pos'][1] = updtd_closest_node_pos[0], updtd_closest_node_pos[1]
#
#                 # Record merging weights
#                 G_agg.nodes[closest_node]['weight'] += 1
#
#                 merging_map[closest_node].append(node_names_new[j])
#
#     # What happens to all other nodes in G_new? Add them to G_agg
#     mapped_new_nodes = [*merging_map.values()]
#     mapped_new_nodes = [item for sublist in mapped_new_nodes for item in sublist]
#     for n in G_new.nodes():
#         if n not in mapped_new_nodes:
#             G_agg.add_node(n, pos=G_new.nodes[n]['pos'], weight=G_new.nodes[n]['weight'], score=G_new.nodes[n]['score'])
#
#     for e in G_new.edges():
#         n = e[0]
#         m = e[1]
#         angle = np.arctan2(G_new.nodes[m]['pos'][1] - G_new.nodes[n]['pos'][1],
#                            G_new.nodes[m]['pos'][0] - G_new.nodes[n]['pos'][0])
#
#         # Add completely new edges
#         if n not in mapped_new_nodes and m not in mapped_new_nodes:
#             G_agg.add_edge(n, m, angle=G_new.edges[e]['angle'])
#
#         # Add leading edges
#         if n in mapped_new_nodes and m not in mapped_new_nodes:
#             angle = np.arctan2(G_new.nodes[m]['pos'][1] - G_agg.nodes[closest_agg_nodes[n]]['pos'][1],
#                                G_new.nodes[m]['pos'][0] - G_agg.nodes[closest_agg_nodes[n]]['pos'][0])
#             G_agg.add_edge(closest_agg_nodes[n], m, angle=angle)
#
#         # Add trailing edges
#         if n not in mapped_new_nodes and m in mapped_new_nodes:
#             angle = np.arctan2(G_agg.nodes[closest_agg_nodes[m]]['pos'][1] - G_new.nodes[n]['pos'][1],
#                                G_agg.nodes[closest_agg_nodes[m]]['pos'][0] - G_new.nodes[n]['pos'][0])
#             G_agg.add_edge(n, closest_agg_nodes[m], angle=angle)
#     return G_agg, merging_map


def aggregate(G_agg, G_new, visited_edges, threshold_px=20, threshold_rad=0.1, closest_lat_thresh=30, w_decay=False, remove=False):

    # Maps from agg nodes to new nodes
    merging_map = defaultdict(list)

    # Add aggregation weight to new predictions
    if w_decay:
        new_in_degree = dict(G_new.in_degree(list(G_new.nodes())))
        # Check if dict is empty
        if len(new_in_degree) > 0:
            # Get key of new_in_degree dict with minimum value
            new_ego_root_node = min(new_in_degree, key=new_in_degree.get)
            shortest_paths_from_root = nx.shortest_path_length(G_new, new_ego_root_node)
            for n in G_new.nodes():
                G_new.nodes[n]['weight'] = 1 - 0.05 * shortest_paths_from_root[n]
    else:
        for n in G_new.nodes():
            G_new.nodes[n]['weight'] = 1.0

    # Add edge angles to new graph
    for e in G_new.edges():
        G_new.edges[e]['angle'] = np.arctan2(G_new.nodes[e[1]]['pos'][1] - G_new.nodes[e[0]]['pos'][1],
                                             G_new.nodes[e[1]]['pos'][0] - G_new.nodes[e[0]]['pos'][0])

    # Get mean of angles of edges connected to each node in G_agg
    for n in G_new.nodes():
        edge_angles_pred = [nx.get_edge_attributes(G_new, 'angle')[(x, n)] for x in G_new.predecessors(n)]
        edge_angles_succ = [nx.get_edge_attributes(G_new, 'angle')[(n, x)] for x in G_new.successors(n)]
        edge_angles = edge_angles_pred + edge_angles_succ
        edge_angles_sin = [np.sin(angle) for angle in edge_angles]
        edge_angles_cos = [np.cos(angle) for angle in edge_angles]
        mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
        if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
            mean_angle = 0
        G_new.nodes[n]['mean_angle'] = mean_angle

    # What if G_agg is empty? Then just return G_new, because it's the first graph and will be used as G_agg in next iteration
    if len(G_agg.nodes) == 0:
        return G_new.copy(), merging_map

    if remove:
        # Remove splits as soon as traveled past them by fixed_eval_lag
        G_agg = remove_unvalidated_splits_merges(G_new, G_agg, visited_edges, ego_agg_min_dist=30, fixed_eval_lag=1, split_weight_thresh=2)
        #G_agg = remove_parallel_paths(G_new, G_agg, visited_edges, ego_agg_min_dist=30, fixed_eval_lag=1)

    # Assign angle attribute on edges of G_agg and G_new
    for e in G_agg.edges():
        G_agg.edges[e]['angle'] = np.arctan2(G_agg.nodes[e[1]]['pos'][1] - G_agg.nodes[e[0]]['pos'][1],
                                             G_agg.nodes[e[1]]['pos'][0] - G_agg.nodes[e[0]]['pos'][0])

    # Get mean of angles of edges connected to each node in G_agg
    for n in G_agg.nodes():
        edge_angles_pred = [nx.get_edge_attributes(G_agg, 'angle')[(x, n)] for x in G_agg.predecessors(n)]
        edge_angles_succ = [nx.get_edge_attributes(G_agg, 'angle')[(n, x)] for x in G_agg.successors(n)]
        edge_angles = edge_angles_pred + edge_angles_succ
        edge_angles_sin = [np.sin(angle) for angle in edge_angles]
        edge_angles_cos = [np.cos(angle) for angle in edge_angles]
        mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
        if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
            mean_angle = 0
        G_agg.nodes[n]['mean_angle'] = mean_angle

    # Get node name map
    node_names_agg = list(G_agg.nodes())
    node_names_new = list(G_new.nodes())

    # Get pairwise distance between nodes in G_agg and G_new
    node_pos_agg = np.array([G_agg.nodes[n]['pos'] for n in G_agg.nodes]).reshape(-1, 2)
    node_pos_new = np.array([G_new.nodes[n]['pos'] for n in G_new.nodes]).reshape(-1, 2)
    node_distances = cdist(node_pos_agg, node_pos_new, metric='euclidean') # i: agg, j: new

    # Get pairwise angle difference between nodes in G_agg and G_new
    node_mean_ang_agg = np.array([G_agg.nodes[n]['mean_angle'] for n in G_agg.nodes]).reshape(-1, 1)
    node_mean_ang_new = np.array([G_new.nodes[n]['mean_angle'] for n in G_new.nodes]).reshape(-1, 1)
    node_mean_ang_distances = cdist(node_mean_ang_agg, node_mean_ang_new, lambda u, v: mean_angle_abs_diff(u, v))

    # Produce a pairwise thresholding that allows the construction of ROIs in terms of Euclidean distance
    # and angle difference
    position_criterium = node_distances < threshold_px
    angle_criterium = node_mean_ang_distances < threshold_rad
    criterium = position_criterium & angle_criterium

    closest_agg_nodes = defaultdict()

    # Loop through all new nodes (columns indexed with j)
    for j in range(criterium.shape[1]):
        # Loop through all close agg-nodes and construct the j-specific local agg graph
        agg_j_multilines = list()

        # Get all agg-nodes that are close to new node j
        # Use orthogonal linear coordinates system to avoid problems arising from OSMnx distance calculation
        G_agg_j = nx.MultiDiGraph(crs="EPSG:3857")
        for i in range(criterium.shape[0]):
            if criterium[i, j]: # check if agg node i is close enough to new node j
                for e in G_agg.edges(node_names_agg[i]):
                    # Add edge to local agg graph
                    G_agg_j.add_node(str(e[0]), x=G_agg.nodes[e[0]]['pos'][0], y=G_agg.nodes[e[0]]['pos'][1])
                    G_agg_j.add_node(str(e[1]), x=G_agg.nodes[e[1]]['pos'][0], y=G_agg.nodes[e[1]]['pos'][1])
                    G_agg_j.add_edge(str(e[0]), str(e[1]))
                    agg_j_multilines.append(((G_agg.nodes[e[0]]['pos'][0], G_agg.nodes[e[0]]['pos'][1]),
                                           (G_agg.nodes[e[1]]['pos'][0], G_agg.nodes[e[1]]['pos'][1])))
        agg_j_shapely = MultiLineString(agg_j_multilines)
        # Find closest edge and closest_node in agg-graph to new node j
        if len(list(G_agg_j.edges)) > 0:
            closest_edge, closest_lat_dist = osmnx.distance.nearest_edges(G_agg_j,
                                                        float(G_new.nodes[node_names_new[j]]['pos'][0]),
                                                        float(G_new.nodes[node_names_new[j]]['pos'][1]),
                                                        return_dist=True)
            closest_node = osmnx.distance.nearest_nodes(G_agg_j,
                                                           float(G_new.nodes[node_names_new[j]]['pos'][0]),
                                                           float(G_new.nodes[node_names_new[j]]['pos'][1]),
                                                           return_dist=False)
            closest_node = eval(closest_node)
            closest_node_dist = np.linalg.norm(np.array(G_agg.nodes[closest_node]['pos']) - G_new.nodes[node_names_new[j]]['pos'])

            if closest_lat_dist < closest_lat_thresh:
                closest_i, closest_j = eval(closest_edge[0]), eval(closest_edge[1])

                # assign second-closest to closest_node not closest_i
                if closest_i == closest_node:
                    sec_closest_node = closest_j
                else:
                    sec_closest_node = closest_i

                closest_agg_nodes[node_names_new[j]] = closest_node

                sec_closest_node_dist = np.linalg.norm(np.array(G_agg.nodes[sec_closest_node]['pos']) - G_new.nodes[node_names_new[j]]['pos'])

                closest_node_dist_x = G_agg.nodes[closest_node]['pos'][0] - G_new.nodes[node_names_new[j]]['pos'][0]
                closest_node_dist_y = G_agg.nodes[closest_node]['pos'][1] - G_new.nodes[node_names_new[j]]['pos'][1]

                alpha = np.arccos(closest_lat_dist/ closest_node_dist)
                beta = np.arctan(closest_node_dist_y / closest_node_dist_x)
                gamma = np.pi/2 - alpha - beta

                sec_alpha = np.arccos(closest_lat_dist / sec_closest_node_dist)

                closest_long_dist = closest_node_dist * np.sin(alpha)
                sec_closest_long_dist = sec_closest_node_dist * np.sin(sec_alpha)

                curr_new_node = np.array(G_new.nodes[node_names_new[j]]['pos'])
                virtual_closest_lat_node = curr_new_node + closest_long_dist * np.array([-np.cos(gamma), np.sin(gamma)])
                virtual_sec_closest_lat_node = curr_new_node + sec_closest_long_dist * np.array([np.cos(gamma), -np.sin(gamma)])

                omega_closest = 1 - closest_node_dist / (closest_node_dist + sec_closest_node_dist)
                omega_sec_closest = 1 - sec_closest_node_dist / (closest_node_dist + sec_closest_node_dist)

                # Calculating the node weights for aggregation
                closest_agg_node_weight = G_agg.nodes[closest_node]['weight']/(G_agg.nodes[closest_node]['weight'] + 1)
                closest_new_node_weight = omega_closest * 1 / (G_agg.nodes[closest_node]['weight'] + 1)
                # Normalization of closest weights
                closest_weights_sum = closest_agg_node_weight + closest_new_node_weight
                closest_agg_node_weight = closest_agg_node_weight / closest_weights_sum
                closest_new_node_weight = closest_new_node_weight / closest_weights_sum

                sec_closest_agg_node_weight = G_agg.nodes[sec_closest_node]['weight'] / (G_agg.nodes[sec_closest_node]['weight'] + 1)
                sec_closest_new_node_weight = omega_sec_closest * 1 / (G_agg.nodes[sec_closest_node]['weight'] + 1)
                # Normalization of sec-closest weights
                sec_closest_weights_sum = sec_closest_agg_node_weight + sec_closest_new_node_weight
                sec_closest_agg_node_weight = sec_closest_agg_node_weight / sec_closest_weights_sum
                sec_closest_new_node_weight = sec_closest_new_node_weight / sec_closest_weights_sum

                updtd_closest_node_pos = closest_agg_node_weight * np.array(G_agg.nodes[closest_node]['pos']) + closest_new_node_weight * np.array(virtual_closest_lat_node)
                updtd_sec_closest_node_pos = sec_closest_agg_node_weight * np.array(G_agg.nodes[sec_closest_node]['pos']) + sec_closest_new_node_weight * np.array(virtual_sec_closest_lat_node)

                # Check if the updated node is not NaN
                if not math.isnan(updtd_closest_node_pos[0] * updtd_closest_node_pos[1]):
                    G_agg.nodes[closest_node]['pos'][0], G_agg.nodes[closest_node]['pos'][1] = updtd_closest_node_pos[0], updtd_closest_node_pos[1]
                if not math.isnan(updtd_sec_closest_node_pos[0] * updtd_sec_closest_node_pos[1]):
                    G_agg.nodes[sec_closest_node]['pos'][0], G_agg.nodes[sec_closest_node]['pos'][1] = updtd_sec_closest_node_pos[0], updtd_sec_closest_node_pos[1]

                # Record merging weights
                G_agg.nodes[closest_node]['weight'] += 1
                G_agg.nodes[sec_closest_node]['weight'] += 1

                merging_map[closest_node].append(node_names_new[j])
                merging_map[sec_closest_node].append(node_names_new[j])


    # What happens to all other nodes in G_new? Add them to G_agg
    mapped_new_nodes = [*merging_map.values()]
    mapped_new_nodes = [item for sublist in mapped_new_nodes for item in sublist]
    for n in G_new.nodes():
        if n not in mapped_new_nodes:
            G_agg.add_node(n,
                           pos=G_new.nodes[n]['pos'],
                           weight=G_new.nodes[n]['weight'],
                           score=G_new.nodes[n]['score'])


    for e in G_new.edges():
        n = e[0]
        m = e[1]

        angle = np.arctan2(G_new.nodes[m]['pos'][1] - G_new.nodes[n]['pos'][1],
                           G_new.nodes[m]['pos'][0] - G_new.nodes[n]['pos'][0])

        # Add completely new edges
        if n not in mapped_new_nodes and m not in mapped_new_nodes:
            G_agg.add_edge(n, m, angle=G_new.edges[e]['angle'])

        # Add leading edges
        if n in mapped_new_nodes and m not in mapped_new_nodes:
            angle = np.arctan2(G_new.nodes[m]['pos'][1] - G_agg.nodes[closest_agg_nodes[n]]['pos'][1],
                               G_new.nodes[m]['pos'][0] - G_agg.nodes[closest_agg_nodes[n]]['pos'][0])
            G_agg.add_edge(closest_agg_nodes[n], m, angle=angle)

        # Add trailing edges
        if n not in mapped_new_nodes and m in mapped_new_nodes:
            angle = np.arctan2(G_agg.nodes[closest_agg_nodes[m]]['pos'][1] - G_new.nodes[n]['pos'][1],
                               G_agg.nodes[closest_agg_nodes[m]]['pos'][0] - G_new.nodes[n]['pos'][0])
            G_agg.add_edge(n, closest_agg_nodes[m], angle=angle)
    return G_agg, merging_map


# def fuse(G_agg, threshold_px=10, threshold_rad=0.2):
#
#     # Maps from agg nodes to new nodes
#     merging_map = defaultdict(list)
#
#     for e in G_agg.edges():
#         G_agg.edges[e]['angle'] = np.arctan2(G_agg.nodes[e[1]]['pos'][1] - G_agg.nodes[e[0]]['pos'][1],
#                                              G_agg.nodes[e[1]]['pos'][0] - G_agg.nodes[e[0]]['pos'][0])
#
#     # Get mean of angles of edges connected to each node in G_agg
#     for n in G_agg.nodes():
#         edge_angles_pred = [nx.get_edge_attributes(G_agg, 'angle')[(x, n)] for x in G_agg.predecessors(n)]
#         edge_angles_succ = [nx.get_edge_attributes(G_agg, 'angle')[(n, x)] for x in G_agg.successors(n)]
#         edge_angles = edge_angles_pred + edge_angles_succ
#         edge_angles_sin = [np.sin(angle) for angle in edge_angles]
#         edge_angles_cos = [np.cos(angle) for angle in edge_angles]
#         mean_angle = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
#         if len(edge_angles_pred) == 0 and len(edge_angles_succ) == 0:
#             mean_angle = 0
#         G_agg.nodes[n]['mean_angle'] = mean_angle
#
#     # Get node name map
#     node_names_agg = list(G_agg.nodes())
#
#     # Get pairwise distance between nodes in G_agg and G_new
#     node_pos_agg = np.array([G_agg.nodes[n]['pos'] for n in G_agg.nodes]).reshape(-1, 2)
#     node_distances = cdist(node_pos_agg, node_pos_agg, metric='euclidean') # i: agg, j: new
#
#     # Get pairwise angle difference between nodes in G_agg and G_new
#     node_mean_ang_agg = np.array([G_agg.nodes[n]['mean_angle'] for n in G_agg.nodes]).reshape(-1, 1)
#     node_mean_ang_distances = cdist(node_mean_ang_agg, node_mean_ang_agg, lambda u, v: mean_angle_abs_diff(u, v))
#
#     # Produce a pairwise thresholding that allows the construction of ROIs in terms of Euclidean distance
#     # and angle difference
#     position_criterium = node_distances < threshold_px
#     angle_criterium = node_mean_ang_distances < threshold_rad
#     criterium = position_criterium & angle_criterium
#
#     closest_agg_nodes = defaultdict()
#
#     assigned = dict()
#     # Loop through all new nodes (columns indexed with j)
#     for j in range(criterium.shape[1]):
#         # Loop through all close agg-nodes and construct the j-specific local agg graph
#         for i in range(criterium.shape[0]):
#             if criterium[i, j]: # check if agg node i is close enough to new node j
#                 if j not in assigned.values():
#                     assigned[i] = j
#                     merging_map[node_names_agg[i]].append(node_names_agg[j])
#
#     for i, tobemerged in merging_map.items():
#         for j in tobemerged:
#             if i != j:
#                 G_agg = nx.contracted_nodes(G_agg, i, j, self_loops=False)
#
#     for i in G_agg.nodes():
#         if "contraction" in G_agg.nodes[i].keys():
#             pos_0 = np.mean([k['pos'][0] for k in G_agg.nodes[i]['contraction'].values()], axis=0)
#             pos_1 = np.mean([k['pos'][1] for k in G_agg.nodes[i]['contraction'].values()], axis=0)
#
#             G_agg.nodes[i]['pos'] = np.array([pos_0, pos_1])
#             G_agg.nodes[i]['weight'] = np.sum([k['weight'] for k in G_agg.nodes[i]['contraction'].values()], axis=0)
#             G_agg.nodes[i]['score'] = np.mean([k['score'] for k in G_agg.nodes[i]['contraction'].values()], axis=0)
#
#             edge_angles_sin = [np.sin(k['mean_angle']) for k in G_agg.nodes[i]['contraction'].values()]
#             edge_angles_cos = [np.cos(k['mean_angle']) for k in G_agg.nodes[i]['contraction'].values()]
#             G_agg.nodes[i]['mean_angle'] = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))
#
#     return G_agg, merging_map


# def plot_graphs_online(G_pred_list, G_pred_agg_before, G_pred_new, assoc, G_pred_agg_updtd, satellite_image):
#
#     fig, axarr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(32, 8))
#
#     axarr[0].set_title('All pred graph edges')
#     axarr[1].set_title('Curr Agg graph (red) and new pred graph (blue)')
#     axarr[2].set_title('Updated aggregated graph (red), old agg graph (magenta)')
#     axarr[3].set_title('Found associations')
#     axarr[0].imshow(satellite_image)
#     axarr[1].imshow(satellite_image)
#     axarr[2].imshow(satellite_image)
#     axarr[3].imshow(satellite_image)
#
#     # [0] Plot all pred graph edges
#     #for G in G_pred_list:
#     #    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=False, ax=axarr[0], node_size=1, edge_color='r',
#     #            node_color='r')
#
#     # [1] Plot curr agg graph (red) and new pred graph (blue)
#     nx.draw(G_pred_agg_before, pos=nx.get_node_attributes(G_pred_agg_before, 'pos'), with_labels=False, ax=axarr[1], node_size=10,
#             edge_color='r', node_color='r')
#     nx.draw(G_pred_new, pos=nx.get_node_attributes(G_pred_new, 'pos'), with_labels=False, ax=axarr[1], node_size=10,
#             edge_color='b', node_color='b')
#
#     # [2] Updated aggregated graph
#     nx.draw(G_pred_agg_before, pos=nx.get_node_attributes(G_pred_agg_before, 'pos'), with_labels=False, ax=axarr[2],
#             node_size=0, edge_color='m', node_color='m', alpha=0.5)
#
#     node_weight_cmap = plt.cm.get_cmap('Reds', len(nx.get_node_attributes(G_pred_agg_updtd, 'weight').values()))
#
#     weight_array = np.array(list(nx.get_node_attributes(G_pred_agg_updtd, "weight").values()))
#     normalized_node_intensities = weight_array/np.max(weight_array)
#     node_colors = [node_weight_cmap(weight) for weight in normalized_node_intensities.tolist()]
#     nx.draw(G_pred_agg_updtd, pos=nx.get_node_attributes(G_pred_agg_updtd, 'pos'), with_labels=False, ax=axarr[2],
#             node_size=10,
#             edge_color='r', node_color=node_colors)
#     sm = plt.cm.ScalarMappable(cmap=node_weight_cmap, norm=plt.Normalize(vmin=np.min(weight_array), vmax=np.max(weight_array)))
#     sm._A = []
#
#     # Attach colorbar to plot without changing plot size
#     divider = make_axes_locatable(axarr[2])
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(sm, cax=cax)
#     #plt.colorbar(sm, ax=axarr[2], orientation="vertical", fraction=0.046, pad=0.04)
#
#     # [3] Plot found associations
#     # assoc maps from existing nodes to new nodes (pixel-indices, not node idcs)
#     nx.draw(G_pred_agg_before, pos=nx.get_node_attributes(G_pred_agg_before, 'pos'), arrowsize=5, with_labels=False,
#             ax=axarr[3], node_size=0,edge_color='r', node_color='w', alpha=0.5)
#     nx.draw(G_pred_new, pos=nx.get_node_attributes(G_pred_new, 'pos'), arrowsize=5, with_labels=False, ax=axarr[3],
#             node_size=0, edge_color='b', node_color='w', alpha=0.5)
#
#     merge_cmap = plt.cm.get_cmap('tab10', len(assoc.keys()))
#
#     for n_idx, n_assoc in enumerate(assoc.items()):
#         anchor, anchored_list = n_assoc
#         c = list(merge_cmap(n_idx))
#
#         # plot anchor node with x y coords
#         axarr[3].scatter(anchor[0], anchor[1], color=c, s=30, marker='o')
#         anchored = np.array(anchored_list)
#         axarr[3].scatter(anchored[:, 0], anchored[:, 1], color=c, s=30, marker='o')
#
#         axarr[3].text(anchor[0], anchor[1], str(n_idx), color=c, fontsize=10)
#         for anchored in anchored_list:
#             axarr[3].text(anchored[0], anchored[1], str(n_idx), color=c, fontsize=10)
#
#     # Get x y limits of G_pred_agg_updtd
#     xmin = np.array([G_pred_agg_updtd.nodes[n]['pos'][0] for n in G_pred_agg_updtd.nodes()]).min()
#     xmax = np.array([G_pred_agg_updtd.nodes[n]['pos'][0] for n in G_pred_agg_updtd.nodes()]).max()
#     ymin = np.array([G_pred_agg_updtd.nodes[n]['pos'][1] for n in G_pred_agg_updtd.nodes()]).min()
#     ymax = np.array([G_pred_agg_updtd.nodes[n]['pos'][1] for n in G_pred_agg_updtd.nodes()]).max()
#
#     axarr[0].set_xlim(left=xmin - 100, right=xmax + 100)
#     axarr[0].set_ylim(top=ymin - 100, bottom=ymax + 100)
#
#     #xarr[0].axis('off')
#     #axarr[1].axis('off')
#
#     plt.tight_layout()
#     plt.show()


# def plot_graphs(G_pred_list, G_pred_agg, G_gt_list, G_gt_agg, ego_positions, satellite_image):
#
#     fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 20))
#     plt.tight_layout()
#
#     axarr[0, 0].set_title('All Pred graphs')
#     axarr[0, 1].set_title('Aggregated Pred graph')
#     axarr[1, 0].set_title('All GT graphs')
#     axarr[1, 1].set_title('Aggregated GT graph')
#     axarr[0, 0].imshow(satellite_image)
#     axarr[0, 1].imshow(satellite_image)
#     axarr[1, 0].imshow(satellite_image)
#     axarr[1, 1].imshow(satellite_image)
#
#     for G in G_pred_list:
#         nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=False, ax=axarr[0, 0], node_size=1, edge_color='r', node_color='r')
#     for G in G_gt_list:
#         nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), with_labels=False, ax=axarr[1, 0], node_size=1, edge_color='g', node_color='g')
#
#     nx.draw(G_pred_agg, pos=nx.get_node_attributes(G_pred_agg, 'pos'), with_labels=False, ax=axarr[0, 1], node_size=10, edge_color='r', node_color='r')
#     nx.draw(G_gt_agg, pos=nx.get_node_attributes(G_gt_agg, 'pos'), with_labels=False, ax=axarr[1, 1], node_size=10, edge_color='g', node_color='g')
#
#     axarr[0, 0].set_xlim(left=200, right=1000)
#     axarr[0, 0].set_ylim(top=150, bottom=1000)
#
#     plt.show()
#
#
# def get_target_data(params, data, split):
#
#     tile_no = int(data.tile_no[0].cpu().detach().numpy())
#     walk_no = int(data.walk_no[0].cpu().detach().numpy())
#     idx = int(data.idx[0].cpu().detach().numpy())
#     city = data.city[0]
#
#
#     json_fname = "{}{}/{}/{}/{:03d}_{:03d}_{:03d}-targets.json".format(params.paths.dataroot, params.paths.rel_dataset, city, split, tile_no, walk_no, idx)
#     with open(json_fname, 'r') as f:
#         targets = json.load(f)
#
#     targets['tile_no'] = tile_no
#     targets['walk_no'] = walk_no
#     targets['idx'] = idx
#
#     return targets



def get_gt_graph(targets):

    nodes = np.array(targets['bboxes'])
    edges = np.array(targets['relation_labels'])

    graph_gt = nx.DiGraph()

    # Populate graph with nodes
    for i, n in enumerate(nodes):
        graph_gt.add_node(i, pos=n, weight=1.0)
    for e in edges:
        graph_gt.add_edge(e[0], e[1])

    graph_gt.remove_edges_from(nx.selfloop_edges(graph_gt))

    return graph_gt


def is_in_roi(ego_x_y_yaw, roi_area_xxyy, margin=0.0):
    return ego_x_y_yaw[0]-margin > roi_area_xxyy[0] and \
           ego_x_y_yaw[0]+margin < roi_area_xxyy[1] and \
           ego_x_y_yaw[1]-margin > roi_area_xxyy[2] and \
           ego_x_y_yaw[1]+margin < roi_area_xxyy[3]


def transform_graph_to_pos_indexing(G):
    G_ = nx.DiGraph()
    for n in G.nodes():
        pos = G.nodes[n]['pos']
        pos_int = (int(pos[0]), int(pos[1]))
        if 'weight' not in G.nodes[n]:
            G.nodes[n]['weight'] = 1.0
        G_.add_node(pos_int, pos=G.nodes[n]['pos'], weight=G.nodes[n]['weight'], score=G.nodes[n]['score'])
    for e in G.edges():
        pos_start = G.nodes[e[0]]['pos']
        pos_end = G.nodes[e[1]]['pos']
        pos_start_int = (int(pos_start[0]), int(pos_start[1]))
        pos_end_int = (int(pos_end[0]), int(pos_end[1]))
        G_.add_edge(pos_start_int, pos_end_int)

    return G_
