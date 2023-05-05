from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import psutil
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
from scipy.interpolate import splprep, splev
from data.av2.settings import *
from aggregation.utils import get_scenario_centerlines, crop_graph


# random shuffle seed
np.random.seed(seed=int(time.time()))
# np.random.seed(seed=0)


def get_parallel_lanes(subgraph_visible, successor_graph):
    # get starting node of successor_graph
    start_edge_ego = None
    for e in successor_graph.edges():
        if successor_graph.in_degree(e[0]) == 0:
            start_edge_ego = e
            break

    # get starting nodes in subgraph_visible
    start_edges = []
    for e in subgraph_visible.edges():
        if subgraph_visible.in_degree(e[0]) == 0:
            start_edges.append(e)

    # get euclidean distances between start_edge_ego and start_edges
    distances = []
    for e in start_edges:
        distances.append(
            np.linalg.norm(subgraph_visible.nodes[e[0]]["pos"] - successor_graph.nodes[start_edge_ego[0]]["pos"]))
    distances = np.array(distances)

    # get angles between start_edge_ego and start_edges
    angles = []
    for e in start_edges:
        angles.append(
            np.arctan2(subgraph_visible.nodes[e[0]]["pos"][1] - successor_graph.nodes[start_edge_ego[0]]["pos"][1],
                       subgraph_visible.nodes[e[0]]["pos"][0] - successor_graph.nodes[start_edge_ego[0]]["pos"][0]))
    angles = np.array(angles)

    max_distance = 50
    min_distance = 2
    max_angle = np.pi / 4

    criterium = np.logical_and(distances < max_distance, np.abs(angles) < max_angle, distances > min_distance)

    parallel_lanes = [edge for i, edge in enumerate(start_edges) if criterium[i]]

    return parallel_lanes


def process_successor_graph(G_):
    query_position = np.array([crop_size // 2, crop_size - 1])

    G = G_.copy(as_view=False)

    # Get the closest node in G_existing with respect to ego_pos
    closest_node = None
    closest_node_dist = np.inf
    for n in G_.nodes():
        dist = np.linalg.norm(query_position - G_.nodes[n]['pos'])
        if dist < closest_node_dist:
            closest_node_dist = dist
            closest_node = n

    # keep only nodes that are reachable from closest_node
    reachable_nodes = nx.descendants(G, closest_node)
    reachable_nodes.add(closest_node)
    G = G.subgraph(reachable_nodes).copy(as_view=False)

    return G


def process_keypoint_graph(G_, intersections_gt_crop):
    G = G_.copy(as_view=False)

    # add node attributes for convenience
    for node in G.nodes():
        node_pos = G.nodes[node]["pos"]
        G.nodes[node]["is_intersection"] = intersections_gt_crop[node_pos[1], node_pos[0]]
        G.nodes[node]["is_start"] = False
        G.nodes[node]["is_end"] = False
        G.nodes[node]["is_enter_intersection"] = False
        G.nodes[node]["is_exit_intersection"] = False

        # if no predecessors: set is_start to True
        if len(list(G.predecessors(node))) == 0:
            G.nodes[node]["is_start"] = True
            print("start node position: ", node_pos)

        # if no successors: set is_end to True
        if len(list(G.successors(node))) == 0:
            G.nodes[node]["is_end"] = True
            print("end node position: ", node_pos)

    for node in G.nodes():
        node_pos = G.nodes[node]["pos"]
        G_successors = list(G.successors(node))
        if len(G_successors) > 0:
            if not G.nodes[node]["is_intersection"] and any(
                    [G.nodes[succ]["is_intersection"] for succ in G_successors]):
                G.nodes[node]["is_enter_intersection"] = True
                print("enter intersection node position: ", node_pos)

            elif G.nodes[node]["is_intersection"] and not any(
                    [G.nodes[succ]["is_intersection"] for succ in G_successors]):
                G.nodes[node]["is_exit_intersection"] = True
                print("exit intersection node position: ", node_pos)
        else:
            if G.nodes[node]["is_intersection"]:
                G.nodes[node]["is_exit_intersection"] = True
                print("exit intersection node position: ", node_pos)

        G_predecessors = list(G.predecessors(node))
        if len(G_predecessors) == 0:
            if G.nodes[node]["is_intersection"]:
                G.nodes[node]["is_enter_intersection"] = True
                print("enter intersection node position: ", node_pos)

    # make new graph and add edges according to new node attributes
    G_intersections = nx.DiGraph()
    for node in G.nodes():
        if any([G.nodes[node]["is_start"],
                G.nodes[node]["is_end"],
                G.nodes[node]["is_enter_intersection"],
                G.nodes[node]["is_exit_intersection"]]):
            G_intersections.add_node(node,
                                     pos=G.nodes[node]["pos"],
                                     is_intersection=G.nodes[node]["is_intersection"],
                                     is_start=G.nodes[node]["is_start"],
                                     is_end=G.nodes[node]["is_end"],
                                     is_enter_intersection=G.nodes[node]["is_enter_intersection"],
                                     is_exit_intersection=G.nodes[node]["is_exit_intersection"])

    # go through all pairs of nodes and add edges between them if they are connected in G without passing through
    # any of the other nodes
    for node_candidate_1 in G_intersections.nodes():
        for node_candidate_2 in G_intersections.nodes():
            if node_candidate_1 != node_candidate_2:
                all_other_candidates = [node for node in G_intersections.nodes() if
                                        node != node_candidate_1 and node != node_candidate_2]
                if nx.has_path(G, node_candidate_1, node_candidate_2):
                    path = nx.shortest_path(G, node_candidate_1, node_candidate_2)
                    if all([node not in path for node in all_other_candidates]):
                        path_coordinates = [G.nodes[node]["pos"] for node in path]
                        G_intersections.add_edge(node_candidate_1, node_candidate_2, path=path_coordinates)

                        print("added edge between two nodes")

    return G_intersections


def fit_spline_to_edges(G_):
    for edge in G_.edges():
        waypoints = G_.edges[edge]["path"]
        if len(waypoints) > 5:
            # fit cubic spline to waypoints
            waypoints = np.array(waypoints)
            tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=0.0)
            # t is the knot points
            # c are the coefficients
            # k is the degree of the spline

            u_fine = np.linspace(0, 1, 100)
            x_fine, y_fine = splev(u_fine, tck, der=0)

            waypoints_fine = np.stack([x_fine, y_fine], axis=1)
            G_.edges[edge]["path_fine"] = waypoints_fine.tolist()
            G_.edges[edge]["path_tck"] = tck
        else:
            G_.edges[edge]["path_fine"] = G_.edges[edge]["path"]
            G_.edges[edge]["path_tck"] = None
    return G_




def get_dataset_split(city_name, x, y, y_min_cut):
    coordinates = city_split_coordinates_dict[city_name]

    y = y + y_min_cut

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

        if sat_image_crop is not None:
            break


    tracklet_image_crop = crop_img_at_pose(tracklet_image, [pos_x, pos_y, angle], crop_size)
    drivable_gt_crop = crop_img_at_pose(drivable_gt, [pos_x, pos_y, angle], crop_size)

    crop_center_x = int(pos_x + np.sin(angle) * crop_size / 2)
    crop_center_y = int(pos_y - np.cos(angle) * crop_size / 2)

    return sat_image_crop, tracklet_image_crop, drivable_gt_crop, [crop_center_x, crop_center_y, angle]


def process_samples(args, city_name, G_annot, sat_image_, drivable_gt, intersections_gt, out_path_root,
                    max_num_samples=100, crop_size=256, y_min_cut=0):

    print("In process_samples for city: {}".format(city_name))

    num_train_samples = 0
    num_eval_samples = 0
    num_test_samples = 0

    num_branching = 0
    num_straight = 0


    if args.source == "lanegraph":

        edge_0_pos = np.array([G_annot.nodes[edge[0]]['pos'] for edge in G_annot.edges()])
        edge_1_pos = np.array([G_annot.nodes[edge[1]]['pos'] for edge in G_annot.edges()])
        G_annot_edge_pos = np.array([edge_0_pos, edge_1_pos])

        sample_num = 0
        start_time = time.time()

        while sample_num < max_num_samples:

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
            agent_trajectory = agent_trajectory[0:-50]
            agent_trajectory = agent_trajectory[::10]
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

                dataset_split = get_dataset_split(city_name, x_noise, y_noise, y_min_cut)
                if dataset_split == None:
                    continue

                if args.eval_test == True:
                    if dataset_split == "train":
                        continue

                if num_eval_samples > 2000 and dataset_split == "eval":
                    continue
                if num_test_samples > 2000 and dataset_split == "test":
                    continue

                out_path = os.path.join(out_path_root, dataset_split)
                sample_id = "{}-{}-{}".format(city_name, x_noise, y_noise)

                if os.path.exists(os.path.join(out_path, "{}.pth".format(sample_id))):
                    continue

                pos_noise = np.array([x_noise, y_noise])
                R_noise = np.array([[np.cos(yaw_noise), -np.sin(yaw_noise)],
                                    [np.sin(yaw_noise),  np.cos(yaw_noise)]])

                subgraph_visible = crop_graph(G_annot, G_annot_edge_pos, x_noise-500, x_noise+500, y_noise-500, y_noise+500)

                if subgraph_visible.number_of_edges() < 5 or subgraph_visible.number_of_nodes() < 5:
                    continue

                # Source points are around center in satellite image crop
                center = np.array([512, 512])

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

                intersections_gt_crop_ = intersections_gt[int(y_noise - crop_size_large):int(y_noise + crop_size_large),
                                         int(x_noise - crop_size_large):int(x_noise + crop_size_large)].copy()
                intersections_gt_crop = cv2.warpPerspective(intersections_gt_crop_, M, (crop_size, crop_size), cv2.INTER_NEAREST)


                successor_graph = process_successor_graph(subgraph_visible)
                parallel_lanes = get_parallel_lanes(subgraph_visible, successor_graph)

                keypoint_graph = process_keypoint_graph(successor_graph, intersections_gt_crop)
                keypoint_graph = fit_spline_to_edges(keypoint_graph)


                # Visualize graph
                fig, ax = plt.subplots(1, 2, figsize=(10, 10))
                ax[0].imshow(sat_image_crop)
                ax[0].imshow(intersections_gt_crop, alpha=0.5)
                nx.draw(subgraph_visible, ax=ax[0], pos=nx.get_node_attributes(subgraph_visible, 'pos'), node_size=10,
                        node_color="red", width=1)
                ax[1].imshow(sat_image_crop)
                ax[1].imshow(intersections_gt_crop, alpha=0.5)
                nx.draw(keypoint_graph, ax=ax[1], pos=nx.get_node_attributes(keypoint_graph, 'pos'), node_size=10,
                        node_color="red", width=1)

                # visualize parallel edges
                for edge in parallel_lanes:
                    edge_start = subgraph_visible.nodes[edge[0]]["pos"]
                    edge_end = subgraph_visible.nodes[edge[1]]["pos"]

                    ax[1].plot([edge_start[0], edge_end[0]], [edge_start[1], edge_end[1]], color="red", linewidth=1)

                # visualize paths
                for edge in keypoint_graph.edges():
                    waypoints = np.array(keypoint_graph.edges[edge]["path"])
                    ax[1].plot(waypoints[:, 0], waypoints[:, 1], color="red", linewidth=1)
                    waypoints_fine = np.array(keypoint_graph.edges[edge]["path_fine"])
                    ax[1].plot(waypoints_fine[:, 0], waypoints_fine[:, 1], color="blue", linewidth=1)


                print("---- TID: {}/{}: Sample {}/{} ({}/{}) - Samples / s = {:.2f}".format(args.thread_id, args.num_parallel,
                                                                                               out_path, sample_id, sample_num,
                                                                                               max_num_samples,
                                                                                               sample_num / (time.time() - start_time)))

                plt.savefig("{}/{}-rgb-viz.png".format(out_path, sample_id))
                Image.fromarray(sat_image_crop).save("{}/{}-rgb.png".format(out_path, sample_id))
                Image.fromarray(drivable_gt_crop.astype(np.uint8)).save("{}/{}-drivable-gt.png".format(out_path, sample_id))

                sample_num += 1

                if dataset_split == "train":
                    num_train_samples += 1
                elif dataset_split == "eval":
                    num_eval_samples += 1
                elif dataset_split == "test":
                    num_test_samples += 1
                else:
                    continue

                # if sample_type == "branching":
                #     num_branching += 1
                # elif sample_type == "straight":
                #     num_straight += 1
                # else:
                #     continue


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
    parser.add_argument("--eval_test", action="store_true", help="Generate eval and test samples only")

    args = parser.parse_args()

    print("Parsing config:", args)

    logging.basicConfig(level=logging.DEBUG, filename='aggregate_av2.log', filemode='w')

    out_path_root = args.out_path_root
    city_name = args.city_name

    city_name_dict = {
        "PIT": "pittsburgh",
        "MIA": "miami",
        "ATX": "austin",
        "WDC": "washington",
        "PAO": "paloalto",
        "DTW": "detroit",
    }

    export_final = args.export_final

    # parameters
    num_cpus = args.num_cpus

    os.makedirs(os.path.join(out_path_root, 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'eval'), exist_ok=True)
    os.makedirs(os.path.join(out_path_root, 'test'), exist_ok=True)

    sat_image = np.asarray(Image.open(os.path.join(args.urbanlanegraph_root, "{}/{}.png".format(city_name, city_name)))).astype(np.uint8)
    drivable = np.asarray(Image.open(os.path.join(args.urbanlanegraph_root, "{}/{}_drivable.png".format(city_name, city_name)))).astype(np.uint8)
    intersections = np.asarray(Image.open(os.path.join(args.urbanlanegraph_root, "{}/{}_intersections.png".format(city_name, city_name)))).astype(np.uint8)

    print("Satellite resolution: {}x{}".format(sat_image.shape[1], sat_image.shape[0]))
    print("Exporting {} tracklet annotations!".format(args.source))

    [transform_R, transform_c, transform_t] = get_transform_params(city_name.lower())
    all_tracking_files = glob('/data/argoverse2-full/*_tracking.pickle')

    # get lanes from graph file
    graph_files = glob(os.path.join(args.urbanlanegraph_root, "{}/tiles/*/*.gpickle".format(city_name)))
    G_tiles = []
    for graph_file in graph_files:
        with open(graph_file, 'rb') as f:
            G_tile = pickle.load(f)
        G_tiles.append(G_tile)

    # join all tiles
    G_annot_ = nx.DiGraph()
    for G_tile in G_tiles:
        G_annot_ = nx.union(G_annot_, G_tile, rename=("G", "H"))

    # crop graph
    G_annot = G_annot_.copy(as_view=False)
    for node in G_annot_.nodes():
        if G_annot_.nodes[node]['pos'][0] < 0 or G_annot_.nodes[node]['pos'][0] > sat_image.shape[1] or \
                G_annot_.nodes[node]['pos'][1] < 0 or G_annot_.nodes[node]['pos'][1] > sat_image.shape[0]:
            G_annot.remove_node(node)

    # single core
    if num_cpus <= 1:
        process_samples(args,
                        city_name,
                        G_annot,
                        sat_image,
                        drivable,
                        intersections,
                        out_path_root,
                        args.max_num_samples,
                        crop_size=args.crop_size)