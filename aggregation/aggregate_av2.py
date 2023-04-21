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

from data.av2.settings import *
from aggregation.utils import get_scenario_centerlines, resample_trajectory, Tracklet, \
    filter_tracklet, merge_successor_trajectories, iou_mask, smooth_trajectory, get_endpoints, \
    crop_graph


# random shuffle seed
np.random.seed(seed=int(time.time()))
# np.random.seed(seed=0)


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


def process_samples(args, city_name, trajectories_vehicles_, trajectories_ped_, G_annot,
                        sat_image_, tracklets_image, drivable_gt, out_path_root, max_num_samples=100, crop_size=256, y_min_cut=0):

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

                dataset_split = get_dataset_split(city_name, x_noise, y_noise)
                if dataset_split == None:
                    continue

                out_path = os.path.join(out_path_root, dataset_split)
                sample_id = "{}-{}-{}".format(city_name, x_noise, y_noise)

                if os.path.exists(os.path.join(out_path, "{}.pth".format(sample_id))):
                    continue

                pos_noise = np.array([x_noise, y_noise])
                R_noise = np.array([[np.cos(yaw_noise), -np.sin(yaw_noise)],
                                    [np.sin(yaw_noise),  np.cos(yaw_noise)]])

                subgraph_visible = crop_graph(G_annot, G_annot_edge_pos, x_noise-500, x_noise+500, y_noise-500, y_noise+500)

                if subgraph_visible.number_of_edges() < 5:
                    continue

                if subgraph_visible.number_of_nodes() < 5:
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

                # make list of annots out of edges of subgraph
                roots = [n for (n, d) in subgraph_visible.in_degree if d == 0]
                leafs = [n for (n, d) in subgraph_visible.out_degree if d == 0]

                branches = []
                for root in roots:
                    for path in nx.all_simple_paths(subgraph_visible, root, leafs):
                        if len(path) > 2:
                            branches.append(path)

                annots_ = []
                for branch in branches:
                    coordinates = [subgraph_visible.nodes[n]["pos"] for n in branch]
                    if len(coordinates) > 1:
                        coordinates = np.array(coordinates)
                        annots_.append(coordinates)

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

                query_distance_threshold = 10
                joining_distance_threshold = 4
                joining_angle_threshold = np.pi / 8

                query_points = np.array([[crop_size // 2, crop_size - 1]])

                tracklets_im_list = []

                for i_query, q in enumerate(query_points):

                    succ_traj, mask_total, mask_angle_colorized, tracklets_viz = \
                        merge_successor_trajectories(q, annots,
                                                     trajectories_ped=[],
                                                     query_distance_threshold=query_distance_threshold,
                                                     joining_distance_threshold=joining_distance_threshold,
                                                     joining_angle_threshold=joining_angle_threshold)

                    num_clusters, endpoints = get_endpoints(succ_traj, crop_size)

                    if num_clusters > 1:
                        sample_type = "branching"
                    else:
                        sample_type = "straight"

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

                    sat_image_crop_ = sat_image_[int(y_noise - crop_size_large):int(y_noise + crop_size_large),
                                      int(x_noise - crop_size_large):int(x_noise + crop_size_large)].copy()
                    sat_image_crop = cv2.warpPerspective(sat_image_crop_, M, (crop_size, crop_size), cv2.INTER_LINEAR)
                    drivable_gt_crop_ = drivable_gt[int(y_noise - crop_size_large):int(y_noise + crop_size_large),
                                        int(x_noise - crop_size_large):int(x_noise + crop_size_large)].copy()
                    drivable_gt_crop = cv2.warpPerspective(drivable_gt_crop_, M, (crop_size, crop_size),
                                                           cv2.INTER_NEAREST)

                    [cv2.circle(tracklets_viz, (qq[0], qq[1]), 2, (0, 150, 255), -1) for qq in query_points]

                    sat_image_viz = cv2.addWeighted(tracklets_viz, 0.5, sat_image_crop, 0.5, 0)

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
                    print("    Samples / s = {:.2f}".format(sample_num / (time.time() - start_time)))

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
        centers = [np.mean(t, axis=0) for t in annot_veh_]

        def print_elapsed_time(mark):
            print("{}, Elapsed time: {:.2f} s".format(mark, time.time() - start_time))

        start_time = time.time()

        sample_num = 0
        while sample_num < max_num_samples:


            # this is not guaranteed to give valid crops
            sat_image_crop, tracklet_image_crop, drivable_gt_crop, crop_center = \
                random_cropping(sat_image_, tracklets_image, drivable_gt, annot_veh_, crop_size=crop_size)

            print_elapsed_time("Random cropping")

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


            is_close = np.linalg.norm(np.array(centers) - [crop_center[0], crop_center[1]], axis=1) < np.sqrt(2) * crop_size
            annot_candidates = np.array(annot_veh_)[is_close]

            print_elapsed_time("annot candidates")


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

            print_elapsed_time("filtering")


            # filter out too short annots
            annots = [a for a in annots if len(a) > 5]

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

                succ_traj, mask_total, mask_angle_colorized, tracklets_viz = \
                    merge_successor_trajectories(q,
                                                 annots,
                                                 trajectories_ped=[],
                                                 query_distance_threshold=query_distance_threshold,
                                                 joining_distance_threshold=joining_distance_threshold,
                                                 joining_angle_threshold=joining_angle_threshold)
                print_elapsed_time("get_endpoints")

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

                [cv2.circle(tracklets_viz, (qq[0], qq[1]), 2, (0, 150, 255), -1) for qq in query_points]

                sat_image_viz = cv2.addWeighted(tracklets_viz, 0.5, sat_image_crop, 0.5, 0)

                pos_encoding = np.zeros(sat_image_crop.shape, dtype=np.float32)
                x, y = np.meshgrid(np.arange(sat_image_crop.shape[1]), np.arange(sat_image_crop.shape[0]))
                pos_encoding[q[1], q[0], 0] = 1
                pos_encoding[..., 1] = np.abs((x - q[0])) / sat_image_crop.shape[1]
                pos_encoding[..., 2] = np.abs((y - q[1])) / sat_image_crop.shape[0]
                pos_encoding = (pos_encoding * 255).astype(np.uint8)

                sample_num += 1


                print_elapsed_time("final")


                print("    Samples / s = {:.2f}".format(sample_num / (time.time() - start_time)))


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
    drivable_gt_ = np.asarray(Image.open(os.path.join(args.urbanlanegraph_root, "{}/{}_drivable.png".format(city_name, city_name)))).astype(np.uint8)

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
    G_annot_ = nx.DiGraph()
    for G_tile in G_tiles:
        G_annot_ = nx.union(G_annot_, G_tile, rename=("G", "H"))

    #G_annot_ = crop_graph(G_annot, x_min=0, x_max=sat_image_.shape[1], y_min=0, y_max=sat_image_.shape[0])

    # crop graph
    G_annot = G_annot_.copy(as_view=False)
    for node in G_annot_.nodes():
        if G_annot_.nodes[node]['pos'][0] < 0 or G_annot_.nodes[node]['pos'][0] > sat_image_.shape[1] or \
                G_annot_.nodes[node]['pos'][1] < 0 or G_annot_.nodes[node]['pos'][1] > sat_image_.shape[0]:
            G_annot.remove_node(node)

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

        sat_image = sat_image_[y_min_cut:y_max_cut, :, :].copy()
        drivable_gt = drivable_gt_[y_min_cut:y_max_cut, :].copy()

        memory_prior = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        del sat_image_
        del drivable_gt_

        memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

        print("Memory usage reduced from {} to {} GB".format(int(memory_prior / 1024.), int(memory_after / 1024.)))


        trajectories_ = [t - np.array([0, y_min_cut]) for t in trajectories_]
        trajectories_ped_ = [t - np.array([0, y_min_cut]) for t in trajectories_ped_]

        # delete trajectories that are outside of the image
        trajectories_ = [t for t in trajectories_ if np.all(t[:, 1] >= 0) and np.all(t[:, 1] < sat_image.shape[0])]
        trajectories_ped_ = [t for t in trajectories_ped_ if np.all(t[:, 1] >= 0) and np.all(t[:, 1] < sat_image.shape[0])]

        for node in G_annot.nodes:
            G_annot.nodes[node]["pos"][1] = G_annot.nodes[node]["pos"][1] - y_min_cut

        # delete nodes outside of image
        nodes_to_delete = []
        for node in G_annot.nodes:
            if G_annot.nodes[node]["pos"][1] < 0 or G_annot.nodes[node]["pos"][1] >= sat_image.shape[0]:
                nodes_to_delete.append(node)
        G_annot.remove_nodes_from(nodes_to_delete)

        print("Thread: {}, img shape: {}, len(traj): {}, len(traj_ped): {}, G_annot.number_of_nodes(): {}".
              format(args.thread_id, sat_image.shape, len(trajectories_), len(trajectories_ped_), G_annot.number_of_nodes()))

        if len(trajectories_) == 0:
            print("No trajectories in this thread. Exiting...")
            exit(0)

    else:
        sat_image = sat_image_.copy()
        drivable_gt = drivable_gt_.copy()
        del sat_image_
        del drivable_gt_

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
                        sat_image,
                        tracklets_image,
                        drivable_gt,
                        out_path_root,
                        args.max_num_samples,
                        crop_size=args.crop_size,
                        y_min_cut=y_min_cut)