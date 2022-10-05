import networkx as nx
import numpy as np
import torch
import os
import argparse

from glob import glob
from PIL import Image
import cv2
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import codecs
import json
import time
import pickle
import torchvision.transforms as T
import torch.nn as nn

from shapely.geometry import LineString, MultiLineString, Point

import ray
from ray.util.multiprocessing import Pool

from lane_mp.utils import ParamLib, is_in_mask_loop, get_gt_sdf_with_direction, get_pointwise_edge_gt, \
    get_delaunay_triangulation, get_crop_mask_img, halton, get_random_edges

from lanenet import build_net  

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_context_centerline_regressor(ckpt=None, use_cuda=False):
    model = build_net.build_network(snapshot=ckpt, backend='resnet152', use_cuda=use_cuda, n_classes=1)
    return model

def get_ego_centerline_regressor(ckpt=None, use_cuda=False, num_channels=3):
    model = build_net.build_network(snapshot=ckpt, backend='resnet152', use_cuda=use_cuda, n_classes=1, num_channels=num_channels)
    return model

def get_node_endpoint_gt(rgb, waypoints, relation_labels, edges, node_feats):
    """
    Args:
        waypoints: list of gt graph points
        relation_labels: list of gt graph edges
        edges: list of edges
        node_feats: list of node features

    Returns:
        node_endpoint_gt: list of node endpoint gt

    """

    # swap x y axis of waypoints
    waypoints = np.array(waypoints)
    waypoints[:, 0], waypoints[:, 1] = waypoints[:, 1], waypoints[:, 0].copy()

    node_feats = node_feats.numpy()
    # possible_endpoints = np.array([edge[1] for edge in edges])
    # possible_endpoints = np.unique(possible_endpoints)
    # possible_endpoints_coords = node_feats[possible_endpoints]


    gt_starting_indices = np.unique([edge[0] for edge in relation_labels])
    gt_end_indices = np.unique([edge[1] for edge in relation_labels])

    # get end indices that are not in start indices
    gt_end_indices = gt_end_indices[~np.isin(gt_end_indices, gt_starting_indices)]

    node_endpoint_gt = torch.zeros(len(node_feats), dtype=torch.long)

    # Get the node_feats with euclidean distance closest to the endpoints
    for gt_end in gt_end_indices:
        gt_end_node_idx = np.argmin(np.sum((node_feats - waypoints[gt_end]) ** 2, axis=1))
        node_endpoint_gt[gt_end_node_idx] = 1

    # # Visualization
    # plt.imshow(rgb)
    # for e in relation_labels:
    #     s_x, s_y = waypoints[e[0]][1], waypoints[e[0]][0]
    #     e_x, e_y = waypoints[e[1]][1], waypoints[e[1]][0]
    #     plt.arrow(s_x, s_y, e_x-s_x, e_y-s_y, color='g', width=0.2, head_width=1)
    # for i, point in enumerate(node_feats):
    #     if node_endpoint_gt[i] == 1.0:
    #         plt.plot(point[1], point[0], 'g.')
    #     else:
    #         plt.plot(point[1], point[0], 'k.')
    # plt.show()

    return node_endpoint_gt




def process_chunk(data):

    # Load the data for this chunk.
    rgb_files, rgb_context_files, sdf_ego_files, sdf_context_files, drivable_files, json_files, params = data

    context_regressor = get_context_centerline_regressor(ckpt=os.path.join(params.paths.checkpoints, params.preprocessing.context_regressor_ckpt),
                                                         use_cuda=False)
    ego_regressor = get_ego_centerline_regressor(ckpt=os.path.join(params.paths.checkpoints, params.preprocessing.ego_regressor_ckpt),
                                                 num_channels=params.preprocessing.ego_regressor_num_channels,
                                                 use_cuda=False)

    context_regressor.eval()
    ego_regressor.eval()

    for rgb_file, rgb_context_file, sdf_ego_file, sdf_context_file, drivable_file, json_file in zip(rgb_files, rgb_context_files, sdf_ego_files, sdf_context_files, drivable_files, json_files):

        try:
            rgb_context = np.asarray(Image.open(rgb_context_file)) # int [0, 255]
            rgb = np.asarray(Image.open(rgb_file)) # int [0, 255]
            drivable = np.array(Image.open(drivable_file))
            graph = json.loads(codecs.open(json_file, 'r', encoding='utf-8').read())
            # sdf_context = np.asarray(Image.open(sdf_context_file).convert('L'))
            # sdf_ego = np.asarray(Image.open(sdf_ego_file).convert('L'))


            scene_tag = rgb_file.split('/')[-1].split('-')[0]
            city_name = rgb_file.split('/')[-3]

            # Skip existing samples
            maybe_existing = os.path.join(params.paths.export_dir, scene_tag) + '_{}-rgb-context.pth'.format(city_name)
            if os.path.exists(maybe_existing):
                print('Skipping existing sample: {}'.format(maybe_existing))
                continue



            # GT graph representation 
            waypoints = np.array(graph["bboxes"])
            relation_labels = np.array(graph["relation_labels"])


            # Get 1 graph start node and N graph end nodes
            G_gt_nx = nx.DiGraph()
            for e in relation_labels:
                if not G_gt_nx.has_node(e[0]):
                    G_gt_nx.add_node(e[0], pos=waypoints[e[0]])
                if not G_gt_nx.has_node(e[1]):
                    G_gt_nx.add_node(e[1], pos=waypoints[e[1]])
                G_gt_nx.add_edge(e[0], e[1])

            # Throw out all easy samples
            # max_degree = max([G_gt_nx.degree(x) for x in G_gt_nx.nodes()])
            # if max_degree <= 2:
            #     print("no intersection scenario. Skipping")
            #     continue

            start_node = [x for x in G_gt_nx.nodes() if G_gt_nx.in_degree(x) == 0 and G_gt_nx.out_degree(x) > 0][0]
            start_node_pos = G_gt_nx.nodes[start_node]['pos']
            end_nodes = [x for x in G_gt_nx.nodes() if G_gt_nx.out_degree(x) == 0 and G_gt_nx.in_degree(x) > 0]
            end_node_pos_list = [G_gt_nx.nodes[x]['pos'] for x in end_nodes]

            gt_lines = []
            gt_multilines = []
            gt_graph_edge_index = list()
            for l in relation_labels:
                line = [waypoints[l[0], 0], waypoints[l[0], 1], waypoints[l[1], 0], waypoints[l[1], 1]]
                gt_multilines.append(((waypoints[l[0], 0], waypoints[l[0], 1]), (waypoints[l[1], 0], waypoints[l[1], 1])))
                gt_lines.append(line)
                gt_graph_edge_index.append((l[0], l[1]))

            gt_lines = np.array(gt_lines)

            gt_lines_shapely = []
            for l in gt_lines:
                x1 = l[0]
                y1 = l[1]
                x2 = l[2]
                y2 = l[3]
                gt_lines_shapely.append(LineString([(x1, y1), (x2, y2)]))

            gt_multiline_shapely = MultiLineString(gt_multilines)

            # Remove park areas and set max-value for drivable surface
            # park-area 255, drivable 128, non-drivable 0

            num_park_pixels = np.sum(drivable == 255)
            num_lane_pixels = np.sum(drivable == 128)
            
            if float(num_park_pixels) / (float(num_lane_pixels) + 1) > 0.2:
                drivable[drivable > 128] = 255.0
                drivable[drivable < 128] = 0.0
                drivable[drivable == 128] = 255.0
            else:
                drivable[drivable > 128] = 0.0
                drivable[drivable < 128] = 0.0
                drivable[drivable == 128] = 255.0
            
            # Mask of non-drivable surface for violating edge rejection
            # [depr] non_drivable_mask = drivable < 255
            # [depr] sdf_ego = cv2.GaussianBlur(sdf_ego, (31, 31), cv2.BORDER_DEFAULT)

            # Feed ego-RGB / context-RGB to regressors and produce SDF approximations / drivable surface (used for node sampling)
            # RGB2BGR is necessary because regressors are trained with cv2-color-order images
            rgbcontext2sdf = torch.from_numpy(cv2.cvtColor(rgb_context, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() / 255.0
            context_regr = torch.nn.Sigmoid()(context_regressor(rgbcontext2sdf.unsqueeze(0)))
            context_regr_ = context_regr.detach().cpu().numpy()[0, 0]
            context_regr_smooth = cv2.GaussianBlur(context_regr_, (21, 21), cv2.BORDER_CONSTANT)


            if params.preprocessing.ego_regressor_num_channels == 4:
                # print("Using 4-channel ego-regressor")
                rgb_for_cat = torch.from_numpy(cv2.cvtColor(rgb_context, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() / 255.0
                rgb2sdf = torch.cat((rgb_for_cat, context_regr[0]), dim=0)
            else:
                rgb2sdf = torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() / 255.0

            ego_regr = torch.nn.Sigmoid()(ego_regressor(rgb2sdf.unsqueeze(0)))
            ego_regr = ego_regr.detach().cpu().numpy()[0, 0]
            ego_regr_smooth = cv2.GaussianBlur(ego_regr, (101, 101), cv2.BORDER_CONSTANT)

            if params.preprocessing.ego_regressor_num_channels == 4:
                # We need to cut this down to proper size again (remove context)
                ego_regr_smooth = ego_regr_smooth[128:128 + 256, 128:128 + 256]

            non_drivable_mask = ego_regr_smooth < 0.1

            # Normalize drivable surface to create a uniform distribution
            drivable_distrib = drivable/np.sum(drivable)

            #print("-- node sampling")

            if params.preprocessing.gt_pointwise:
                # ----- DOES NOT WORK RIGHT NOW: Coordinates greater than 255 are created
                if params.preprocessing.sampling_method == "uniform":
                    # Node sampling
                    # Create a flat copy of the array & sample index from 1D array
                    # with prob of the original array
                    flat = drivable_distrib.flatten()
                    sample_index = np.random.choice(a=flat.size, p=flat, size=params.preprocessing.num_node_samples)
                    adjusted_index = np.unravel_index(sample_index, drivable_distrib.shape)
                    point_coords = list(zip(*adjusted_index))

                    # Append starting point as a node
                    point_coords.append((255, 128))

                elif params.preprocessing.sampling_method == "halton":

                    # Single-level halton sampling


                    point_coords = halton(2, params.preprocessing.num_node_samples-1) * 255
                    halton_points = point_coords.astype(np.int32)

                    # filter all points where non_drivable_mask is True
                    point_coords = halton_points[np.logical_not(non_drivable_mask[halton_points[:, 0], halton_points[:, 1]])]

                    point_coords = np.concatenate((point_coords, np.array([[255, 128]])), axis=0)

                    # plot
                    # plt.imshow(non_drivable_mask)
                    # plt.scatter(point_coords[:, 1], point_coords[:, 0], c='r')
                    # plt.show()


                    # Deprecated 2-level halton sampling

                    # point_coords_sparse = halton(2, params.preprocessing.num_node_samples) * 255
                    # point_coords_dense = halton(2, 2 * params.preprocessing.num_node_samples) * 255
                    # point_coords_sparse = point_coords_sparse.astype(np.int32)
                    # point_coords_dense = point_coords_dense.astype(np.int32)
                    #
                    # dense_region = ego_regr_smooth > 0.5
                    # sparse_region = (ego_regr_smooth < 0.5) * (ego_regr_smooth > 0.1)
                    #
                    # # filter all points where non_drivable_mask is True
                    # pd = point_coords_dense[dense_region[point_coords_dense[:, 0], point_coords_dense[:, 1]]]
                    # ps = point_coords_sparse[sparse_region[point_coords_sparse[:, 0], point_coords_sparse[:, 1]]]
                    #
                    # point_coords = np.concatenate((ps, pd, np.array([[255, 128]])), axis=0)

                    # plt.imshow(ego_regr_smooth)
                    # for point in pd:
                    #     plt.plot(point[1], point[0], 'k.')
                    # for point in ps:
                    #     plt.plot(point[1], point[0], 'r.')
                    # plt.show()

            else: 
                print("SDF-wise edge GT not implemented")


            if params.preprocessing.visualize:
                # Plot non-drivable mask
                plt.cla()
                plt.imshow(non_drivable_mask)
                # Plot node positions
                for i in point_coords:
                    plt.scatter(i[1], i[0], c='red', s=6.0)
                plt.savefig(params.paths.home + "trash/figprint{}.png".format(scene_tag))

            #print("--edge construction")

            # Construct edges based on obstacle rejection
            if params.preprocessing.edge_proposal_method == 'triangular':
                edge_proposal_pairs = get_delaunay_triangulation(point_coords)
            elif params.preprocessing.edge_proposal_method == 'random':
                edge_proposal_pairs = get_random_edges(point_coords)

            edge_proposal_pairs = np.unique(edge_proposal_pairs, axis=0)
            edge_proposal_pairs = edge_proposal_pairs.tolist()

            # Triangulation based edge proposal generation
            edges = list()
            edges_locs = list()
            node_gt_list = list()
            node_feats_list = list()

            for i, anchor in enumerate(point_coords):
                node_tensor = torch.tensor([anchor[0], anchor[1]]).reshape(1, -1)
                node_feats_list.append(node_tensor)
                shapely_point = Point([(anchor[1], anchor[0])])
                node_gt_score = shapely_point.distance(gt_multiline_shapely)
                node_gt_list.append(node_gt_score)

            if len(node_feats_list) == 0:
                print("No nodes found. Skipping sample")
                continue

            node_feats = torch.cat(node_feats_list, dim=0)


            for [i, j] in edge_proposal_pairs:
                anchor = point_coords[i]
                point = point_coords[j]

                if is_in_mask_loop(non_drivable_mask, anchor[1], anchor[0], point[1], point[0], params.preprocessing.N_interp):
                    # neglected edge removal function for now
                    # feasible = graph_traversability_check(anchor[1], anchor[0], point[1], point[0], gt_lines_shapely, gt_graph_edge_index)
                    
                    edges_locs.append((anchor, point))
                    edges.append((i, j))

            if len(edges) == 0:
                print("No edges found. Skipping sample")
                continue

            #print("--normalize node gt")

            # Min-max scaling of node_scores
            node_gt_score = torch.FloatTensor(node_gt_list)
            node_gt_score -= node_gt_score.min()
            node_gt_score /= node_gt_score.max()
            node_gt_score = 1 - node_gt_score
            node_gt_score = node_gt_score**8

            # Scales edge img feature to VGG16 input size
            transform2vgg = T.Compose([
                T.ToPILImage(),
                T.Resize(32),
                T.ToTensor()])

            # Crop edge img feats and infer edge GT from SDF
            # print("len(edges)", len(edges))
            gt_sdf, angles_gt_dense = get_gt_sdf_with_direction(gt_lines_shapely)

            edge_attr_list = list()
            edge_img_feats_list = list()
            edge_idx_list = list()

            if params.preprocessing.gt_pointwise:
                cum_edge_dist_list = list()
                angle_penalty_list = list()

            #print("--edge feat constr")

            for edge_idx, edge in enumerate(edges):
                i, j = edge
                s_x, s_y = point_coords[i][1], point_coords[i][0]
                e_x, e_y = point_coords[j][1], point_coords[j][0]

                if params.preprocessing.visualize:
                    plt.arrow(s_x, s_y, e_x-s_x, e_y-s_y, color="red", width=0.5, head_width=5)

                delta_x, delta_y = e_x - s_x, e_y - s_y
                mid_x, mid_y = s_x + delta_x/2, s_y + delta_y/2

                edge_len = np.sqrt(delta_x**2 + delta_y**2)
                edge_angle = np.arctan(delta_y/(delta_x + 1e-6))

                edge_tensor = torch.tensor([edge_angle, edge_len, mid_x, mid_y]).reshape(1, -1)
                edge_attr_list.append(edge_tensor)

                # Crop edge images:
                crop_img_rgb = get_crop_mask_img(edge_angle, mid_x, mid_y, rgb_context)
                crop_img_rgb_resized = transform2vgg(crop_img_rgb).unsqueeze(0)
                crop_img_sdf = get_crop_mask_img(edge_angle, mid_x, mid_y, context_regr_smooth)
                crop_img_sdf_resized = transform2vgg(crop_img_sdf).unsqueeze(0)
                # RGB and SDF in range [0.0, 1.0] float32

                if params.preprocessing.gt_pointwise:
                    cum_edge_distance, angle_penalty = get_pointwise_edge_gt(s_x, s_y, e_x, e_y, params.preprocessing.N_interp, gt_multiline_shapely, angles_gt_dense)

                    cum_edge_dist_list.append(cum_edge_distance)
                    angle_penalty_list.append(angle_penalty)
                    edge_idx_list.append((i, j))

                edge_img_feats_list.append(torch.cat([crop_img_rgb_resized, crop_img_sdf_resized], dim=1))

            edge_img_feats = torch.cat(edge_img_feats_list, dim=0)

            edge_attr = torch.cat(edge_attr_list, dim=0)

            if params.preprocessing.visualize:
                plt.show()

            #print("--normalize edge GT")

            # Pointwise edge score normalization
            if params.preprocessing.gt_pointwise:
                try:
                    cum_edge_dist_gt = np.array(cum_edge_dist_list)
                    cum_edge_dist_gt -= cum_edge_dist_gt.min()
                    cum_edge_dist_gt /= cum_edge_dist_gt.max()
                    cum_edge_dist_gt = 1 - cum_edge_dist_gt
                    edge_gt_score = cum_edge_dist_gt * np.array(angle_penalty_list)
                    edge_gt_score = cum_edge_dist_gt**8
                except:
                    pass


            # Now we correct the edge weights according to dijsktra path
            G_proposal_nx = nx.DiGraph()
            for edge_idx, e in enumerate(edge_idx_list):
                if not G_proposal_nx.has_node(e[0]):
                    G_proposal_nx.add_node(e[0], pos=point_coords[e[0]])
                if not G_proposal_nx.has_node(e[1]):
                    G_proposal_nx.add_node(e[1], pos=point_coords[e[1]])
                G_proposal_nx.add_edge(e[0], e[1], weight=1-edge_gt_score[edge_idx])

            # Now we search for shortest path through the G_proposal_nx from start node to end nodes
            point_coords_swapped = np.array(point_coords)[:, ::-1]
            start_node_idx = np.argmin(np.linalg.norm(point_coords_swapped - start_node_pos, axis=1))
            end_node_idx_list = [np.argmin(np.linalg.norm(point_coords_swapped - end_node_pos, axis=1)) for end_node_pos in end_node_pos_list]

            # Find shortest path
            try:
                shortest_paths = [nx.shortest_path(G_proposal_nx, start_node_idx, end_node_idx, weight="weight") for end_node_idx in end_node_idx_list]
                dijkstra_edge_list = list()
                for path in shortest_paths:
                    dijkstra_edge_list += list(zip(path[:-1], path[1:]))
            except nx.NetworkXNoPath as e:
                print(e)
                dijkstra_edge_list = list()

            # Now we correct the edge weights according to dijsktra path
            edge_gt_score_dijkstra = edge_gt_score.copy()
            for idx in range(len(edge_idx_list)):
                e = edge_idx_list[idx]
                if (e[0], e[1]) in dijkstra_edge_list:
                    # print("Maximizing score for edge {}-{} because it is in dijkstra edge list".format(e[0], e[1]))
                    edge_gt_score_dijkstra[idx] = 1.0
                    edge_gt_score[idx] = 1.0

            # Maybe one-hot encoding of path is better?
            edge_gt_score_dijkstra[edge_gt_score_dijkstra < 0.999] = 0

            # print("Took {} seconds to find dijkstra path".format(time.time() - start_time))

            # # And we plot it for debugging
            # fig, axarr = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
            # axarr[0].imshow(rgb)
            # axarr[1].imshow(rgb)
            # axarr[0].title.set_text('Old scoring')
            # axarr[1].title.set_text('Dijkstra scoring')
            #
            # # sort edge list based on scores
            # edge_gt_score, edge_idx_list = zip(*sorted(zip(edge_gt_score, edge_idx_list), key=lambda x: x[0], reverse=False))
            #
            # for list_idx, e in enumerate(edge_idx_list):
            #     s_x, s_y = point_coords[e[0]][1], point_coords[e[0]][0]
            #     e_x, e_y = point_coords[e[1]][1], point_coords[e[1]][0]
            #     axarr[0].arrow(s_x, s_y, e_x-s_x, e_y-s_y, color=plt.get_cmap('viridis')(edge_gt_score[list_idx]), width=0.2, head_width=1)
            #     axarr[1].arrow(s_x, s_y, e_x-s_x, e_y-s_y, color=plt.get_cmap('viridis')(edge_gt_score_dijkstra[list_idx]), width=0.2, head_width=1)
            # plt.plot(start_node_pos[0], start_node_pos[1], 'bo')
            # for end_node in end_node_pos_list:
            #     plt.plot(end_node[0], end_node[1], 'ro')
            # plt.show()


            node_endpoint_gt = get_node_endpoint_gt(rgb, waypoints, relation_labels, edges, node_feats)


            edge_gt_score = torch.from_numpy(edge_gt_score).float()
            edge_gt_score_dijkstra = torch.from_numpy(edge_gt_score_dijkstra).float()

            gt_graph = torch.tensor(gt_lines) # [num_gt_graph_edges, 4]
            edges = torch.tensor(edges)

            if not os.path.exists(params.paths.export_dir):
                os.makedirs(params.paths.export_dir, exist_ok=True)

            print("saving to", os.path.join(params.paths.export_dir, scene_tag) + '*.pth')


            torch.save(node_feats, os.path.join(params.paths.export_dir, scene_tag) + '_{}-node-feats.pth'.format(city_name))
            torch.save(node_endpoint_gt, os.path.join(params.paths.export_dir, scene_tag) + '_{}-node-endpoint-gt.pth'.format(city_name))
            torch.save(edges, os.path.join(params.paths.export_dir, scene_tag) + '_{}-edges.pth'.format(city_name))
            torch.save(edge_attr, os.path.join(params.paths.export_dir, scene_tag) + '_{}-edge-attr.pth'.format(city_name))

            # convert edge_img_feats to float range [0, 255] before casting to uint8 [0, 255]
            edge_img_feats = edge_img_feats * 255.0
            torch.save(edge_img_feats.to(torch.uint8), os.path.join(params.paths.export_dir, scene_tag) + '_{}-edge-img-feats.pth'.format(city_name))

            torch.save(node_gt_score, os.path.join(params.paths.export_dir, scene_tag) + '_{}-node-gt.pth'.format(city_name))
            torch.save(edge_gt_score, os.path.join(params.paths.export_dir, scene_tag) + '_{}-edge-gt.pth'.format(city_name))
            torch.save(edge_gt_score_dijkstra, os.path.join(params.paths.export_dir, scene_tag) + '_{}-edge-gt-onehot.pth'.format(city_name))
            torch.save(context_regr_smooth, os.path.join(params.paths.export_dir, scene_tag) + '_{}-context-regr-smooth.pth'.format(city_name)) # [0.0, 1.0]
            torch.save(ego_regr_smooth, os.path.join(params.paths.export_dir, scene_tag) + '_{}-ego-regr-smooth.pth'.format(city_name)) # [0.0, 1.0]
            torch.save(gt_graph, os.path.join(params.paths.export_dir, scene_tag) + '_{}-gt-graph.pth'.format(city_name))
            torch.save(torch.FloatTensor(rgb), os.path.join(params.paths.export_dir, scene_tag) + '_{}-rgb.pth'.format(city_name)) # [0.0,255.0]
            torch.save(torch.FloatTensor(rgb_context), os.path.join(params.paths.export_dir, scene_tag) + '_{}-rgb-context.pth'.format(city_name)) # [0.0,255.0]
        except Exception as e:
             print(e)
             print(rgb_file)
             log_file = open(os.path.join(params.paths.package, "prepr_log.txt"), 'a')
             log_file.write(rgb_file)
             log_file.close()
             continue
    

def process_all_chunks(params: ParamLib, chunk_size: int, city_name: str):

    path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, city_name, params.paths.split)
    export_dir = os.path.join(params.paths.dataroot_ssd, params.paths.rel_dataset, "preprocessed", params.paths.split,
                              params.paths.config_name)
    params.paths.export_dir = export_dir


    rgb_files = sorted(glob(path + '/*-rgb.png'))
    rgb_context_files = sorted(glob(path + '/*-rgb-context.png'))
    sdf_context_files = sorted(glob(path + '/*-sdf-context.png'))
    sdf_ego_files = sorted(glob(path + '/*-centerlines-sdf-ego.png'))
    drivable_files = sorted(glob(path + '/*-drivable.png'))
    json_files = sorted(glob(path + '/*-targets.json'))

    print(len(json_files))

    assert len(rgb_files) > 0
    assert len(rgb_files) == len(rgb_context_files)
    assert len(rgb_context_files) == len(sdf_context_files)
    assert len(sdf_context_files) == len(sdf_ego_files)
    assert len(sdf_ego_files) == len(drivable_files)
    assert len(drivable_files) == len(json_files)


    # Shuffle all files with same permutation
    perm = np.random.permutation(len(rgb_files))
    rgb_files = [rgb_files[i] for i in perm]
    rgb_context_files = [rgb_context_files[i] for i in perm]
    sdf_context_files = [sdf_context_files[i] for i in perm]
    sdf_ego_files = [sdf_ego_files[i] for i in perm]
    drivable_files = [drivable_files[i] for i in perm]
    json_files = [json_files[i] for i in perm]

    # Divide the set of scenes (per split) into chunks that are each processed by a different worker node.
    rgb_chunks = list(chunks(rgb_files, chunk_size))
    rgb_context_chunks = list(chunks(rgb_context_files, chunk_size))
    sdf_context_chunks = list(chunks(sdf_context_files, chunk_size))
    drivable_chunks = list(chunks(drivable_files, chunk_size))
    json_chunks = list(chunks(json_files, chunk_size))
    sdf_ego_chunks = list(chunks(sdf_ego_files, chunk_size))

    chunk_data = list()
    for rgb_chunk, rgb_context_chunk, sdf_ego_chunk, sdf_context_chunk, drivable_chunk, json_chunk in \
            zip(rgb_chunks, rgb_context_chunks, sdf_ego_chunks, sdf_context_chunks, drivable_chunks, json_chunks):
        chunk_data.append((rgb_chunk, rgb_context_chunk, sdf_ego_chunk, sdf_context_chunk, drivable_chunk, json_chunk, params))

    # Keep for debugging:
    #process_chunk(chunk_data[2])

    # Write preprocessing log
    global_log_file = open(os.path.join(params.paths.package, "prepr_log.txt"), "w")
    global_log_file.write('Hello World!')
    global_log_file.close()

    # Parallelized operation
    pool = Pool()
    pool.map(process_chunk, [data for data in chunk_data])


if __name__ == '__main__':

    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Do Preprocessing")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--city', type=str, help='one of all,mia,pit,pao,atx', required=True)

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)

    # num_cpus = 26
    num_cpus = 18

    city_name = "*" if opt.city == "all" else opt.city
    print("Preprocessing city: {}".format(city_name))

    path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, city_name, params.paths.split)
    num_samples = len(glob(path + '/*-rgb.png'))
    num_chunks = int(np.ceil(num_samples / num_cpus))

    ray.init(num_cpus=num_cpus,
             include_dashboard=False,
             _system_config={"automatic_object_spilling_enabled": True,
                             "object_spilling_config": json.dumps(
                                 {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}}, )}, )

    # Construct graphs using the generated scene metadata & store them to config_name-folder
    process_all_chunks(params, num_chunks, city_name)
