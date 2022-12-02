import numpy as np
import torch
import os
from glob import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2334477275000
from av2.datasets.motion_forecasting import scenario_serialization
from pathlib import Path

import cv2
import time
import torch_geometric.data.dataset
import torchvision.transforms as T
from shapely.geometry import LineString, MultiLineString, Point
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scipy
from data.av2.settings import get_transform_params


from lanegnn.utils import is_in_mask_loop, get_gt_sdf_with_direction, get_pointwise_edge_gt, get_delaunay_triangulation, \
    halton, get_random_edges, get_crop_mask_img, get_node_endpoint_gt, poisson_disk_sampling


class TrajectoryDatasetAV2(torch_geometric.data.Dataset):

    def __init__(self, path, params, num_node_samples: int = 500, gt_pointwise: bool = True, N_interp: int = 10):
        super(TrajectoryDatasetAV2, self).__init__(path)
        self.params = params
        self.path = path

        self.city_name = "austin"

        self.sat_image = np.asarray(Image.open("/data/lane-segmentation/woven-data/original/Austin_extended.png"))

        if os.path.exists("scenario_files.txt"):
            print("Loading scenario files from file scenario_files.txt")
            with open("scenario_files.txt", "r") as f:
                self.scenario_files = f.readlines()
                self.scenario_files = [x.strip() for x in self.scenario_files]
        else:
            all_scenario_files = sorted(glob("/data/argoverse2/motion-forecasting/val/00*/*.parquet"))
            self.scenario_files = []

            for sf in all_scenario_files:
                scenario = scenario_serialization.load_argoverse_scenario_parquet(Path(sf))
                if scenario.city_name == self.city_name:
                    self.scenario_files.append(sf)

            # serialize all scenario files
            with open("scenario_files.txt", "w") as f:
                for sf in self.scenario_files:
                    f.write(sf + "\n")

        print("Loading tracklets for {} scenarios.".format(len(self.scenario_files)))

        [self.R, self.c, self.t] = get_transform_params(self.city_name)

        self.num_node_samples = num_node_samples
        self.gt_pointwise = gt_pointwise
        self.N_interp = N_interp
        self.point_samping_method = "poisson"  # Either uniform or halton or poisson


    def __len__(self):
        return len(self.scenario_files)


    def get_crop_mask_img(self, edge_angle, mid_x, mid_y, rgb_context):

        # Size of quadratic destination image
        crop_size = 180

        crop_rad = edge_angle
        crop_x = mid_x + 128
        crop_y = mid_y + 128
        center = np.array([crop_x, crop_y])

        # Source point coordinates already in coordinate system around center point of future crop
        src_pts = np.array([[-crop_size//2,    crop_size//2-1],
                            [-crop_size//2,   -crop_size//2],
                            [ crop_size//2-1, -crop_size//2],
                            [ crop_size//2-1,  crop_size//2-1]])

        # Rotate source points
        R = np.array([[np.cos(crop_rad), -np.sin(crop_rad)],
                    [np.sin(crop_rad),  np.cos(crop_rad)]])
        src_pts = np.matmul(R, src_pts.T).T + center
        src_pts = src_pts.astype(np.float32)

        # Destination points are simply the corner points in the new image
        dst_pts = np.array([[0,           crop_size-1],
                            [0,           0],
                            [crop_size-1, 0],
                            [crop_size-1, crop_size-1]],
                            dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the rectangle
        crop = cv2.warpPerspective(rgb_context, M, (crop_size, crop_size))

        return crop


    def bayes_update_gridmap(self, map, x, y, p):
        """
        Update the probability of a grid cell given the current probability and the new measurement
        :param map: The grid map
        :param x: The x coordinate of the cell
        :param y: The y coordinate of the cell
        :param p: The probability of the measurement
        :return: The updated probability of the cell
        """
        return map[y, x] * p / (map[y, x] * p + (1 - map[y, x]) * (1 - p))




    def get_scenario_data(self, scenario_file):

        tracklet_min_x = 1e10
        tracklet_min_y = 1e10
        tracklet_max_x = -1e10
        tracklet_max_y = -1e10

        scenario_path = Path(scenario_file.strip())
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)

        scenario_data = {
            "scenario_id": scenario.scenario_id,
            "tracklets": [],
            "timestamps": [],
            "types": [],
            "city_name": scenario.city_name,
        }

        for track in scenario.tracks:
            # Get actor trajectory and heading history
            actor_trajectory = np.array([list(object_state.position) for object_state in track.object_states])
            actor_headings = np.array([object_state.heading for object_state in track.object_states])
            actor_timesteps = np.array([object_state.timestep for object_state in track.object_states])

            actor_trajectory = actor_trajectory[::5]
            actor_headings = actor_headings[::5]
            actor_timesteps = actor_timesteps[::5]


            if track.object_type != "vehicle":
                continue

            # skip parked vehicles
            if np.max(actor_trajectory[:, 0]) - np.min(actor_trajectory[:, 0]) < 10:
                continue
            if np.max(actor_trajectory[:, 1]) - np.min(actor_trajectory[:, 1]) < 10:
                continue

            # Coordinate transformation
            for i in range(len(actor_trajectory)):
                bb = np.array([actor_trajectory[i, 0], actor_trajectory[i, 1], 0])
                tmp = self.t + self.c * self.R @ bb
                actor_trajectory[i] = tmp[0:2]

            tracklet_min_x = min(tracklet_min_x, np.min(actor_trajectory[:, 0]))
            tracklet_min_y = min(tracklet_min_y, np.min(actor_trajectory[:, 1]))
            tracklet_max_x = max(tracklet_max_x, np.max(actor_trajectory[:, 0]))
            tracklet_max_y = max(tracklet_max_y, np.max(actor_trajectory[:, 1]))

            actor_xyh = np.concatenate([actor_trajectory, actor_headings[:, None]], axis=1)

            scenario_data["tracklets"].append(actor_xyh)
            scenario_data["timestamps"].append(actor_timesteps)
            scenario_data["types"].append(track.object_type)

        for i in range(len(scenario_data["tracklets"])):
            scenario_data["tracklets"][i][:, 0] -= tracklet_min_x
            scenario_data["tracklets"][i][:, 1] -= tracklet_min_y

        rgb_crop = self.sat_image[int(tracklet_min_y):int(tracklet_max_y),
                                  int(tracklet_min_x):int(tracklet_max_x), :].copy()

        print(tracklet_min_x, tracklet_max_x, tracklet_min_y, tracklet_max_y)
        print(rgb_crop.shape)
        print(scenario_data["city_name"])

        if np.min(rgb_crop.shape[0:2]) < 100:
            print("Empty crop!")
            return None

        if len(scenario_data["types"]) == 0:
            print("No tracklets in scenario!")
            return None

        scenario_data["rgb"] = rgb_crop

        return scenario_data


    def __getitem__(self, index):

        scenario_data = self.get_scenario_data(self.scenario_files[index])
        if scenario_data is None:
            return "empty-trajectory"

        # Images
        rgb = scenario_data["rgb"]
        rgb_context = scenario_data["rgb"]
        context_regr_smooth = np.zeros(rgb_context.shape[:2], dtype=np.float32)

        # Get 1 graph start node and N graph end nodes
        G_tracklet = nx.DiGraph()
        gt_multilines = []
        gt_lines = []

        for trajectory in scenario_data["tracklets"]:
            for i in range(len(trajectory) - 1):
                p1 = trajectory[i][0:2]
                p2 = trajectory[i + 1][0:2]

                p1 = int(p1[0]), int(p1[1])
                p2 = int(p2[0]), int(p2[1])
                G_tracklet.add_node(p1, pos=[p1[0], p1[1]])
                G_tracklet.add_node(p2, pos=[p2[0], p2[1]])
                G_tracklet.add_edge(p1, p2)

                line = [trajectory[i][0], trajectory[i][1], trajectory[i+1][0], trajectory[i+1][1]]
                gt_lines.append(line)
                gt_multilines.append((p1, p2))


        gt_multiline_shapely = MultiLineString(gt_multilines)
        gt_lines = np.array(gt_lines)

        gt_lines_shapely = []
        for l in gt_lines:
            x1 = l[0]
            y1 = l[1]
            x2 = l[2]
            y2 = l[3]
            gt_lines_shapely.append(LineString([(x1, y1), (x2, y2)]))

        # Normalize drivable surface to create a uniform distribution
        drivable = np.ones(rgb.shape[0:2], dtype=np.float32)
        non_drivable_mask = drivable < 0.5

        if self.params.preprocessing.sampling_method == "halton":
            point_coords = halton(2, self.params.preprocessing.num_node_samples - 1) * 255
            halton_points = point_coords.astype(np.int32)

            # filter all points where non_drivable_mask is True
            point_coords = halton_points[
                np.logical_not(non_drivable_mask[halton_points[:, 0], halton_points[:, 1]])]

        elif self.params.preprocessing.sampling_method == "poisson":
            poisson_points = poisson_disk_sampling(r_min=25,
                                                   width=non_drivable_mask.shape[0],
                                                   height=non_drivable_mask.shape[1])
            poisson_points = np.array(poisson_points).astype(np.int32)

            # filter all points where non_drivable_mask is True
            point_coords = poisson_points[np.logical_not(non_drivable_mask[poisson_points[:, 0], poisson_points[:, 1]])]
        else:
            raise NotImplementedError("Sampling method not implemented")


        # Construct edges based on obstacle rejection
        if self.params.preprocessing.edge_proposal_method == 'triangular':
            edge_proposal_pairs = get_delaunay_triangulation(point_coords)
        elif self.params.preprocessing.edge_proposal_method == 'random':
            edge_proposal_pairs = get_random_edges(point_coords)

        edge_proposal_pairs = np.unique(edge_proposal_pairs, axis=0).tolist()

        # Triangulation based edge proposal generation
        edges = list()
        edges_locs = list()
        node_gt_list = list()
        node_pos_list = list()

        for i, anchor in enumerate(point_coords):
            node_tensor = torch.tensor([anchor[0], anchor[1]]).reshape(1, -1)
            node_pos_list.append(node_tensor)
            shapely_point = Point([(anchor[1], anchor[0])])
            node_gt_score = shapely_point.distance(gt_multiline_shapely)
            node_gt_list.append(node_gt_score)

        if len(node_pos_list) == 0:
            return "empty-trajectory"

        node_pos = torch.cat(node_pos_list, dim=0)

        for [i, j] in edge_proposal_pairs:
            anchor = point_coords[i]
            point = point_coords[j]

            if is_in_mask_loop(non_drivable_mask, anchor[1], anchor[0], point[1], point[0],
                               self.params.preprocessing.N_interp):

                edges_locs.append((anchor, point))
                edges.append((i, j))

        if len(edges) == 0:
            return "empty-trajectory"

        # print("--normalize node gt")

        # Min-max scaling of node_scores
        node_gt_score = torch.FloatTensor(node_gt_list)
        node_gt_score -= node_gt_score.min()
        node_gt_score /= node_gt_score.max()
        node_gt_score = 1 - node_gt_score
        node_gt_score = node_gt_score ** 8

        # Scales edge img feature to VGG16 input size
        transform2vgg = T.Compose([
            T.ToPILImage(),
            T.Resize(32),
            T.ToTensor()])

        # Crop edge img feats and infer edge GT from SDF
        # print("len(edges)", len(edges))
        gt_sdf, angles_gt_dense = get_gt_sdf_with_direction(gt_lines_shapely, rgb.shape)

        edge_attr_list = list()
        edge_img_feats_list = list()
        edge_idx_list = list()

        if self.params.preprocessing.gt_pointwise:
            cum_edge_dist_list = list()
            angle_penalty_list = list()

        # print("--edge feat constr")

        for edge_idx, edge in enumerate(edges):
            i, j = edge
            s_x, s_y = point_coords[i][1], point_coords[i][0]
            e_x, e_y = point_coords[j][1], point_coords[j][0]

            delta_x, delta_y = e_x - s_x, e_y - s_y
            mid_x, mid_y = s_x + delta_x / 2, s_y + delta_y / 2

            edge_len = np.sqrt(delta_x ** 2 + delta_y ** 2)
            edge_angle = np.arctan(delta_y / (delta_x + 1e-6))

            edge_tensor = torch.tensor([edge_angle, edge_len, mid_x, mid_y]).reshape(1, -1)
            edge_attr_list.append(edge_tensor)

            # Crop edge images:
            crop_img_rgb = get_crop_mask_img(edge_angle, mid_x, mid_y, rgb_context)
            crop_img_rgb_resized = transform2vgg(crop_img_rgb).unsqueeze(0)
            crop_img_sdf = get_crop_mask_img(edge_angle, mid_x, mid_y, context_regr_smooth)
            crop_img_sdf_resized = transform2vgg(crop_img_sdf).unsqueeze(0)
            # RGB and SDF in range [0.0, 1.0] float32

            if self.params.preprocessing.gt_pointwise:
                cum_edge_distance, angle_penalty = get_pointwise_edge_gt(s_x, s_y, e_x, e_y,
                                                                         self.params.preprocessing.N_interp,
                                                                         gt_multiline_shapely, angles_gt_dense)
                cum_edge_dist_list.append(cum_edge_distance)
                angle_penalty_list.append(angle_penalty)
                edge_idx_list.append((i, j))

            edge_img_feats_list.append(torch.cat([crop_img_rgb_resized, crop_img_sdf_resized], dim=1))

        edge_img_feats = torch.cat(edge_img_feats_list, dim=0)

        edge_attr = torch.cat(edge_attr_list, dim=0)

        # Pointwise edge score normalization
        if self.params.preprocessing.gt_pointwise:
            try:
                cum_edge_dist_gt = np.array(cum_edge_dist_list)
                cum_edge_dist_gt -= cum_edge_dist_gt.min()
                cum_edge_dist_gt /= cum_edge_dist_gt.max()
                cum_edge_dist_gt = 1 - cum_edge_dist_gt
                edge_gt_score = cum_edge_dist_gt ** 8
            except:
                edge_gt_score = np.zeros(len(cum_edge_dist_list))
                pass


        if any(np.isnan(edge_gt_score)):
            print("Warning: edge_gt_score contains NaNs")
            return "empty-trajectory"
        if any(np.isnan(node_gt_score)):
            print("Warning: node_gt_score contains NaNs")
            return "empty-trajectory"


        edge_gt_score = torch.from_numpy(edge_gt_score).float()
        gt_graph = torch.tensor(gt_lines)  # [num_gt_graph_edges, 4]
        edges = torch.tensor(edges)

        g = nx.DiGraph()

        node_pos_ = node_pos.detach().numpy()
        node_pos_[:, [1, 0]] = node_pos_[:, [0, 1]]

        for edge_idx, edge in enumerate(edges):
            i, j = edge
            i, j = i.item(), j.item()
            #if edge_gt_score[edge_idx] > 0.1:
            g.add_edge(i, j)
            g.add_node(j, pos=node_pos_[j])
            g.add_node(i, pos=node_pos_[i])

        cmap = plt.get_cmap('viridis')
        cedge = np.hstack([cmap(edge_gt_score)[:, 0:3], edge_gt_score[:, None]])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(rgb)
        nx.draw_networkx(g,
                         ax=ax,
                         edge_color=cedge,
                         pos=nx.get_node_attributes(g, 'pos'),
                         with_labels=False,
                         node_size=5)
        plt.show()

        data = torch_geometric.data.Data(node_pos=node_pos,
                                         edge_index=edges.t().contiguous(),
                                         edge_attr=edge_attr,
                                         edge_img_feats=edge_img_feats,
                                         node_scores=node_gt_score.t().contiguous(),
                                         edge_scores=edge_gt_score.t().contiguous(),
                                         gt_graph=gt_graph,
                                         num_nodes=node_pos.shape[0],
                                         batch_idx=torch.tensor(len(gt_graph)),
                                         rgb=torch.FloatTensor(rgb / 255.), # [0.0, 1.0]
                                         rgb_context=torch.FloatTensor(rgb_context / 255.), # [0.0, 1.0]
                                         context_regr_smooth=torch.FloatTensor(context_regr_smooth), # [0.0, 1.0]
                                         G_tracklet=G_tracklet,
                                         )

        return data


class PreprocessedDataset(torch_geometric.data.Dataset):

    def __init__(self, path):
        super(PreprocessedDataset, self).__init__(path)
        print("Loading preprocessed dataset from {}".format(path))

        self.path = path
        self.pth_files = sorted(glob(path + '/*.pth')) + sorted(glob(path + '/*.pt'))
        print("Found {} files".format(len(self.pth_files)))
        self.check_files()

    def __len__(self):
        return len(self.pth_files)


    def check_files(self):
        for i, pth_file in enumerate(self.pth_files):
            try:
                data = torch.load(pth_file)
            except:
                print("Error loading file {}".format(pth_file))
                continue


    def __getitem__(self, index):

        data = torch.load(self.pth_files[index])

        rgb = data['rgb']
        edge_pos_feats = data['edge_pos_feats'].float()
        edge_img_feats = data['edge_img_feats'].float()
        edge_scores = data['edge_scores'].float()
        edge_indices = data['edge_indices']
        graph = data['graph']
        node_feats = data['node_feats'].float()
        node_scores = data['node_scores'].float()

        # watershed = 0.7
        # edge_scores[edge_scores > watershed] = 1.0
        # node_scores[node_scores > watershed] = 1.0
        # edge_scores[edge_scores <= watershed] = 0.0
        # node_scores[node_scores <= watershed] = 0.0

        data = torch_geometric.data.Data(node_feats=node_feats,
                                         edge_indices=edge_indices.contiguous(),
                                         edge_pos_feats=edge_pos_feats,
                                         edge_img_feats=edge_img_feats,
                                         node_scores=node_scores.contiguous(),
                                         edge_scores=edge_scores.contiguous(),
                                         edge_len=torch.tensor(len(edge_scores)),
                                         gt_graph=graph,
                                         num_nodes=node_feats.shape[0],
                                         batch_idx=torch.tensor(index),
                                         rgb=torch.FloatTensor(rgb / 255.),
                                         )

        return data





class PreprocessedDatasetSuccessor(torch_geometric.data.Dataset):

    def __init__(self, path):
        super(PreprocessedDatasetSuccessor, self).__init__()


        self.node_feats_files = []
        self.edge_files = []
        self.edge_attr_files = []
        self.edge_img_feats_files = []
        self.node_gt_files = []
        self.node_endpoint_gt_files = []
        self.edge_gt_files = []
        self.edge_gt_onehot_files = []
        self.gt_graph_files = []
        self.rgb_files = []
        self.rgb_context_files = []
        self.context_regr_smooth_files = []
        self.ego_regr_smooth_files = []

        city_str = '*'
        print(path + '/{}-node-feats.pth'.format(city_str))
        self.node_feats_files.extend(glob(path + '/{}-node-feats.pth'.format(city_str)))
        self.edge_files.extend(glob(path + '/{}-edges.pth'.format(city_str)))
        self.edge_attr_files.extend(glob(path + '/{}-edge-attr.pth'.format(city_str)))
        self.edge_img_feats_files.extend(glob(path + '/{}-edge-img-feats.pth'.format(city_str)))
        self.node_gt_files.extend(glob(path + '/{}-node-gt.pth'.format(city_str)))
        self.node_endpoint_gt_files.extend(glob(path + '/{}-node-endpoint-gt.pth'.format(city_str)))
        self.edge_gt_files.extend(glob(path + '/{}-edge-gt.pth'.format(city_str)))
        self.edge_gt_onehot_files.extend(glob(path + '/{}-edge-gt-onehot.pth'.format(city_str)))
        self.gt_graph_files.extend(glob(path + '/{}-gt-graph.pth'.format(city_str)))
        self.rgb_files.extend(glob(path + '/{}-rgb.pth'.format(city_str)))
        self.rgb_context_files.extend(glob(path + '/{}-rgb-context.pth'.format(city_str)))
        self.context_regr_smooth_files.extend(glob(path + '/{}-context-regr-smooth.pth'.format(city_str)))
        self.ego_regr_smooth_files.extend(glob(path + '/{}-ego-regr-smooth.pth'.format(city_str)))

        self.node_feats_files = sorted(self.node_feats_files)
        self.edge_files = sorted(self.edge_files)
        self.edge_attr_files = sorted(self.edge_attr_files)
        self.edge_img_feats_files = sorted(self.edge_img_feats_files)
        self.node_gt_files = sorted(self.node_gt_files)
        self.node_endpoint_gt_files = sorted(self.node_endpoint_gt_files)
        self.edge_gt_files = sorted(self.edge_gt_files)
        self.edge_gt_onehot_files = sorted(self.edge_gt_onehot_files)
        self.gt_graph_files = sorted(self.gt_graph_files)
        self.rgb_files = sorted(self.rgb_files)
        self.rgb_context_files = sorted(self.rgb_context_files)
        self.context_regr_smooth_files = sorted(self.context_regr_smooth_files)
        self.ego_regr_smooth_files = sorted(self.ego_regr_smooth_files)

        print("Found {} samples in path {}".format(len(self.rgb_files), path))


    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, index):
        # Return reduced data object if the index is in the index_filter (to save time)

        start_time = time.time()

        node_feats = torch.load(self.node_feats_files[index])
        edges = torch.load(self.edge_files[index])
        edge_attr = torch.load(self.edge_attr_files[index])
        edge_img_feats = torch.load(self.edge_img_feats_files[index]).to(torch.float32) / 255.0 # cast uint8 to float32
        node_gt = torch.load(self.node_gt_files[index])
        node_endpoint_gt = torch.load(self.node_endpoint_gt_files[index]).float()
        edge_gt = torch.load(self.edge_gt_files[index])
        edge_gt_onehot = torch.load(self.edge_gt_onehot_files[index])
        gt_graph = torch.load(self.gt_graph_files[index])
        rgb = torch.load(self.rgb_files[index])
        rgb_context = torch.load(self.rgb_context_files[index])
        context_regr_smooth = torch.load(self.context_regr_smooth_files[index])
        ego_regr_smooth = torch.load(self.ego_regr_smooth_files[index])

        # switch node columns to match the order of the edge columns
        node_feats = torch.cat((node_feats[:, 1:2], node_feats[:, 0:1]), dim=1)


        data = torch_geometric.data.Data(node_feats=node_feats,
                                         edge_indices=edges.t().contiguous(),
                                         edge_pos_feats=edge_attr,
                                         edge_img_feats=edge_img_feats,
                                         node_scores=node_gt.t().contiguous(),
                                         #node_endpoint_gt=node_endpoint_gt.t().contiguous(),
                                         edge_scores=edge_gt.t().contiguous(),
                                         #edge_gt_onehot=edge_gt_onehot.t().contiguous(),
                                         gt_graph=gt_graph,
                                         num_nodes=node_feats.shape[0],
                                         batch_idx=torch.tensor(len(gt_graph)),
                                         rgb=torch.FloatTensor(rgb / 255.), # [0.0, 1.0]
                                         rgb_context=torch.FloatTensor(rgb_context / 255.), # [0.0, 1.0]
                                         context_regr_smooth=torch.FloatTensor(context_regr_smooth), # [0.0, 1.0]
                                         ego_regr_smooth=torch.FloatTensor(ego_regr_smooth), # [0.0, 1.0]
                                         data_time=torch.tensor(time.time() - start_time),
                                         )

        return data

