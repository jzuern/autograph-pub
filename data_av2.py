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
        self.roi_xxyy = np.array([15000, 20000, 35000, 40000])
        # self.roi_xxyy = np.array([0, 40000, 0, 40000])
        self.sat_image = self.sat_image[self.roi_xxyy[2]:self.roi_xxyy[3],
                                        self.roi_xxyy[0]:self.roi_xxyy[1], :]

        if os.path.exists(os.path.join(self.path, "scenario_files.txt")):
            print("Loading scenario files from file", os.path.join(self.path, "scenario_files.txt"))
            with open(os.path.join(self.path, "scenario_files.txt"), "r") as f:
                self.all_scenario_files = f.readlines()
        else:
            self.all_scenario_files = sorted(glob("/data/argoverse2/motion-forecasting/val/00*/*.parquet"))

            for sf in self.all_scenario_files:
                scenario_path = Path(sf.strip())
                scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
                if scenario.city_name != self.city_name:
                    self.all_scenario_files.remove(sf)
            self.all_scenario_files = self.all_scenario_files[:10000]

            # serialize all scenario files
            with open(os.path.join(self.path, "scenario_files.txt"), "w") as f:
                for sf in self.all_scenario_files:
                    f.write(sf + "\n")

        print("Loading tracklets for {} scenarios.".format(len(self.all_scenario_files)))


        [self.R, self.c, self.t] = get_transform_params(self.city_name)


        self.num_node_samples = num_node_samples
        self.gt_pointwise = gt_pointwise
        self.N_interp = N_interp
        self.point_samping_method = "poisson"  # Either uniform or halton or poisson


    def __len__(self):
        return len(self.all_scenario_files)


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


    def get_scenario_data(self, scenario_file):

        tracklet_min_x = 1e10
        tracklet_min_y = 1e10
        tracklet_max_x = -1e10
        tracklet_max_y = -1e10
        margin = 500

        scenario_path = Path(scenario_file.strip())
        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)

        scenario_data = {
            "scenario_id": scenario.scenario_id,
            "tracklets": [],
            "timestamps": [],
            "types": [],
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

            # Coordinate transformation
            for i in range(len(actor_trajectory)):
                bb = np.array([actor_trajectory[i, 0], actor_trajectory[i, 1], 0])
                tmp = self.t + self.c * self.R @ bb
                actor_trajectory[i] = tmp[0:2] - self.roi_xxyy[::2]

            if np.min(actor_trajectory[:, 0]) < margin or \
                    np.min(actor_trajectory[:, 1]) < margin or \
                    np.max(actor_trajectory[:, 0]) > (self.roi_xxyy[1]-self.roi_xxyy[0]-margin) or \
                    np.max(actor_trajectory[:, 1]) > (self.roi_xxyy[3]-self.roi_xxyy[2]-margin):
                continue

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

        print(tracklet_min_x, tracklet_min_y, tracklet_max_x, tracklet_max_y)

        rgb_crop = self.sat_image[int(tracklet_min_y):int(tracklet_max_y),
                                  int(tracklet_min_x):int(tracklet_max_x), :].copy()
        scenario_data["rgb"] = rgb_crop

        plt.imshow(rgb_crop)
        for t in scenario_data["tracklets"]:
            plt.plot(t[:, 0], t[:, 1], "b")
        plt.show()

        return scenario_data


    def __getitem__(self, index):

        scenario_data = self.get_scenario_data(self.all_scenario_files[index])

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
        drivable = np.ones([rgb.shape[0:2]], dtype=np.float32)
        non_drivable_mask = drivable < 0.5

        if self.params.preprocessing.sampling_method == "halton":
            point_coords = halton(2, self.params.preprocessing.num_node_samples - 1) * 255
            halton_points = point_coords.astype(np.int32)

            # filter all points where non_drivable_mask is True
            point_coords = halton_points[
                np.logical_not(non_drivable_mask[halton_points[:, 0], halton_points[:, 1]])]

        elif self.params.preprocessing.sampling_method == "poisson":
            poisson_points = poisson_disk_sampling(r_min=20,
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
            return "empty-trajectory"

        node_feats = torch.cat(node_feats_list, dim=0)

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

            if self.params.preprocessing.visualize:
                plt.arrow(s_x, s_y, e_x - s_x, e_y - s_y, color="red", width=0.5, head_width=5)

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


        # Now we correct the edge weights according to dijsktra path
        G_proposal_nx = nx.DiGraph()
        for edge_idx, e in enumerate(edge_idx_list):
            if not G_proposal_nx.has_node(e[0]):
                G_proposal_nx.add_node(e[0], pos=point_coords[e[0]])
            if not G_proposal_nx.has_node(e[1]):
                G_proposal_nx.add_node(e[1], pos=point_coords[e[1]])
            G_proposal_nx.add_edge(e[0], e[1], weight=1 - edge_gt_score[edge_idx])


        edge_gt_score = torch.from_numpy(edge_gt_score).float()
        gt_graph = torch.tensor(gt_lines)  # [num_gt_graph_edges, 4]
        edges = torch.tensor(edges)

        data = torch_geometric.data.Data(x=node_feats,
                                         edge_index=edges.t().contiguous(),
                                         edge_attr=edge_attr,
                                         edge_img_feats=edge_img_feats,
                                         node_distance=node_gt_score.t().contiguous(),
                                         edge_distance=edge_gt_score.t().contiguous(),
                                         #edge_dijkstra=edge_gt_score_dijkstra.t().contiguous(),
                                         gt_graph=gt_graph,
                                         num_nodes=node_feats.shape[0],
                                         batch_idx=torch.tensor(len(gt_graph)),
                                         rgb=torch.FloatTensor(rgb / 255.), # [0.0, 1.0]
                                         rgb_context=torch.FloatTensor(rgb_context / 255.), # [0.0, 1.0]
                                         context_regr_smooth=torch.FloatTensor(context_regr_smooth), # [0.0, 1.0]
                                         G_tracklet=G_tracklet,
                                         )

        return data





class PreprocessedAV2Dataset(torch_geometric.data.Dataset):

    def __init__(self, path):
        super(PreprocessedAV2Dataset, self).__init__(path)

        self.path = path
        self.pth_files = sorted(glob(path + '/*.pt'))

    def __len__(self):
        return len(self.pth_files)

    def __getitem__(self, index):

        fname = self.pth_files[index]
        data = torch.load(fname)[0]

        return data


