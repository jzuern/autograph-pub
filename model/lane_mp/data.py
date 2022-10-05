import numpy as np
import torch
import os
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import codecs
import json
import time
import torch_geometric.data.dataset
import torchvision.transforms as T
from shapely.geometry import LineString, MultiLineString, Point
from lane_mp.utils import is_in_mask_loop, get_gt_sdf_with_direction, get_pred_distance_sdf, get_pointwise_edge_gt, \
    get_delaunay_triangulation, halton, get_crop_mask_img


class TrajectoryDataset(torch_geometric.data.Dataset):

    def __init__(self, path, num_node_samples: int = 500, gt_pointwise: bool = True, N_interp: int = 10):
        super(TrajectoryDataset, self).__init__(path)


        self.bev_rgb_files = sorted(glob(path + '/bev_rgb/*.png'))
        self.bev_semantic_files = sorted(glob(path + '/bev_sem/*.png'))
        self.bev_lidar_files = sorted(glob(path + '/lidar_render/*.png'))

        self.num_node_samples = num_node_samples
        self.gt_pointwise = gt_pointwise
        self.N_interp = N_interp

        self.point_samping_method = "halton"  # Either uniform or halton

        # get vehicle_pos_files
        pos_files = glob(os.path.join(path, "vehicles_pos", "*.txt"))

        # generate trajectories from vehicle_pos_files
        self.vehicles_traj_dict = {}

        for pos_file in pos_files:
            with open(pos_file, "r") as f:
                time_step = int(pos_file.split("/")[-1].split(".")[0])
                lines = f.readlines()
                for line in lines:
                    vehicle_id, x, y, z = line.strip().split(",")
                    if vehicle_id not in self.vehicles_traj_dict:
                        self.vehicles_traj_dict[vehicle_id] = [[time_step, float(x), float(y), float(z)]]
                    else:
                        self.vehicles_traj_dict[vehicle_id].append([time_step, float(x), float(y), float(z)])

        # sort vehicle dict by time_step
        for vehicle_id in self.vehicles_traj_dict:
            self.vehicles_traj_dict[vehicle_id] = sorted(self.vehicles_traj_dict[vehicle_id], key=lambda x: x[0])



    def __len__(self):
        return len(self.vehicles_traj_dict)

    
    def get_crop_mask_img(self, edge_angle, mid_x, mid_y, rgb_context):
        
        # Size of quadratic destination image
        crop_size = 180
        imsize = 512

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
        
        """
        # visualize cropped region
        for i in range(len(src_pts)):
            #print(i, i-1, (src_pts[i, 0], src_pts[i, 1]), (src_pts[i-1, 0], src_pts[i-1, 1]))
            cv2.line(rgb_context, (src_pts[i, 0], src_pts[i, 1]), (src_pts[i-1, 0], src_pts[i-1, 1]), (255, 255, 0), 2)
            # Query image
            cv2.line(rgb_context, (128, 128), (128+256, 128), (255, 0, 0.0), 2)
            cv2.line(rgb_context, (128+256, 128), (128+256, 128+256), (255, 0, 0), 2)
            cv2.line(rgb_context, (128, 128+256), (128+256, 128+256), (255, 0, 0), 2)
            cv2.line(rgb_context, (128, 128), (128, 128+256), (255, 0, 0), 2)

        fig, axarr = plt.subplots(1, 2)
        fig.set_figheight(16)
        fig.set_figwidth(16)
        axarr[0].imshow(rgb_context)
        axarr[1].imshow(crop)
        plt.show()
        """

        return crop


    def __getitem__(self, index):

        start_time = time.time()

        # Source and parse input files
        trajectory = list(self.vehicles_traj_dict)[index]
        trajectory = self.vehicles_traj_dict[trajectory]

        ts = [t[0] for t in trajectory]
        xs = [t[1] for t in trajectory]
        ys = [t[2] for t in trajectory]

        rgb_file = self.bev_rgb_files[ts[0]]
        sem_file = self.bev_semantic_files[ts[0]]
        bev_lidar_file = self.bev_lidar_files[ts[0]]

        rgb = np.asarray(Image.open(rgb_file))
        sem = np.asarray(Image.open(sem_file))
        lidar = np.asarray(Image.open(bev_lidar_file))

        # Transform xs, ys to 512x512 image coordinates
        xs = [int(x*2+256) for x in xs]
        ys = [int(y*2+256) for y in ys]


        # resize images
        rgb_context = cv2.resize(rgb, (512, 512))
        rgb = np.ascontiguousarray(rgb_context[128:128+256, 128:128+256])
        sem = cv2.resize(sem, (512, 512))
        lidar = cv2.resize(lidar, (512, 512))
        sdf_context = np.zeros((512, 512), dtype=np.float32)


        drivable = (sem == 7).astype(np.float32)
        non_drivable_mask = drivable > 0.5


        gt_lines = []
        gt_multilines = []

        for i in range(len(ts)-1):
            line = [xs[i], ys[i], xs[i+1], ys[i+1]]
            gt_lines.append(line)
            gt_multilines.append(((xs[i], ys[i]), (xs[i+1], ys[i+1])))
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
        drivable_distrib = drivable/np.sum(drivable)

        if self.gt_pointwise:
            if self.point_samping_method == "uniform":
                # Node sampling
                # Create a flat copy of the array & sample index from 1D array
                # with prob of the original array
                flat = drivable_distrib.flatten()
                sample_index = np.random.choice(a=flat.size, p=flat, size=self.num_node_samples)
                adjusted_index = np.unravel_index(sample_index, drivable_distrib.shape)
                point_coords = list(zip(*adjusted_index))

                # Append starting point as a node
                point_coords.append((255, 128))

            elif self.point_samping_method == "halton":

                point_coords = halton(2, 1000) * 255
                halton_points = point_coords.astype(np.int)

                # Filter self.points according to the non-drivable mask
                point_coords = []
                indicator = []
                for p in halton_points:
                    if not non_drivable_mask[p[0], p[1]]:
                        point_coords.append(p)
                        indicator.append(1)
                    else:
                        indicator.append(0)
                        point_coords.append(np.zeros(2))
                point_coords = np.array(point_coords).astype(np.int32)
                indicator = np.array(indicator)

                selected_indices = np.random.choice(a=np.arange(0, len(indicator)), p=indicator / np.sum(indicator),
                                                    size=self.num_node_samples)

                point_coords = point_coords[selected_indices]
                point_coords = np.concatenate((point_coords, np.array([[255, 128]])), axis=0)
            else:
                raise NotImplementedError("Point sampling method not implemented: ", self.point_samping_method)

        else: 
            # Filter self.points according to the non-drivable mask
            point_coords = []
            indicator = []
            for p in self.points:
                if not non_drivable_mask[p[0], p[1]]:
                    point_coords.append(p)
                    indicator.append(1)
                else:
                    indicator.append(0)
                    point_coords.append(np.zeros(2))
            point_coords = np.array(point_coords).astype(np.int32)
            indicator = np.array(indicator)

            selected_indices = np.random.choice(a=np.arange(0, len(indicator)), p=indicator/np.sum(indicator), size=self.num_node_samples)

            indicator = np.zeros_like(indicator)
            point_coords = point_coords[selected_indices]
            point_coords = np.concatenate((point_coords, np.array([[255, 128]])), axis=0)

            indicator[selected_indices] = 1

        triangulation_pairs = get_delaunay_triangulation(point_coords).tolist()

        # Triangulation based edge proposal generation
        edges = list()
        edges_locs = list()
        edge_gt_list = list()
        node_gt_list = list()
        node_feats = torch.empty([0, 2])

        for i, anchor in enumerate(point_coords):
            node_tensor = torch.tensor([anchor[0], anchor[1]]).reshape(1, -1)
            node_feats = torch.cat([node_feats, node_tensor], dim=0)
            shapely_point = Point([(anchor[1], anchor[0])])
            node_gt_score = shapely_point.distance(gt_multiline_shapely)
            node_gt_list.append(node_gt_score)

            for j, point in enumerate(point_coords):
                if [i, j] in triangulation_pairs and i != j:
                    if is_in_mask_loop(non_drivable_mask, anchor[1], anchor[0], point[1], point[0], self.N_interp):
                        edges_locs.append((anchor, point))
                        edges.append((i, j))

        # Min-max scaling of node_scores
        node_gt = torch.FloatTensor(node_gt_list)
        node_gt -= node_gt.min()
        node_gt /= node_gt.max()
        node_gt = 1 - node_gt

        # Scales edge img feature to VGG16 input size
        transform2vgg = T.Compose([
            T.ToPILImage(),
            T.Resize(32),
            T.ToTensor()])

        # Crop edge img feats and infer edge GT from SDF
        gt_sdf, angles_gt_dense = get_gt_sdf_with_direction(gt_lines_shapely)

        edge_attr_list = list()
        edge_img_feats_list = list()

        if self.gt_pointwise:
            cum_edge_dist_list = list()
            angle_penalty_list = list()

        for edge_idx, edge in enumerate(edges):
            i, j = edge
            s_x, s_y = point_coords[i][1], point_coords[i][0]
            e_x, e_y = point_coords[j][1], point_coords[j][0]

            delta_x, delta_y = e_x - s_x, e_y - s_y
            mid_x, mid_y = s_x + delta_x/2, s_y + delta_y/2

            edge_len = np.sqrt(delta_x**2 + delta_y**2)
            edge_angle = np.arctan(delta_y/(delta_x + 1e-6))

            edge_tensor = torch.tensor([edge_angle, edge_len, mid_x, mid_y]).reshape(1, -1)
            edge_attr_list.append(edge_tensor)

            crop_img_rgb = self.get_crop_mask_img(edge_angle, mid_x, mid_y, rgb_context)
            crop_img_rgb_resized = transform2vgg(crop_img_rgb).unsqueeze(0)
            crop_img_sdf = self.get_crop_mask_img(edge_angle, mid_x, mid_y, sdf_context)
            crop_img_sdf_resized = transform2vgg(crop_img_sdf).unsqueeze(0)

            if self.gt_pointwise:
                cum_edge_distance, angle_penalty = get_pointwise_edge_gt(s_x, s_y, e_x, e_y, self.N_interp, gt_multiline_shapely, angles_gt_dense)

                cum_edge_dist_list.append(cum_edge_distance)
                angle_penalty_list.append(angle_penalty)
            else:
                edge_gt_value = get_pred_distance_sdf(s_x, s_y, e_x, e_y, gt_sdf, angles_gt_dense, self.sdf_tensor, self.sdf_tensor_dict)
                edge_gt_list.append(edge_gt_value)

            # edge_img_feats_list.append(crop_img_resized)
            edge_img_feats_list.append(torch.cat([crop_img_rgb_resized, crop_img_sdf_resized], dim=1))

        edge_img_feats = torch.cat(edge_img_feats_list, dim=0)
        edge_attr = torch.cat(edge_attr_list, dim=0)

        # Pointwise edge score normalization
        if self.gt_pointwise:
            try:
                cum_edge_dist_gt = torch.FloatTensor(cum_edge_dist_list)
                cum_edge_dist_gt -= cum_edge_dist_gt.min()
                cum_edge_dist_gt /= cum_edge_dist_gt.max()
                cum_edge_dist_gt = 1 - cum_edge_dist_gt

                edge_gt = cum_edge_dist_gt * torch.FloatTensor(angle_penalty_list)

            except Exception as e:
                print(e)
                pass

        # SDF-based edge score normalization
        else:
            # Min-max scaling of edge scores
            try:
                edge_gt = torch.FloatTensor(edge_gt_list)
                edge_gt -= edge_gt.min()
                edge_gt /= edge_gt.max()
            except:
                pass

        gt_graph = torch.tensor(gt_lines) # [num_gt_graph_edges, 4]

        edges = torch.tensor(edges).long()

        data_time = time.time() - start_time

        print("node_feats", node_feats.shape)
        print("edge_index", edges.t().contiguous().shape)
        print("edge_attr", edge_attr.shape)
        print("node_gt", node_gt.t().contiguous().shape)
        print("edge_gt", edge_gt.t().contiguous().shape)
        # print("goalpoint_gt", goalpoint_gt.t().contiguous().shape)

        data = torch_geometric.data.Data(x=node_feats,
                                         edge_index=edges.t().contiguous(),
                                         edge_attr=edge_attr,
                                         edge_img_feats=edge_img_feats,
                                         node_gt=node_gt.t().contiguous(),
                                         edge_gt=edge_gt.t().contiguous(),
                                         edge_len=torch.tensor(len(edge_gt)),
                                         gt_graph=gt_graph,
                                         num_nodes=node_feats.shape[0],
                                         batch_idx=torch.tensor(index),
                                         rgb=torch.FloatTensor(rgb / 255.),
                                         data_time=torch.tensor(data_time),
                                         )
        return data
