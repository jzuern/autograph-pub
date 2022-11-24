import numpy as np
import torch
import os
from glob import glob
from PIL import Image
import cv2
import time
import torch_geometric.data.dataset
import torchvision.transforms as T
from shapely.geometry import LineString, MultiLineString, Point
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from svgpathtools import svg2paths
import scipy



from lanegnn.utils import is_in_mask_loop, get_gt_sdf_with_direction, \
    get_pointwise_edge_gt, get_delaunay_triangulation, halton, \
    get_random_edges, get_crop_mask_img, get_node_endpoint_gt, poisson_disk_sampling


class TrajectoryDatasetCarla(torch_geometric.data.Dataset):

    def __init__(self, path, params, num_node_samples: int = 500, gt_pointwise: bool = True, N_interp: int = 10):
        super(TrajectoryDatasetCarla, self).__init__(path)
        self.params = params
        self.path = path

        self.preprocessed_fname = os.path.join(self.path, "preprocessed/{:05d}.pth")

        self.bev_rgb_files = sorted(glob(path + '/bev_rgb/*.png'))
        self.bev_semantic_files = sorted(glob(path + '/bev_sem/*.png'))
        self.bev_lidar_files = sorted(glob(path + '/lidar_render/*.png'))

        self.num_node_samples = num_node_samples
        self.gt_pointwise = gt_pointwise
        self.N_interp = N_interp

        self.point_samping_method = "poisson"  # Either uniform or halton or poisson

        # get vehicle_pos_files
        pos_files = glob(os.path.join(path, "vehicles_pos", "*.txt"))

        # generate trajectories from vehicle_pos_files
        self.traj_dict = {}

        for pos_file in pos_files:
            with open(pos_file, "r") as f:
                time_step = int(pos_file.split("/")[-1].split(".")[0])
                lines = f.readlines()
                for line in lines:
                    vehicle_id, x, y, z = line.strip().split(",")

                    x = float(x) * 30 + 512
                    y = float(y) * 30 + 512

                    if vehicle_id not in self.traj_dict:
                        self.traj_dict[vehicle_id] = [[time_step, float(x), float(y), float(z)]]
                    else:
                        self.traj_dict[vehicle_id].append([time_step, float(x), float(y), float(z)])

        # sort vehicle dict by time_step
        for vehicle_id in self.traj_dict:
            self.traj_dict[vehicle_id] = sorted(self.traj_dict[vehicle_id], key=lambda x: x[0])



    def __len__(self):
        return len(self.traj_dict)


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


    def __getitem__(self, index):

        start_time = time.time()

        # Source and parse input files
        # generating more than one trajectory per sample

        #indices = [index]  # single trajectory
        indices = []
        for i in range(10):
            indices.append(np.random.randint(0, len(self.traj_dict)))
        trajectories = []
        for index in indices:
            t = list(self.traj_dict)[index]
            trajectories.append(self.traj_dict[t])
        trajectories = np.stack(trajectories)
        print(trajectories.shape)


        ts = np.array([t[0] for t in trajectories])
        xs = np.array([t[1] for t in trajectories])
        ys = np.array([t[2] for t in trajectories])

        if ts[0] > len(self.bev_rgb_files):
            return "empty-trajectory"

        # rgb_file = self.bev_rgb_files[ts[0]]
        rgb_file = self.bev_rgb_files[0]
        sem_file = self.bev_semantic_files[ts[0]]
        bev_lidar_file = self.bev_lidar_files[ts[0]]

        rgb = np.asarray(Image.open(rgb_file))
        sem = np.asarray(Image.open(sem_file))
        lidar = np.asarray(Image.open(bev_lidar_file))

        # resize images
        resize_factor = 256 / rgb.shape[0]
        rgb = cv2.resize(rgb, (256, 256))
        rgb_context = rgb.copy()
        sem = cv2.resize(sem, (256, 256))
        lidar = cv2.resize(lidar, (256, 256))

        context_regr_smooth = np.zeros((256, 256), dtype=np.float32)
        drivable = (sem == 6).astype(np.float32) + (sem == 7).astype(np.float32) + (sem == 10).astype(np.float32)
        non_drivable_mask = drivable < 0.5

        xs = xs * resize_factor
        ys = ys * resize_factor


        x_filter = np.logical_and(xs > 0, xs < 256)
        y_filter = np.logical_and(ys > 0, ys < 256)
        filter = np.logical_and(x_filter, y_filter)

        xs = xs[filter]
        ys = ys[filter]
        ts = np.array(ts)[filter]

        if len(xs) < 2:
            return "empty-trajectory"

        # GT graph representation

        waypoints = []
        relation_labels = []
        for i in range(len(xs)):
            waypoints.append([xs[i], ys[i]])
        for i in range(len(xs) - 1):
            relation_labels.append([i, i+1])

        # Get 1 graph start node and N graph end nodes
        G_gt_nx = nx.DiGraph()
        for e in relation_labels:
            if not G_gt_nx.has_node(e[0]):
                G_gt_nx.add_node(e[0], pos=waypoints[e[0]])
            if not G_gt_nx.has_node(e[1]):
                G_gt_nx.add_node(e[1], pos=waypoints[e[1]])
            G_gt_nx.add_edge(e[0], e[1])


        start_node = [x for x in G_gt_nx.nodes() if G_gt_nx.in_degree(x) == 0 and G_gt_nx.out_degree(x) > 0][0]
        start_node_pos = G_gt_nx.nodes[start_node]['pos']
        end_nodes = [x for x in G_gt_nx.nodes() if G_gt_nx.out_degree(x) == 0 and G_gt_nx.in_degree(x) > 0]
        end_node_pos_list = [G_gt_nx.nodes[x]['pos'] for x in end_nodes]


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

        if self.params.preprocessing.gt_pointwise:
            # ----- DOES NOT WORK RIGHT NOW: Coordinates greater than 255 are created
            if self.params.preprocessing.sampling_method == "uniform":
                # Node sampling
                # Create a flat copy of the array & sample index from 1D array
                # with prob of the original array
                flat = drivable_distrib.flatten()
                sample_index = np.random.choice(a=flat.size, p=flat, size=self.params.preprocessing.num_node_samples)
                adjusted_index = np.unravel_index(sample_index, drivable_distrib.shape)
                point_coords = list(zip(*adjusted_index))

                # Append starting point as a node
                point_coords.append((255, 128))

            elif self.params.preprocessing.sampling_method == "halton":

                # Single-level halton sampling

                point_coords = halton(2, self.params.preprocessing.num_node_samples - 1) * 255
                halton_points = point_coords.astype(np.int32)

                # filter all points where non_drivable_mask is True
                point_coords = halton_points[
                    np.logical_not(non_drivable_mask[halton_points[:, 0], halton_points[:, 1]])]

                point_coords = np.concatenate((point_coords, np.array([[255, 128]])), axis=0)

        else:
            print("SDF-wise edge GT not implemented")

        if self.params.preprocessing.visualize:
            # Plot non-drivable mask
            plt.cla()
            plt.imshow(non_drivable_mask)
            # Plot node positions
            for i in point_coords:
                plt.scatter(i[1], i[0], c='red', s=6.0)
            plt.savefig(self.params.paths.home + "trash/figprint.png")

        # print("--edge construction")

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

        # Now we search for shortest path through the G_proposal_nx from start node to end nodes
        point_coords_swapped = np.array(point_coords)[:, ::-1]
        start_node_idx = np.argmin(np.linalg.norm(point_coords_swapped - start_node_pos, axis=1))
        end_node_idx_list = [np.argmin(np.linalg.norm(point_coords_swapped - end_node_pos, axis=1)) for end_node_pos in
                             end_node_pos_list]

        # Find shortest path
        try:
            shortest_paths = [nx.shortest_path(G_proposal_nx, start_node_idx, end_node_idx, weight="weight") for
                              end_node_idx in end_node_idx_list]
            dijkstra_edge_list = list()
            for path in shortest_paths:
                dijkstra_edge_list += list(zip(path[:-1], path[1:]))
        except nx.NetworkXNoPath as e:
            print(e)
            dijkstra_edge_list = list()

        # Now we correct the edge weights according to dijsktra path
        edge_gt_score_dijkstra = np.zeros_like(edge_gt_score)
        for idx in range(len(edge_idx_list)):
            e = edge_idx_list[idx]
            if (e[0], e[1]) in dijkstra_edge_list:
                edge_gt_score_dijkstra[idx] = 1.0
                edge_gt_score[idx] = 1.0


        edge_gt_score = torch.from_numpy(edge_gt_score).float()
        edge_gt_score_dijkstra = torch.from_numpy(edge_gt_score_dijkstra).float()

        gt_graph = torch.tensor(gt_lines)  # [num_gt_graph_edges, 4]
        edges = torch.tensor(edges)



        data = torch_geometric.data.Data(x=node_feats,
                                         edge_index=edges.t().contiguous(),
                                         edge_attr=edge_attr,
                                         edge_img_feats=edge_img_feats,
                                         node_distance=node_gt_score.t().contiguous(),
                                         edge_distance=edge_gt_score.t().contiguous(),
                                         edge_dijkstra=edge_gt_score_dijkstra.t().contiguous(),
                                         gt_graph=gt_graph,
                                         num_nodes=node_feats.shape[0],
                                         batch_idx=torch.tensor(len(gt_graph)),
                                         rgb=torch.FloatTensor(rgb / 255.), # [0.0, 1.0]
                                         rgb_context=torch.FloatTensor(rgb_context / 255.), # [0.0, 1.0]
                                         context_regr_smooth=torch.FloatTensor(context_regr_smooth), # [0.0, 1.0]
                                         data_time=torch.tensor(time.time() - start_time),
                                         )

        return data




class TrajectoryDatasetIND(torch_geometric.data.Dataset):

    def __init__(self, path, params, num_node_samples: int = 500, gt_pointwise: bool = True, N_interp: int = 10):
        super(TrajectoryDatasetIND, self).__init__(path)
        self.params = params
        self.path = path
        self.scenario = 20

        self.scaling_factor = {
            1: 10,
            4: 6.55,
            20: 10.3,
        }

        print("Loading data from {}".format(path))

        self.preprocessed_fname = os.path.join(self.path, "preprocessed/{:05d}.pth")

        self.bev_rgb_file = path + '/{:02d}_background.png'.format(self.scenario)
        self.bev_rgb = np.array(Image.open(self.bev_rgb_file))
        self.tracks_file = path + '/{:02d}_tracks.csv'.format(self.scenario)
        self.meta_file = path + '/{:02d}_tracksMeta.csv'.format(self.scenario)

        self.num_node_samples = num_node_samples
        self.gt_pointwise = gt_pointwise
        self.N_interp = N_interp

        self.point_samping_method = "poisson"  # Either uniform or halton


        # generate trajectories from vehicle_pos_files
        self.traj_dict = self.get_tracks_dict()

        # sort vehicle dict by time_step
        to_delete = []
        for vehicle_id in self.traj_dict:
            self.traj_dict[vehicle_id] = sorted(self.traj_dict[vehicle_id], key=lambda x: x[0])
            self.traj_dict[vehicle_id] = np.array(self.traj_dict[vehicle_id])

            max_step_size = np.max(np.linalg.norm(np.diff(self.traj_dict[vehicle_id][:, 1:3], axis=0), axis=1))
            if max_step_size > 10:
                to_delete.append(vehicle_id)
            else:
                self.traj_dict[vehicle_id] = self.traj_dict[vehicle_id][::20]

        self.traj_dict = {k: v for k, v in self.traj_dict.items() if k not in to_delete}




        # # get svg data
        # paths, attributes = svg2paths('/data/self-supervised-graph/inD/tracklet-annotation.svg')
        #
        # # print(paths, attributes)
        # vehicle_id = 0
        # self.traj_dict = dict()
        # for k, v in enumerate(attributes):
        #     line = v['d']
        #     coordinates_x = [float(x.split(",")[0]) for x in line.split(" ")[1:]]
        #     coordinates_y = [float(x.split(",")[1]) for x in line.split(" ")[1:]]
        #     coordinates_x = np.array(coordinates_x) / 309 * self.bev_rgb.shape[1]
        #     coordinates_y = np.array(coordinates_y) / 206 * self.bev_rgb.shape[0]
        #     coordinates_x = coordinates_x * 2
        #     coordinates_y = coordinates_y * 2
        #
        #     self.traj_dict[vehicle_id] = np.array(list([float(x), float(y)] for x, y in zip(coordinates_x, coordinates_y)))
        #     vehicle_id += 1

        self.traj_dict = dict()
        self.traj_dict[0] = np.array([[120, 400], [400, 330], [960, 240]])
        self.traj_dict[1] = np.array([[400, 150], [460, 380], [545, 568]])
        self.traj_dict[2] = np.array([[400, 150], [450, 220], [560, 290], [960, 240]])



        # plot image and trajectories
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        plt.imshow(self.bev_rgb)
        for vehicle_id in self.traj_dict:
            vehicle_pos = self.traj_dict[vehicle_id]
            random_color = np.random.rand(3, )
            for i in range(0, len(vehicle_pos) - 1):
                ax.arrow(vehicle_pos[i, 0], vehicle_pos[i, 1],
                         vehicle_pos[i + 1, 0] - vehicle_pos[i, 0],
                         vehicle_pos[i + 1, 1] - vehicle_pos[i, 1],
                         head_width=5, head_length=5, fc=random_color, ec=random_color)
        plt.show()

    def get_tracks_dict(self):

        meta = pd.read_csv(self.meta_file)
        tracks = pd.read_csv(self.tracks_file, index_col=0)
        recordingId = meta['recordingId'].values[0]
        tracks_xy = tracks[['xCenter', 'yCenter']].values
        tracks_xy[:, 1] = -tracks_xy[:, 1]
        tracks_xy *= self.scaling_factor[recordingId]

        tracks_id = np.array(tracks[['trackId']].values).flatten()
        tracks_id_unique = np.unique(tracks_id)
        tracks_dict = {}

        track_classes = meta['class'].values

        for t, cls in zip(tracks_id_unique, track_classes) :
            if cls != 'car':
                continue
            track = tracks_xy[tracks_id == t]
            tracks_dict[t] = track

        return tracks_dict


    def __len__(self):
        #return len(self.traj_dict)
        return len(self.traj_dict) * 20


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


    def __getitem__(self, index):

        start_time = time.time()

        #indices = [index % len(self.traj_dict)]  # single trajectory
        indices = []
        for i in range(len(self.traj_dict)):
            indices.append(i)  # all trajectories


        trajectories = []
        traj_indices = []
        for index in indices:
            t = list(self.traj_dict)[index]
            traj_indices += [t] * (len(self.traj_dict[t]) - 1)
            trajectories.append(self.traj_dict[t])
        trajectories = np.concatenate(trajectories, axis=0)

        xs_, ys_ = trajectories[:, 0], trajectories[:, 1]
        interp = scipy.interpolate.interp1d(xs_, ys_)
        xs = np.linspace(xs_[0], xs_[-1], 5)
        ys = interp(xs)


        # resize images
        resize_factor = 1
        rgb = cv2.resize(self.bev_rgb, fx=resize_factor, fy=resize_factor, dsize=None, interpolation=cv2.INTER_CUBIC)
        rgb_context = rgb.copy()
        context_regr_smooth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        drivable = rgb[:, :, 0] > 0
        non_drivable_mask = drivable < 0.5

        xs = xs * resize_factor
        ys = ys * resize_factor

        # add noise
        xs = xs + np.random.normal(0, 2, len(xs))
        ys = ys + np.random.normal(0, 2, len(ys))

        x_filter = np.logical_and(xs > 0, xs < rgb.shape[1])
        y_filter = np.logical_and(ys > 0, ys < rgb.shape[0])
        filter = np.logical_and(x_filter, y_filter)

        xs = xs[filter]
        ys = ys[filter]

        if len(xs) < 2:
            return "empty-trajectory"

        # GT graph representation
        print('test')

        waypoints = []
        relation_labels = []
        for i in range(len(xs)):
            waypoints.append([xs[i], ys[i]])
        for i in range(len(xs) - 1):
            relation_labels.append([i, i+1])

        # Get 1 graph start node and N graph end nodes
        G_gt_nx = nx.DiGraph()
        for e in relation_labels:
            if not G_gt_nx.has_node(e[0]):
                G_gt_nx.add_node(e[0], pos=waypoints[e[0]])
            if not G_gt_nx.has_node(e[1]):
                G_gt_nx.add_node(e[1], pos=waypoints[e[1]])
            G_gt_nx.add_edge(e[0], e[1])


        start_node = [x for x in G_gt_nx.nodes() if G_gt_nx.in_degree(x) == 0 and G_gt_nx.out_degree(x) > 0][0]
        start_node_pos = G_gt_nx.nodes[start_node]['pos']
        end_nodes = [x for x in G_gt_nx.nodes() if G_gt_nx.out_degree(x) == 0 and G_gt_nx.in_degree(x) > 0]
        end_node_pos_list = [G_gt_nx.nodes[x]['pos'] for x in end_nodes]


        gt_lines = []
        gt_multilines = []

        for i in range(len(xs)-1):
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

        if self.params.preprocessing.gt_pointwise:
            # ----- DOES NOT WORK RIGHT NOW: Coordinates greater than 255 are created
            if self.params.preprocessing.sampling_method == "uniform":
                # Node sampling
                # Create a flat copy of the array & sample index from 1D array
                # with prob of the original array
                flat = drivable_distrib.flatten()
                sample_index = np.random.choice(a=flat.size, p=flat, size=self.params.preprocessing.num_node_samples)
                adjusted_index = np.unravel_index(sample_index, drivable_distrib.shape)
                point_coords = list(zip(*adjusted_index))

            elif self.params.preprocessing.sampling_method == "halton":

                # Single-level halton sampling

                halton_points = halton(2, self.params.preprocessing.num_node_samples)
                halton_points[:, 0] = halton_points[:, 0] * rgb.shape[0]
                halton_points[:, 1] = halton_points[:, 1] * rgb.shape[1]
                halton_points = halton_points.astype(np.int32)

                # filter all points where non_drivable_mask is True
                point_coords = halton_points[np.logical_not(non_drivable_mask[halton_points[:, 0], halton_points[:, 1]])]


            elif self.params.preprocessing.sampling_method == "poisson":

                # Single-level halton sampling

                poisson_points = poisson_disk_sampling(r_min=20,
                                                       width=non_drivable_mask.shape[0],
                                                       height=non_drivable_mask.shape[1])
                poisson_points = np.array(poisson_points).astype(np.int32)

                # filter all points where non_drivable_mask is True
                point_coords = poisson_points[np.logical_not(non_drivable_mask[poisson_points[:, 0], poisson_points[:, 1]])]

        else:
            print("SDF-wise edge GT not implemented")

        if self.params.preprocessing.visualize:
            # Plot non-drivable mask
            plt.cla()
            plt.imshow(non_drivable_mask)
            # Plot node positions
            for i in point_coords:
                plt.scatter(i[1], i[0], c='red', s=6.0)
            plt.show()

        print("--edge construction")

        # Construct edges based on obstacle rejection
        if self.params.preprocessing.edge_proposal_method == 'triangular':
            edge_proposal_pairs = get_delaunay_triangulation(point_coords)
        elif self.params.preprocessing.edge_proposal_method == 'random':
            edge_proposal_pairs = get_random_edges(point_coords, min_point_dist=13, max_point_dist=15)

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

        print("--normalize node gt")

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

        # Now we search for shortest path through the G_proposal_nx from start node to end nodes
        point_coords_swapped = np.array(point_coords)[:, ::-1]
        start_node_idx = np.argmin(np.linalg.norm(point_coords_swapped - start_node_pos, axis=1))
        end_node_idx_list = [np.argmin(np.linalg.norm(point_coords_swapped - end_node_pos, axis=1)) for end_node_pos in
                             end_node_pos_list]

        # # plot the graph
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.imshow(rgb)
        # p = np.array([point_coords[i] for i in range(len(point_coords))])
        # nx.draw_networkx(G_proposal_nx, pos=p, ax=ax, with_labels=False, node_size=1)
        # plt.show()

        # Find shortest path
        try:
            shortest_paths = [nx.shortest_path(G_proposal_nx, start_node_idx, end_node_idx, weight="weight") for
                              end_node_idx in end_node_idx_list]
            dijkstra_edge_list = list()
            for path in shortest_paths:
                dijkstra_edge_list += list(zip(path[:-1], path[1:]))
        except Exception as e:
            print(e)
            return "empty-trajectory"
            #dijkstra_edge_list = list()

        # Now we correct the edge weights according to dijsktra path
        edge_gt_score_dijkstra = np.zeros_like(edge_gt_score)
        for idx in range(len(edge_idx_list)):
            e = edge_idx_list[idx]
            if (e[0], e[1]) in dijkstra_edge_list:
                edge_gt_score_dijkstra[idx] = 1.0
                edge_gt_score[idx] = 1.0


        edge_gt_score = torch.from_numpy(edge_gt_score).float()
        edge_gt_score_dijkstra = torch.from_numpy(edge_gt_score_dijkstra).float()

        gt_graph = torch.tensor(gt_lines)  # [num_gt_graph_edges, 4]
        edges = torch.tensor(edges)

        data = torch_geometric.data.Data(node_feats=node_feats,
                                         edge_indices=edges.t().contiguous(),
                                         edge_pos_feats=edge_attr,
                                         edge_img_feats=edge_img_feats,
                                         node_scores=node_gt_score.t().contiguous(),
                                         #edge_distance=edge_gt_score.t().contiguous(),
                                         edge_scores=edge_gt_score_dijkstra.t().contiguous(),
                                         graph=gt_graph,
                                         num_nodes=node_feats.shape[0],
                                         batch_idx=torch.tensor(len(gt_graph)),
                                         rgb=torch.FloatTensor(rgb / 255.), # [0.0, 1.0]
                                         rgb_context=torch.FloatTensor(rgb_context / 255.), # [0.0, 1.0]
                                         context_regr_smooth=torch.FloatTensor(context_regr_smooth), # [0.0, 1.0]
                                         data_time=torch.tensor(time.time() - start_time),
                                         )

        return data



class PreprocessedTrajectoryDataset(torch_geometric.data.Dataset):

    def __init__(self, path):
        super(PreprocessedTrajectoryDataset, self).__init__(path)

        self.path = path
        self.pth_files = sorted(glob(path + '/*.pt'))

    def __len__(self):
        return len(self.pth_files)

    def __getitem__(self, index):

        fname = self.pth_files[index]
        data = torch.load(fname)[0]

        return data


