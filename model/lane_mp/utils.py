import numpy as np
import skfmm
import cv2
import numpy as np
import yaml
import os
import sys
import cv2
import matplotlib.pyplot as plt
import json
import codecs
from shapely.geometry import LineString, Point
from scipy.interpolate import griddata
import time
import argparse
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import networkx as nx
from torch_geometric.utils import degree
import torch
import torchvision
from collections import defaultdict



def unbatch(src, batch, dim: int = 0):
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)

def unbatch_edge_index(edge_index, batch):
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)



class ParamNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def overwrite(self, args: argparse.Namespace):
        for k, v in vars(args).items():
            if k in self.__dict__.keys() and v is not None:
                self.__dict__[k] = v

class ParamLib:
    def __init__(self, config_path: str):
        self.config_path = config_path

        # Create all parameter dictionaries
        self.main = ParamNamespace()
        self.paths = ParamNamespace()
        self.preprocessing = ParamNamespace()
        self.model = ParamNamespace()
        self.driving = ParamNamespace()

        # Load config file with parametrization, create paths and do sys.path.inserts
        self.load_config_file(self.config_path)
        #self.create_dir_structure()
        self.add_system_paths()

    def load_config_file(self, path: str):
        """
        Loads a config YAML file and sets the different dictionaries.
        Args:
            path: path to some configuration file in yaml format

        Returns:
        """

        with open(path, 'r') as stream:
            try:
                config_file = yaml.safe_load(stream)
            except yaml.YAMLError as exception:
                print(exception)

        # Copy yaml content to the different dictionaries.
        vars(self.main).update(config_file['main'])
        vars(self.paths).update(config_file['paths'])
        vars(self.preprocessing).update(config_file['preprocessing'])
        vars(self.model).update(config_file['model'])
        vars(self.driving).update(config_file['driving'])

        # Set some secondary paths that are important
        if self.main.dataset == "paloalto":
            pass
            # print("using palo alto")
            # paths to preprocessed data
            #self.paths.preprocessed_data = os.path.join(self.paths.home, self.main.dataset, 'preprocessed/')

        else:
            raise NotImplementedError

    def create_dir_structure(self):
        """
        Loops through the paths dictionary in order to create
        the paths if they do not exist.
        Args:
            paths_dict: some para

        Returns:
            -
        """
        for name, path in vars(self.paths).items():
            # exclude all paths to files
            if len(path.split('.')) == 1:
                if not os.path.exists(path):
                    os.makedirs(path)

    def add_system_paths(self):
        """
        Loops through the paths dictionary in order to create
        the paths if they do not exist.
        Args:
            paths_dict: some para

        Returns:
            -
        """
        sys.path.insert(0, self.paths.package)
        #sys.path.insert(0, os.path.join(self.paths.package, 'utils'))

def get_crop_mask_img(x,y,angle, rgb_context):
    mid_x = x
    mid_y = y

    crop_size = 160

    angle = torch.tensor(angle) * (180 / torch.pi)  # .cuda()
    angle_rad = angle * (torch.pi / 180)  # .cuda()

    src_pts = np.array([[-crop_size // 2, -crop_size // 2],
                        [-crop_size // 2, crop_size // 2],
                        [crop_size // 2, crop_size // 2],
                        [crop_size // 2, -crop_size // 2]])

    center = np.array([[mid_x.item(), mid_y.item()]], dtype="float32")
    # Rotate source points
    R = np.array([[np.cos(angle_rad.item()), -np.sin(angle_rad.item())],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    src_pts = np.matmul(R, src_pts.T).T + center
    src_pts = src_pts.astype(np.float32)

    context_size = rgb_context.shape[1]

    dst_pts = np.array([[0, context_size],
                        [0, 0],
                        [context_size, 0],
                        [context_size, context_size]],
                       dtype="float32")

    src_pts = torch.from_numpy(src_pts).contiguous()  # .cuda()
    dst_pts = torch.from_numpy(dst_pts).contiguous()  # .cuda()

    new_img = torchvision.transforms.functional.perspective(rgb_context, src_pts, dst_pts)
    new_img = torchvision.transforms.Resize(size=(48, 48))(new_img)
    new_img = torchvision.transforms.functional.hflip(new_img)
    return new_img

def get_crop_mask_img(edge_angle, mid_x, mid_y, rgb_context):
    # Size of quadratic destination image
    crop_size = 100
    imsize = 512

    crop_rad = edge_angle
    crop_x = mid_x + rgb_context.shape[1] // 4
    crop_y = mid_y + rgb_context.shape[1] // 4
    center = np.array([crop_x, crop_y])

    # Source point coordinates already in coordinate system around center point of future crop
    src_pts = np.array([[-crop_size // 2, crop_size // 2 - 1],
                        [-crop_size // 2, -crop_size // 2],
                        [crop_size // 2 - 1, -crop_size // 2],
                        [crop_size // 2 - 1, crop_size // 2 - 1]])

    # Rotate source points
    R = np.array([[np.cos(crop_rad), -np.sin(crop_rad)],
                  [np.sin(crop_rad), np.cos(crop_rad)]])
    src_pts = np.matmul(R, src_pts.T).T + center
    src_pts = src_pts.astype(np.float32)

    # Destination points are simply the corner points in the new image
    dst_pts = np.array([[0, crop_size - 1],
                        [0, 0],
                        [crop_size - 1, 0],
                        [crop_size - 1, crop_size - 1]],
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




def is_in_mask_loop(mask, x1, y1, x2, y2, N_interp):
    # Interpolate N times between the two points
    for j in range(N_interp):
        # Interpolate between the two points
        x = int((x1 + (float(j) / N_interp) * (x2 - x1)))
        y = int((y1 + (float(j) / N_interp) * (y2 - y1)))

        # Check if the point is inside the mask
        if mask[y, x]:
            return False

    return True

def make_sdf(mask):

    dx = 2 # downsample factor
    f = 20 / dx  # distance function scale

    try:
        mask = cv2.resize(mask, None, fx=1/dx, fy=1/dx, interpolation=cv2.INTER_NEAREST)
        sdf = skfmm.distance(1 - mask)
        sdf[sdf > f] = f
        sdf = sdf / f
        sdf = 1 - sdf
        sdf = cv2.resize(sdf, None, fx=dx, fy=dx, interpolation=cv2.INTER_NEAREST)
    except:
        sdf = np.zeros_like(mask)
        # plt.imshow(mask, cmap='gray')
        # plt.imshow(sdf, cmap='viridis', alpha=0.5)
        # plt.show()

    return sdf


def color_to_angle(color):

    m0 = (color[:, :, 0] != 0)
    m1 = (color[:, :, 1] != 0)
    m2 = (color[:, :, 2] != 0)

    m = m0 | m1 | m2

    # expects color in float [0, 1]
    c = color[:, :, 0] / 2. - 0.5
    s = color[:, :, 1] / 2. - 0.5

    angle = m * np.arctan2(s, c)

    return angle

def angle_to_color(angle):

    c = 0.5 + np.array([np.cos(angle), np.sin(angle), np.zeros_like(angle)]) / 2
    c = np.transpose(c, [1, 2, 0])

    plt.imshow(c)
    plt.show()


    return c


def get_nn_direction(angles, angles_mask):

    # Get dense depth using griddata
    dir_pixels = np.where(angles_mask != 0)
    xs = dir_pixels[1]
    ys = dir_pixels[0]
    points = np.vstack((xs, ys)).T

    # Extrapolate
    grid_x, grid_y = np.mgrid[0:256, 0:256]

    try:
        dense_dir = griddata(points, angles[dir_pixels], (grid_y, grid_x), method='nearest')
    except:
        print("Error in griddata")
        dense_dir = np.zeros_like(angles)
        print(dir_pixels)

    return dense_dir


# def get_distance_metric_sdf(x1, y1, x2, y2, gt_lines_shapely):
#
#     visualize = True
#     # Generate pred sdf mask
#     pred_mask = np.zeros([256, 256])
#     if visualize:
#         cv2.line(pred_mask, (x1, y1), (x2, y2), color=1, thickness=2)
#     pred_sdf = make_sdf(pred_mask)
#
#     # Generate GT mask
#     gt_mask = np.zeros([256, 256])
#     if visualize:
#         for gt_line in gt_lines_shapely:
#             cv2.line(gt_mask, (int(gt_line.coords[0][0]), int(gt_line.coords[0][1])), (int(gt_line.coords[1][0]), int(gt_line.coords[1][1])), color=1, thickness=2)
#
#     gt_sdf = make_sdf(gt_mask)
#
#     if visualize:
#         fig, axarr = plt.subplots(1, 3)
#         axarr[0].imshow(gt_sdf * pred_sdf)
#         axarr[1].imshow(pred_sdf)
#         axarr[2].imshow(gt_sdf)
#         plt.show()
#
#     return np.sum(gt_sdf * pred_sdf) / np.sum(pred_sdf)



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

    node_endpoint_gt = torch.zeros(len(node_feats), dtype=torch.float32)

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




def get_gt_sdf_with_direction(gt_lines_shapely):

    # Get dense Pred angles using griddata
    angles_gt = np.zeros([256, 256])
    angles_gt_mask = np.zeros([256, 256])

    for gt_line in gt_lines_shapely:
        gt_x1 = int(gt_line.coords[0][0])
        gt_y1 = int(gt_line.coords[0][1])
        gt_x2 = int(gt_line.coords[1][0])
        gt_y2 = int(gt_line.coords[1][1])

        dir_rad = np.arctan2(float(gt_y2 - gt_y1), float(gt_x2 - gt_x1))
        cv2.line(angles_gt, (gt_x1, gt_y1), (gt_x2, gt_y2), color=dir_rad, thickness=2)
        cv2.line(angles_gt_mask, (gt_x1, gt_y1), (gt_x2, gt_y2), color=1, thickness=2)

    # Get dense GT angles using griddata
    angles_gt_dense = get_nn_direction(angles_gt, angles_gt_mask)

    # Generate GT sdf mask
    gt_mask = np.zeros([256, 256])
    for gt_line in gt_lines_shapely:
        cv2.line(gt_mask, (int(gt_line.coords[0][0]), int(gt_line.coords[0][1])), (int(gt_line.coords[1][0]), int(gt_line.coords[1][1])), color=1, thickness=4)
    gt_sdf = make_sdf(gt_mask)

    return gt_sdf, angles_gt_dense


def get_pred_distance_sdf(x1, y1, x2, y2, gt_sdf, angles_gt_dense, sdf_tensor, sdf_tensor_dict):

    # Option FAST - make sdf from pre-computed tensor
    sdf_dict_key = (y1, x1, y2, x2)  # Wrong order, i know, but is correct that way
    # sdf_dict_key = (x1, y1, x2, y2)  # Wrong order, i know, but is correct that way
    if sdf_dict_key in sdf_tensor_dict:
        pred_sdf = sdf_tensor[sdf_tensor_dict[sdf_dict_key], :, :]
        pred_sdf = cv2.resize(pred_sdf.numpy(), (256, 256))
        pred_sdf = pred_sdf.T
        # TODO: Is this actuallyt the correct order?
    else:
        # Option 1 - make sdf live
        pred_mask = np.zeros([256, 256])
        cv2.line(pred_mask, (x1, y1), (x2, y2), color=1, thickness=2)
        pred_sdf = make_sdf(pred_mask)


    # Generate pred dir mask
    angles_pred = np.zeros([256, 256]).astype(np.float32)
    angles_pred_mask = np.zeros([256, 256]).astype(np.uint8)
    dir_rad = np.arctan2(float(y2 - y1), float(x2 - x1))

    cv2.line(angles_pred, (x1, y1), (x2, y2), color=dir_rad, thickness=2)
    cv2.line(angles_pred_mask, (x1, y1), (x2, y2), color=1, thickness=2)

    angles_pred_dense = dir_rad * np.ones([256, 256]).astype(np.float32)

    # fig, axarr = plt.subplots(1, 5)

    # relative angle

    angle_relative = np.abs(angles_pred_dense - angles_gt_dense)

    # force angle to be between 0 and pi
    angle_relative[angle_relative > np.pi] = 2 * np.pi - angle_relative[angle_relative > np.pi]

    # normalize between 0 and 1
    angle_relative_normalized = angle_relative / np.pi
    angle_goodness = 1 - angle_relative_normalized

    # axarr[0].imshow(angles_gt_dense)
    # axarr[1].imshow(angles_pred_dense)
    # axarr[2].imshow(angle_relative)
    # axarr[3].imshow(pred_sdf)
    # axarr[4].imshow(gt_sdf)
    # plt.show()

    # goodness = pred_sdf * gt_sdf * angle_goodness

    # fig, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(pred_sdf)
    # axarr[1].imshow(gt_sdf)
    # plt.show()



    # goodness = pred_sdf * gt_sdf
    goodness = pred_sdf * gt_sdf * angle_goodness
    # goodness = angle_goodness

    d = np.sum(goodness) / np.sum(pred_sdf)

    return d

def get_distance_metric_sdf_with_direction(x1, y1, x2, y2, gt_lines_shapely):

    visualize = False

    # Generate pred sdf mask
    pred_mask = np.zeros([256, 256])
    cv2.line(pred_mask, (x1, y1), (x2, y2), color=1, thickness=4)

    # plt.imshow(pred_mask)
    # plt.title('pred_mask')
    # plt.show()

    if visualize:
        cv2.imshow("pred_mask", pred_mask)

    pred_sdf = make_sdf(pred_mask)

    if visualize:
        cv2.imshow("pred_sdf", pred_sdf)


    # Generate pred dir mask
    angles_pred = np.zeros([256, 256]).astype(np.float32)
    angles_pred_mask = np.zeros([256, 256]).astype(np.uint8)
    dir_rad = np.arctan2(float(y2 - y1), float(x2 - x1))

    cv2.line(angles_pred, (x1, y1), (x2, y2), color=dir_rad, thickness=2)
    cv2.line(angles_pred_mask, (x1, y1), (x2, y2), color=1, thickness=2)

    angles_pred_dense = get_nn_direction(angles_pred, angles_pred_mask)

    # Generate GT sdf mask
    gt_mask = np.zeros([256, 256])
    for gt_line in gt_lines_shapely:
        cv2.line(gt_mask, (int(gt_line.coords[0][0]), int(gt_line.coords[0][1])), (int(gt_line.coords[1][0]), int(gt_line.coords[1][1])), color=1, thickness=4)
    gt_sdf = make_sdf(gt_mask)

    if visualize:
        cv2.imshow("gt_sdf", gt_sdf)


    # Get dense Pred angles using griddata
    angles_gt = np.zeros([256, 256])
    angles_gt_mask = np.zeros([256, 256])

    for gt_line in gt_lines_shapely:
        gt_x1 = int(gt_line.coords[0][0])
        gt_y1 = int(gt_line.coords[0][1])
        gt_x2 = int(gt_line.coords[1][0])
        gt_y2 = int(gt_line.coords[1][1])

        dir_rad = np.arctan2(float(gt_y2 - gt_y1), float(gt_x2 - gt_x1))
        cv2.line(angles_gt, (gt_x1, gt_y1), (gt_x2, gt_y2), color=dir_rad, thickness=1)
        cv2.line(angles_gt_mask, (gt_x1, gt_y1), (gt_x2, gt_y2), color=1, thickness=1)


    # plt.imshow(angles_gt_mask)
    # plt.title('angles_gt_mask')
    # plt.show()


    # Get dense GT angles using griddata
    angles_gt_dense = get_nn_direction(angles_gt, angles_gt_mask)

    # angles_gt_dense = cv2.GaussianBlur(angles_gt_dense, (25, 25), 0)

    # Calculate absolute value of scalar product between dense GT and Pred angles
    relative_angle_normalized = (np.abs(angles_gt_dense) - np.abs(angles_pred_dense)) / np.pi
    relative_angle_normalized = np.abs(relative_angle_normalized)
    relative_angle_normalized[relative_angle_normalized > 0.5] = 1 - relative_angle_normalized[relative_angle_normalized > 0.5]
    relative_angle_normalized *= 2  # normalize to [0, 1]

    goodness = pred_sdf * gt_sdf * (1 - relative_angle_normalized)

    if visualize:
        cv2.imshow("goodness", goodness)
        cv2.waitKey(-1)

    # fig, axarr = plt.subplots(1, 7)
    # axarr[0].imshow(angles_gt_dense, vmax=np.pi, vmin=-np.pi)
    # axarr[0].set_title('angles_gt_dense')
    # axarr[1].imshow(angles_pred_dense, vmax=np.pi, vmin=-np.pi)
    # axarr[1].set_title('angles_pred_dense')
    # axarr[2].imshow(relative_angle_normalized)
    # axarr[2].set_title('relative_angle normalized')
    # axarr[3].imshow(pred_sdf, vmax=1, vmin=0)
    # axarr[3].arrow(x1, y1, x2 - x1, y2 - y1, color='r', width=2, head_width=5)
    # axarr[3].set_title('pred_sdf')
    # axarr[4].imshow(gt_sdf, vmax=1, vmin=0)
    # axarr[4].set_title('gt_sdf')
    # axarr[5].imshow(pred_sdf * gt_sdf, vmax=1, vmin=0)
    # axarr[5].set_title('pred_sdf * gt_sdf')
    # axarr[6].imshow(penalty)
    # axarr[6].set_title('final penalty')
    # plt.show()

    d = np.sum(goodness) / np.sum(pred_sdf)

    return d


def get_pointwise_edge_gt(s_x, s_y, e_x, e_y, N_interp, gt_multiline_shapely, angles_gt_dense):
    
    interp_dist = list()
    interp_angle = list()

    angle_pred = np.arctan2(e_y - s_y, e_x - s_x)

    for j in range(N_interp+1):
        # Interpolate between the two points
        x = int((s_x + (float(j) / N_interp) * (e_x - s_x)))
        y = int((s_y + (float(j) / N_interp) * (e_y - s_y)))

        interp_point = Point([(x, y)])
        interp_dist.append(interp_point.distance(gt_multiline_shapely))


        angle_gt = angles_gt_dense[x, y]
        angle_relative = np.abs(angle_pred - angle_gt)

        # force angle to be between 0 and pi
        if angle_relative > np.pi:
            angle_relative = 2 * np.pi - angle_relative

        interp_angle.append(angle_relative)

    cum_edge_distance = np.array(interp_dist).mean()
    cum_angle_distance_normalized = np.array(interp_angle).mean() / np.pi

    angle_penalty = (1 - cum_angle_distance_normalized)

    return cum_edge_distance, angle_penalty


def graph_traversability_check(x1, y1, x2, y2, gt_lines_shapely, gt_graph_edge_index):

    pred_p1 = Point([(x1, y1)])
    pred_p2 = Point([(x2, y2)])

    d1s = []
    d2s = []

    for gt_idx, gt_line in gt_lines_shapely:
        d1 = pred_p1.distance(gt_line)
        d2 = pred_p2.distance(gt_line)
        d1s.append(d1)
        d2s.append(d2)

    d1_gt_idx = np.argmin(d1s)
    d2_gt_idx = np.argmin(d2s)

    # Check if the two points were mapped to the same line segment
    if d1_gt_idx is not d2_gt_idx:

        start, _ = gt_graph_edge_index[d1_gt_idx]
        _, target = gt_graph_edge_index[d2_gt_idx]
        
        path = bfs(graph, start, target)

        rej_edge_score = 0
        if path is None:
            # The two involved nodes do not allow a connection
            rej_edge_score = 0
        else:
            if len(path) <= 4:
                rej_edge_score = 1

    return rej_edge_score


def bfs(graph, start, end):
    # maintain a queue of paths
    queue = []
    # push the first path into the queue
    queue.append([start])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        # path found
        if node == end:
            return path
        # enumerate all adjacent nodes, construct a 
        # new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)

    return None


if __name__ == "__main__":

    grid_x, grid_y = np.mgrid[0:256, 0:256]

    root = '/data/lane-segmentation/woven-data/pao/agent/train/'

    # Define binary mask and upsample
    mask = 1 - (cv2.imread(root + "006_005_011-drivable.png", 0) > 0).astype(np.uint8)
    graph = json.loads(codecs.open(root + "006_005_011-targets.json", 'r', encoding='utf-8').read())

    waypoints = np.array(graph["bboxes"])
    relation_labels = np.array(graph["relation_labels"])

    gt_lines = []
    for l in relation_labels:
        line = [waypoints[l[0], 0], waypoints[l[0], 1], waypoints[l[1], 0], waypoints[l[1], 1]]
        gt_lines.append(line)

    gt_lines = np.array(gt_lines)

    gt_lines_shapely = []
    for l in gt_lines:
        x1 = l[0]
        y1 = l[1]
        x2 = l[2]
        y2 = l[3]
        gt_lines_shapely.append(LineString([(x1, y1), (x2, y2)]))



    N_samples = 300
    counter = 0
    N_interp = 10 


    while counter < N_samples:

        # Sample two random points
        x1, y1 = np.random.randint(50, 255-50, size=2)
        x2, y2 = [x1, y1] + np.random.randint(-50, 50, size=2)

        # x1, y1 = 200, 51 # Left
        # x2, y2 = 50, 50
        # x1, y1 = 128, 200  # Up
        # x2, y2 = 128, 50
        # x1, y1 = 128, 50 # Down
        # x2, y2 = 128, 200
        # x1, y1 = 50, 50  # Right
        # x2, y2 = 200, 51

        if is_in_mask_loop(mask, x1, y1, x2, y2):

            counter += 1
            # now we calculate the distance metric to GT graph

            try:
                now = time.time()
                # d = get_distance_metric_sdf(x1, y1, x2, y2)
                d = get_distance_metric_sdf_with_direction(x1, y1, x2, y2)
                print('Iteration: {} - Time: {:.3f}'.format(counter, time.time() - now))

                plt.arrow(x1, y1, x2 - x1, y2 - y1, color=plt.get_cmap('viridis')(d), head_width=5, head_length=5)
                plt.text(x1, y1, '{:.1f}'.format(d), color=plt.get_cmap('viridis')(d))
            except Exception as e:
                print(e)
                pass

    for l in gt_lines:
        plt.arrow(l[0], l[1], l[2] - l[0], l[3] - l[1], color='red', head_width=5, head_length=5)

    plt.show()



def get_delaunay_triangulation(points):

    tri = Delaunay(points)

    # convert simplices to index pairs
    simplices = []
    for s in tri.simplices:
        simplices.append([s[0], s[1]])
        simplices.append([s[1], s[0]])
        simplices.append([s[0], s[2]])
        simplices.append([s[2], s[0]])
        simplices.append([s[1], s[2]])
        simplices.append([s[2], s[1]])

    return np.array(simplices)


def get_random_edges(point_coords, min_point_dist=10, max_point_dist=50):

    # compute pairwise distances
    dist_mat = cdist(point_coords, point_coords)
    # print(dist_mat.shape)

    # find all pairs of points with a distance less than max_point_dist
    filter_matrix = (dist_mat < max_point_dist) * (dist_mat > min_point_dist)

    valid_edges = np.where(filter_matrix)
    valid_edges = np.array(list(zip(valid_edges[0], valid_edges[1])))
    valid_edges = np.unique(valid_edges, axis=0)

    return valid_edges



def primes_from_2_to(n):
    """Prime number from 2 to n.

    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.

    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.

    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence


def halton(dim, n_sample):
    """Halton sequence.

    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return sample




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


def vector_angle(waypoints, ei, ej):
    """

    Args:
        waypoints: list of trajectory waypoints
        ei: edge i
        ej: edge j

    Returns: angle(deg) between ei and ej

    """

    vector_i = np.array(waypoints[ei[1]]) - np.array(waypoints[ei[0]])
    vector_j = np.array(waypoints[ej[1]]) - np.array(waypoints[ej[0]])

    inner = np.inner(vector_i, vector_j)
    norms = np.linalg.norm(vector_i) * np.linalg.norm(vector_j)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)  # This is always between 0 and 180 degrees

    return deg


def get_average_edge_angles(G, waypoints, node_i, node_j):

    edges_i = list(G.in_edges(node_i))
    edges_i += list(G.out_edges(node_i))
    edges_j = list(G.in_edges(node_j))
    edges_j += list(G.out_edges(node_j))

    angles = []
    for ei in edges_i:
        for ej in edges_j:
            angles.append(vector_angle(waypoints, ei, ej))

    return np.mean(np.array(angles))



def assign_edge_lengths(G):

    for u, v, d in G.edges(data=True):
        d['length'] = np.linalg.norm(np.array(G.nodes[u]['pos']) - np.array(G.nodes[v]['pos']))

    return G


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

def get_ego_regression_target(params, data, split):

    tile_no = int(data.tile_no[0].cpu().detach().numpy())
    walk_no = int(data.walk_no[0].cpu().detach().numpy())
    idx = int(data.idx[0].cpu().detach().numpy())

    image_fname = "{}{}{}/{:03d}_{:03d}_{:03d}-centerlines-sdf-ego.png".format(params.paths.dataroot, params.paths.rel_dataset, split, tile_no, walk_no, idx)
    im = cv2.imread(image_fname, cv2.IMREAD_GRAYSCALE)

    return im

