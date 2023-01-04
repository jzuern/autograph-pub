import torch
from glob import glob
import torch_geometric
import torch_geometric.data.dataset
from tqdm import tqdm
import os
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class RegressorDataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        self.path = path

        print("Looking for files in", path)

        self.sdf_files = sorted(glob(os.path.join(path, '*-sdf-tracklets-dense.png')))
        self.angles_files = sorted(glob(os.path.join(path, '*-angles-tracklets-dense.png')))
        self.rgb_files = sorted(glob(os.path.join(path, '*-rgb.png')))

        def get_id(filename):
            return os.path.basename(filename).split('-')[1] + '-' + os.path.basename(filename).split('-')[2]

        # check if all sdf files have same resolution
        sdf_files = self.sdf_files
        self.sdf_files = []
        for sdf_file in sdf_files:
            sdf_new = cv2.imread(sdf_file, cv2.IMREAD_UNCHANGED)
            if sdf_new.shape[0] == sdf_new.shape[1]:
                self.sdf_files.append(sdf_file)

        # only use files for which we have all three modalities
        file_ids = [get_id(f) for f in self.sdf_files]
        self.sdf_files = [f for f in self.sdf_files if get_id(f) in file_ids]
        self.angles_files = [f for f in self.angles_files if get_id(f) in file_ids]
        self.rgb_files = [f for f in self.rgb_files if get_id(f) in file_ids]


        # check if all files are present
        assert len(self.sdf_files) == len(self.angles_files) == len(self.rgb_files)

        self.split = split

        print("Loaded {} {} files".format(len(self.sdf_files), self.split))

    def __len__(self):
        return len(self.sdf_files)


    def random_rotate(self, return_dict):


        # TODO: DEBBUG THIS

        # random rotation
        k = np.random.choice([0, 1, 2, 3])
        #k = np.random.choice([1])


        return_dict['rgb_unrotated'] = return_dict['rgb']

        return_dict['rgb'] = torch.rot90(return_dict['rgb'], k, [1, 2])
        return_dict['sdf'] = torch.rot90(return_dict['sdf'], k, [0, 1])

        return_dict['angles_x'] = torch.rot90(return_dict['angles_x'], k, [0, 1])
        return_dict['angles_y'] = torch.rot90(return_dict['angles_y'], k, [0, 1])


        # return_dict['angles_x'] += torch.cos(torch.tensor(k * np.pi / 2))
        # return_dict['angles_y'] += torch.sin(torch.tensor(k * np.pi / 2))
        #
        # return_dict['angles_x'][return_dict['angles_x'] > np.pi] -= 2*np.pi
        # return_dict['angles_x'][return_dict['angles_x'] < -np.pi] += 2*np.pi
        #
        # return_dict['angles_y'][return_dict['angles_y'] > np.pi] -= 2*np.pi
        # return_dict['angles_y'][return_dict['angles_y'] < -np.pi] += 2*np.pi

        return return_dict



    def __getitem__(self, idx):
        sdf = cv2.imread(self.sdf_files[idx], cv2.IMREAD_UNCHANGED)
        angles = cv2.imread(self.angles_files[idx], cv2.IMREAD_ANYCOLOR)
        rgb = cv2.imread(self.rgb_files[idx], cv2.IMREAD_UNCHANGED)


        # convert from angles to unit circle xy coordinates
        # to hsv to get hue
        angles = cv2.cvtColor(angles, cv2.COLOR_BGR2HSV)
        angles_mask = (angles[:, :, 1] > 0).astype(np.uint8)

        angles = angles[:, :, 0] / 255.0 * 2 * np.pi - np.pi

        angles_x = np.cos(angles)
        angles_y = np.sin(angles)

        # to tensor
        sdf = torch.from_numpy(sdf).float() / 255.0
        angles_x = torch.from_numpy(angles_x).float()
        angles_y = torch.from_numpy(angles_y).float()
        angles_mask = torch.from_numpy(angles_mask).float()
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        return_dict = {
            'sdf': sdf,
            'angles_mask': angles_mask,
            'angles_x': angles_x,
            'angles_y': angles_y,
            'rgb': rgb
        }

        #return_dict = self.random_rotate(return_dict)

        return return_dict


class PreprocessedDataset(torch_geometric.data.Dataset):

    def __init__(self, path, split='train', num_samples=1000000, in_layers=""):
        super(PreprocessedDataset, self).__init__(path)
        print("Loading preprocessed dataset from {}".format(path))

        self.path = path
        self.split = split
        self.in_layers = in_layers

        self.pth_files = sorted(glob(path + '/*-sparse.pth'))
        print("Found {} files".format(len(self.pth_files)))

        self.pth_files = self.pth_files[:num_samples]
        print("Using {} files".format(len(self.pth_files)))

        if "sdf" in in_layers:
            self.sdf_files = [f.replace("-post", "-pre").replace("-sparse.pth", "-sdf-tracklets-sparse.png") for f in self.pth_files]
            assert all([os.path.exists(f) for f in self.sdf_files])
        if "angle" in in_layers:
            self.angle_files = [f.replace("-post", "-pre").replace("-sparse.pth", "-angles-tracklets-sparse.png") for f in self.pth_files]
            assert all([os.path.exists(f) for f in self.angle_files])

        self.check_files()

    def __len__(self):
        return len(self.pth_files)

    def check_files(self):
        valid_files = []

        for i, pth_file in tqdm(enumerate(self.pth_files)):
            try:
                _ = torch.load(pth_file)
                valid_files.append(pth_file)
            except:
                print("Error loading file {}".format(pth_file))
                continue
        self.pth_files = valid_files


    def random_rotate(self, data):

        image_size = data['rgb'].shape[1]
        center = image_size // 2

        angle = torch.random.choice([0, 90, 180, 270])

        # rotate images
        data.rgb = torch.rot90(data.rgb, k=int(angle / 90), dims=[1, 2])
        data.sdf = torch.rot90(data.sdf, k=int(angle / 90), dims=[1, 2])

        # rotate node feats around center
        data.pos[:, 0] = center + (data.pos[:, 0] - center) * torch.cos(angle / 180 * np.pi) - \
                                  (data.pos[:, 1] - center) * torch.sin(angle / 180 * np.pi)
        data.pos[:, 1] = center + (data.pos[:, 0] - center) * torch.sin(angle / 180 * np.pi) + \
                                  (data.pos[:, 1] - center) * torch.cos(angle / 180 * np.pi)

        # rotate graph around center
        graph_node_pos = nx.get_node_attributes(data.graph, 'pos')
        graph_node_pos[:, 0] = center + (graph_node_pos[:, 0] - center) * torch.cos(angle / 180 * np.pi) - \
                                        (data.pos[:, 1] - center) * torch.sin(angle / 180 * np.pi)
        graph_node_pos[:, 1] = center + (graph_node_pos[:, 0] - center) * torch.sin(angle / 180 * np.pi) + \
                                        (data.pos[:, 1] - center) * torch.cos(angle / 180 * np.pi)
        nx.set_node_attributes(data.graph, graph_node_pos, 'pos')


        return data


    def __getitem__(self, index):

        data = torch.load(self.pth_files[index])

        rgb = data['rgb']
        sdf = data['sdf'].float()
        edge_pos_feats = data['edge_pos_feats'].float()
        edge_img_feats = data['edge_img_feats'].float() / 255.
        edge_scores = data['edge_scores'].float()
        edge_indices = data['edge_indices']
        graph = data['graph']
        node_feats = data['node_feats'].float()
        node_scores = data['node_scores'].float()

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
                                         sdf=torch.FloatTensor(sdf),
                                         )

        #if self.params.preprocessing.augment and self.split == 'train':
        #    data = self.random_rotate(data)

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



