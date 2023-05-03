import torch
from glob import glob
import torch_geometric
import torch_geometric.data.dataset
from tqdm import tqdm
import os
import cv2
import numpy as np
import random
from aggregation.utils import AngleColorizer
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image

get_id = lambda filename: "-".join(os.path.basename(filename).split('-')[0:4])


class SuccessorRegressorDataset(torch.utils.data.Dataset):
    def __init__(self, params, path, split, max_num_samples=None):
        self.path = path
        self.params = params
        self.split = split
        self.ac = AngleColorizer()

        # get all files
        p = Path(self.path).parents[2]

        if params.city == "all":
            params.city = "*"  # redefine for simpler city handling
        params.city = params.city.lower()


        search_pattern = "{}/{}/*/*".format(params.city, self.split)

        print("Dataloader: Looking for files in {} with search pattern {}".format(str(p), search_pattern))

        filelist = [str(f) for f in p.glob(search_pattern) if f.is_file()]
        filelist = sorted(filelist)

        print("     Found {} files in total".format(len(filelist)))

        # filter files
        self.rgb_files = []
        self.sdf_files = []
        self.angles_files = []
        self.pos_enc_files = []
        self.drivable_gt_files = []


        for f in filelist:
            if f.endswith("-rgb.png"):
                self.rgb_files.append(f)
            elif f.endswith("-masks.png"):
                self.sdf_files.append(f)
            elif f.endswith("-angles.png"):
                self.angles_files.append(f)
            elif f.endswith("-pos-encoding.png"):
                self.pos_enc_files.append(f)
            elif f.endswith("-drivable-gt.png"):
                self.drivable_gt_files.append(f)

        self.rgb_files = sorted(list(set(self.rgb_files)))
        self.sdf_files = sorted(list(set(self.sdf_files)))
        self.angles_files = sorted(list(set(self.angles_files)))
        self.pos_enc_files = sorted(list(set(self.pos_enc_files)))
        self.drivable_gt_files = sorted(list(set(self.drivable_gt_files)))

        print("     Number of files before filtering:")
        print("     ", len(self.sdf_files), len(self.angles_files), len(self.rgb_files), len(self.pos_enc_files), len(self.drivable_gt_files))

        if len(self.sdf_files) == 0:
            raise ValueError("No files found in {}".format(path))

        # only use files for which we have all modalities
        file_ids_sdf = [get_id(f) for f in self.sdf_files]
        file_ids_angles = [get_id(f) for f in self.angles_files]
        file_ids_rgb = [get_id(f) for f in self.rgb_files]
        file_ids_pos_enc = [get_id(f) for f in self.pos_enc_files]
        file_ids_drivable_gt = [get_id(f) for f in self.drivable_gt_files]

        file_dict = {}
        for f, p in zip(file_ids_sdf, self.sdf_files):
            if f not in file_dict:
                file_dict[f] = {}
            file_dict[f]["sdf"] = p
        for f, p in zip(file_ids_angles, self.angles_files):
            if f not in file_dict:
                file_dict[f] = {}
            file_dict[f]["angles"] = p
        for f, p in zip(file_ids_rgb, self.rgb_files):
            if f not in file_dict:
                file_dict[f] = {}
            file_dict[f]["rgb"] = p
        for f, p in zip(file_ids_pos_enc, self.pos_enc_files):
            if f not in file_dict:
                file_dict[f] = {}
            file_dict[f]["pos_enc"] = p
        for f, p in zip(file_ids_drivable_gt, self.drivable_gt_files):
            if f not in file_dict:
                file_dict[f] = {}
            file_dict[f]["drivable_gt"] = p


        sdf_files = []
        angles_files = []
        rgb_files = []
        pos_enc_files = []
        drivable_gt_files = []

        for k, v in file_dict.items():
            if len(v) == 5:
                sdf_files.append(v["sdf"])
                angles_files.append(v["angles"])
                rgb_files.append(v["rgb"])
                pos_enc_files.append(v["pos_enc"])
                drivable_gt_files.append(v["drivable_gt"])

        print("     Number of files after all-available filtering:")
        print("     ", len(sdf_files), len(angles_files), len(rgb_files), len(pos_enc_files), len(drivable_gt_files))

        # jointly shuffle them
        c = list(zip(sdf_files, angles_files, rgb_files, pos_enc_files, drivable_gt_files))
        random.shuffle(c)

        self.sdf_files, self.angles_files, self.rgb_files, self.pos_enc_files, self.drivable_gt_files = zip(*c)

        # check if all files are present
        assert len(self.sdf_files) == len(self.angles_files) == len(self.rgb_files) == len(self.pos_enc_files) == len(self.drivable_gt_files)

        # Now we can share the files between type branch and straight
        rgb_branch = [i for i in self.rgb_files if "branching" in i]
        rgb_straight = [i for i in self.rgb_files if "straight" in i]

        print("     Total Branch: {} files".format(len(rgb_branch)))
        print("     Total Straight: {} files".format(len(rgb_straight)))

        # jointly shuffle them
        c = list(zip(self.sdf_files, self.angles_files, self.rgb_files, self.pos_enc_files, self.drivable_gt_files))
        random.shuffle(c)
        self.sdf_files, self.angles_files, self.rgb_files, self.pos_enc_files, self.drivable_gt_files = zip(*c)

        if max_num_samples is not None:
            print("     Limiting number of samples to {}".format(max_num_samples))
            self.sdf_files = self.sdf_files[0:max_num_samples]
            self.angles_files = self.angles_files[0:max_num_samples]
            self.rgb_files = self.rgb_files[0:max_num_samples]
            self.pos_enc_files = self.pos_enc_files[0:max_num_samples]
            self.drivable_gt_files = self.drivable_gt_files[0:max_num_samples]

        print("     Loaded {} {} files".format(len(self.sdf_files), self.split))

    def __len__(self):
        return len(self.sdf_files)

    def augment(self, mask, angles_xy, rgb, pos_enc, drivable_gt):

        angles_x = angles_xy[0]
        angles_y = angles_xy[1]

        # import matplotlib.pyplot as plt
        # fig, axarr  = plt.subplots(2, 6)
        # axarr[0, 0].imshow(mask)
        # axarr[0, 1].imshow(angles_x)
        # axarr[0, 2].imshow(angles_y)
        # axarr[0, 3].imshow(rgb)
        # axarr[0, 4].imshow(pos_enc)
        # axarr[0, 5].imshow(drivable_gt)
        # axarr[1, 0].set_title("mask")
        # axarr[1, 1].set_title("angles x")
        # axarr[1, 2].set_title("angles y")
        # axarr[1, 3].set_title("rgb")
        # axarr[1, 4].set_title("pos_enc")
        # axarr[1, 5].set_title("drivable_gt")

        # convert to PIL image
        mask = Image.fromarray(mask)
        angles_x = Image.fromarray(angles_x)
        angles_y = Image.fromarray(angles_y)
        rgb = Image.fromarray(rgb)
        pos_enc = Image.fromarray(pos_enc)
        drivable_gt = Image.fromarray(drivable_gt)

        # random crop
        if np.random.rand() < 0.6:
            i, j, h, w = transforms.RandomCrop.get_params(rgb, (200, 200))

            rgb = transforms.functional.crop(rgb, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)
            angles_x = transforms.functional.crop(angles_x, i, j, h, w)
            angles_y = transforms.functional.crop(angles_y, i, j, h, w)
            pos_enc = transforms.functional.crop(pos_enc, i, j, h, w)
            drivable_gt = transforms.functional.crop(drivable_gt, i, j, h, w)

            rgb = transforms.functional.resize(rgb, (256, 256))
            mask = transforms.functional.resize(mask, (256, 256))
            angles_x = transforms.functional.resize(angles_x, (256, 256))
            angles_y = transforms.functional.resize(angles_y, (256, 256))
            pos_enc = transforms.functional.resize(pos_enc, (256, 256))
            drivable_gt = transforms.functional.resize(drivable_gt, (256, 256))


        # random color jitter
        if np.random.rand() < 0.5:
            jitter_factor = 1 + 0.2 * (np.random.rand() - 0.5)
            rgb = transforms.functional.adjust_brightness(rgb, jitter_factor)
            rgb = transforms.functional.adjust_contrast(rgb, jitter_factor)
            rgb = transforms.functional.adjust_saturation(rgb, jitter_factor)

        # convert to numpy array
        rgb = np.array(rgb)
        mask = np.array(mask)
        angles_x = np.array(angles_x)
        angles_y = np.array(angles_y)
        angles_xy = np.array([angles_x, angles_y])
        pos_enc = np.array(pos_enc)
        drivable_gt = np.array(drivable_gt)


        # axarr[1, 0].imshow(mask)
        # axarr[1, 1].imshow(angles_x)
        # axarr[1, 2].imshow(angles_y)
        # axarr[1, 3].imshow(rgb)
        # axarr[1, 4].imshow(pos_enc)
        # axarr[1, 5].imshow(drivable_gt)
        # plt.show()


        return mask, angles_xy, rgb, pos_enc, drivable_gt



    def __getitem__(self, idx):
        mask = cv2.imread(self.sdf_files[idx], cv2.IMREAD_COLOR)
        pos_enc = cv2.imread(self.pos_enc_files[idx], cv2.IMREAD_UNCHANGED)
        angles = cv2.imread(self.angles_files[idx], cv2.IMREAD_COLOR)
        angles = cv2.cvtColor(angles, cv2.COLOR_BGR2RGB)
        rgb = cv2.imread(self.rgb_files[idx], cv2.IMREAD_UNCHANGED)
        drivable_gt = cv2.imread(self.drivable_gt_files[idx], cv2.IMREAD_UNCHANGED)

        if mask is None or pos_enc is None or angles is None or rgb is None or drivable_gt is None:
            print("Error loading file: {}".format(self.sdf_files[idx]))

            return_dict = {
                'drivable': torch.zeros([256, 256]),
                'drivable_gt': torch.zeros([256, 256]),
                'mask_successor': torch.zeros([256, 256]),
                'mask_pedestrian': torch.zeros([256, 256]),
                'pos_enc': torch.zeros([3, 256, 256]),
                'angles_xy': torch.zeros([2, 256, 256]),
                'rgb': torch.zeros([3, 256, 256]),
            }
            return return_dict

        angles = self.ac.color_to_angle(angles)
        angles_xy = self.ac.angle_to_xy(angles)

        if self.split == 'train':
            mask, angles_xy, rgb, pos_enc, drivable_gt = self.augment(mask, angles_xy, rgb, pos_enc, drivable_gt)

        mask_full = mask[:, :, 1]
        mask_successor = mask[:, :, 2]
        mask_pedestrian = mask[:, :, 0]

        # to tensor
        angles_xy = torch.from_numpy(angles_xy).float()
        mask_full = torch.from_numpy(mask_full).float() / 255.0
        drivable_gt = torch.from_numpy(drivable_gt).float() / 255.0

        mask_successor = torch.from_numpy(mask_successor).float() / 255.0
        mask_pedestrian = torch.from_numpy(mask_pedestrian).float() / 255.0
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        pos_enc = torch.from_numpy(pos_enc).float().permute(2, 0, 1) / 255.0

        return_dict = {
            'drivable': mask_full,
            'drivable_gt': drivable_gt,
            'mask_successor': mask_successor,
            'mask_pedestrian': mask_pedestrian,
            'pos_enc': pos_enc,
            'angles_xy': angles_xy,
            'rgb': rgb
        }

        return return_dict




# class RegressorDataset(torch.utils.data.Dataset):
#     def __init__(self, path, split):
#         self.path = path
#
#         print("Looking for files in", path)
#
#         self.sdf_files = sorted(glob(os.path.join(path, '*-sdf-tracklets-dense.png')))
#         self.intersection_files = sorted(glob(os.path.join(path, '*-intersection-gt.png')))
#         self.angles_files = sorted(glob(os.path.join(path, '*-angles-tracklets-dense.png')))
#         self.rgb_files = sorted(glob(os.path.join(path, '*-rgb.png')))
#
#         # only use files for which we have all three modalities
#         file_ids = [get_id(f) for f in self.sdf_files]
#         self.sdf_files = [f for f in self.sdf_files if get_id(f) in file_ids]
#         self.angles_files = [f for f in self.angles_files if get_id(f) in file_ids]
#         self.rgb_files = [f for f in self.rgb_files if get_id(f) in file_ids]
#         self.intersection_files = [f for f in self.intersection_files if get_id(f) in file_ids]
#
#         # check if all files are present
#         assert len(self.sdf_files) == len(self.angles_files) == len(self.rgb_files)
#
#         self.split = split
#
#         print("Loaded {} {} files".format(len(self.sdf_files), self.split))
#
#     def __len__(self):
#         return len(self.sdf_files)
#
#
#     def __getitem__(self, idx):
#         sdf = cv2.imread(self.sdf_files[idx], cv2.IMREAD_UNCHANGED)
#         intersection = cv2.imread(self.intersection_files[idx], cv2.IMREAD_UNCHANGED)
#         angles = cv2.imread(self.angles_files[idx], cv2.IMREAD_COLOR)
#         angles = cv2.cvtColor(angles, cv2.COLOR_BGR2RGB)
#         rgb = cv2.imread(self.rgb_files[idx], cv2.IMREAD_UNCHANGED)
#
#
#         # convert from angles to unit circle xy coordinates
#         # to hsv to get hue
#         angles_mask = (angles[:, :, 1] > 0).astype(np.uint8)
#         angles = angles[:, :, 0] / 255.0
#
#         angles_x = np.cos(angles)
#         angles_y = np.sin(angles)
#
#         # to tensor
#         sdf = torch.from_numpy(sdf).float() / 255.0
#         intersection = torch.from_numpy(intersection).float() / 255.0
#         angles_x = torch.from_numpy(angles_x).float()
#         angles_y = torch.from_numpy(angles_y).float()
#         angles_mask = torch.from_numpy(angles_mask).float()
#         angles = torch.from_numpy(angles).float()
#         rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
#
#         return_dict = {
#             'sdf': sdf,
#             'intersection': intersection,
#             'angles_mask': angles_mask,
#             'angles_x': angles_x,
#             'angles_y': angles_y,
#             'angles': angles,
#             'rgb': rgb
#         }
#
#         #return_dict = self.random_rotate(return_dict)
#
#         return return_dict
#
#
# class PreprocessedDataset(torch_geometric.data.Dataset):
#
#     def __init__(self, path, split='train', target=None, num_samples=1000000, in_layers=""):
#         super(PreprocessedDataset, self).__init__(path)
#         print("Loading preprocessed dataset from {}".format(path))
#
#         self.path = path
#         self.split = split
#         self.in_layers = in_layers
#         self.target = target
#         if num_samples is None:
#             num_samples = 100000000
#
#         self.pth_files = sorted(glob(path + '/*-{}.pth'.format(self.target)))
#         print("Found {} files".format(len(self.pth_files)))
#
#         self.pth_files = self.pth_files[0:num_samples]
#         print("Using {} files".format(len(self.pth_files)))
#
#         print("Split: {}".format(self.split))
#         print("File ids: {}".format([get_id(f) for f in self.pth_files]))
#
#         #self.check_files()
#
#     def __len__(self):
#         return len(self.pth_files)
#
#     def check_files(self):
#         print("Checking files...")
#         valid_files = []
#
#         for i, pth_file in tqdm(enumerate(self.pth_files)):
#             try:
#                 _ = torch.load(pth_file)
#                 valid_files.append(pth_file)
#             except:
#                 print("Error loading file {}".format(pth_file))
#                 continue
#         self.pth_files = valid_files
#
#
#     def random_rotate(self, data):
#
#         image_size = data['rgb'].shape[1]
#         center = image_size // 2
#
#         angle = torch.random.choice([0, 90, 180, 270])
#
#         # rotate images
#         data.rgb = torch.rot90(data.rgb, k=int(angle / 90), dims=[1, 2])
#         data.sdf = torch.rot90(data.sdf, k=int(angle / 90), dims=[1, 2])
#
#         # rotate node feats around center
#         data.pos[:, 0] = center + (data.pos[:, 0] - center) * torch.cos(angle / 180 * np.pi) - \
#                                   (data.pos[:, 1] - center) * torch.sin(angle / 180 * np.pi)
#         data.pos[:, 1] = center + (data.pos[:, 0] - center) * torch.sin(angle / 180 * np.pi) + \
#                                   (data.pos[:, 1] - center) * torch.cos(angle / 180 * np.pi)
#
#         # rotate graph around center
#         graph_node_pos = nx.get_node_attributes(data.graph, 'pos')
#         graph_node_pos[:, 0] = center + (graph_node_pos[:, 0] - center) * torch.cos(angle / 180 * np.pi) - \
#                                         (data.pos[:, 1] - center) * torch.sin(angle / 180 * np.pi)
#         graph_node_pos[:, 1] = center + (graph_node_pos[:, 0] - center) * torch.sin(angle / 180 * np.pi) + \
#                                         (data.pos[:, 1] - center) * torch.cos(angle / 180 * np.pi)
#         nx.set_node_attributes(data.graph, graph_node_pos, 'pos')
#
#
#         return data
#
#
#     def __getitem__(self, index):
#
#         data = torch.load(self.pth_files[index])
#
#         rgb = data['rgb']
#         sdf = data['sdf'].float()
#         angles = data['angles'].float()
#         edge_pos_feats = data['edge_pos_feats'].float()
#         edge_img_feats = data['edge_img_feats'].float() / 255.
#         edge_scores = data['edge_scores'].float()
#         edge_indices = data['edge_indices']
#         graph = data['graph']
#         node_feats = data['node_feats'].float()
#         node_scores = data['node_scores'].float()
#
#         data = torch_geometric.data.Data(node_feats=node_feats,
#                                          edge_indices=edge_indices.contiguous(),
#                                          edge_pos_feats=edge_pos_feats,
#                                          edge_img_feats=edge_img_feats,
#                                          node_scores=node_scores.contiguous(),
#                                          edge_scores=edge_scores.contiguous(),
#                                          edge_len=torch.tensor(len(edge_scores)),
#                                          gt_graph=graph,
#                                          num_nodes=node_feats.shape[0],
#                                          batch_idx=torch.tensor(index),
#                                          rgb=torch.FloatTensor(rgb / 255.),
#                                          sdf=torch.FloatTensor(sdf),
#                                          )
#
#         #if self.params.preprocessing.augment and self.split == 'train':
#         #    data = self.random_rotate(data)
#
#         return data
#
#
# class PreprocessedDatasetSuccessor(torch_geometric.data.Dataset):
#
#     def __init__(self, path):
#         super(PreprocessedDatasetSuccessor, self).__init__()
#
#         self.node_feats_files = []
#         self.edge_files = []
#         self.edge_attr_files = []
#         self.edge_img_feats_files = []
#         self.node_gt_files = []
#         self.node_endpoint_gt_files = []
#         self.edge_gt_files = []
#         self.edge_gt_onehot_files = []
#         self.gt_graph_files = []
#         self.rgb_files = []
#         self.rgb_context_files = []
#         self.context_regr_smooth_files = []
#         self.ego_regr_smooth_files = []
#
#         city_str = '*'
#         print(path + '/{}-node-feats.pth'.format(city_str))
#         self.node_feats_files.extend(glob(path + '/{}-node-feats.pth'.format(city_str)))
#         self.edge_files.extend(glob(path + '/{}-edges.pth'.format(city_str)))
#         self.edge_attr_files.extend(glob(path + '/{}-edge-attr.pth'.format(city_str)))
#         self.edge_img_feats_files.extend(glob(path + '/{}-edge-img-feats.pth'.format(city_str)))
#         self.node_gt_files.extend(glob(path + '/{}-node-gt.pth'.format(city_str)))
#         self.node_endpoint_gt_files.extend(glob(path + '/{}-node-endpoint-gt.pth'.format(city_str)))
#         self.edge_gt_files.extend(glob(path + '/{}-edge-gt.pth'.format(city_str)))
#         self.edge_gt_onehot_files.extend(glob(path + '/{}-edge-gt-onehot.pth'.format(city_str)))
#         self.gt_graph_files.extend(glob(path + '/{}-gt-graph.pth'.format(city_str)))
#         self.rgb_files.extend(glob(path + '/{}-rgb.pth'.format(city_str)))
#         self.rgb_context_files.extend(glob(path + '/{}-rgb-context.pth'.format(city_str)))
#         self.context_regr_smooth_files.extend(glob(path + '/{}-context-regr-smooth.pth'.format(city_str)))
#         self.ego_regr_smooth_files.extend(glob(path + '/{}-ego-regr-smooth.pth'.format(city_str)))
#
#         self.node_feats_files = sorted(self.node_feats_files)
#         self.edge_files = sorted(self.edge_files)
#         self.edge_attr_files = sorted(self.edge_attr_files)
#         self.edge_img_feats_files = sorted(self.edge_img_feats_files)
#         self.node_gt_files = sorted(self.node_gt_files)
#         self.node_endpoint_gt_files = sorted(self.node_endpoint_gt_files)
#         self.edge_gt_files = sorted(self.edge_gt_files)
#         self.edge_gt_onehot_files = sorted(self.edge_gt_onehot_files)
#         self.gt_graph_files = sorted(self.gt_graph_files)
#         self.rgb_files = sorted(self.rgb_files)
#         self.rgb_context_files = sorted(self.rgb_context_files)
#         self.context_regr_smooth_files = sorted(self.context_regr_smooth_files)
#         self.ego_regr_smooth_files = sorted(self.ego_regr_smooth_files)
#
#         print("Found {} samples in path {}".format(len(self.rgb_files), path))
#
#
#     def __len__(self):
#         return len(self.rgb_files)
#
#     def __getitem__(self, index):
#         # Return reduced data object if the index is in the index_filter (to save time)
#
#         start_time = time.time()
#
#         node_feats = torch.load(self.node_feats_files[index])
#         edges = torch.load(self.edge_files[index])
#         edge_attr = torch.load(self.edge_attr_files[index])
#         edge_img_feats = torch.load(self.edge_img_feats_files[index]).to(torch.float32) / 255.0 # cast uint8 to float32
#         node_gt = torch.load(self.node_gt_files[index])
#         node_endpoint_gt = torch.load(self.node_endpoint_gt_files[index]).float()
#         edge_gt = torch.load(self.edge_gt_files[index])
#         edge_gt_onehot = torch.load(self.edge_gt_onehot_files[index])
#         gt_graph = torch.load(self.gt_graph_files[index])
#         rgb = torch.load(self.rgb_files[index])
#         rgb_context = torch.load(self.rgb_context_files[index])
#         context_regr_smooth = torch.load(self.context_regr_smooth_files[index])
#         ego_regr_smooth = torch.load(self.ego_regr_smooth_files[index])
#
#         # switch node columns to match the order of the edge columns
#         node_feats = torch.cat((node_feats[:, 1:2], node_feats[:, 0:1]), dim=1)
#
#
#         data = torch_geometric.data.Data(node_feats=node_feats,
#                                          edge_indices=edges.t().contiguous(),
#                                          edge_pos_feats=edge_attr,
#                                          edge_img_feats=edge_img_feats,
#                                          node_scores=node_gt.t().contiguous(),
#                                          #node_endpoint_gt=node_endpoint_gt.t().contiguous(),
#                                          edge_scores=edge_gt.t().contiguous(),
#                                          #edge_gt_onehot=edge_gt_onehot.t().contiguous(),
#                                          gt_graph=gt_graph,
#                                          num_nodes=node_feats.shape[0],
#                                          batch_idx=torch.tensor(len(gt_graph)),
#                                          rgb=torch.FloatTensor(rgb / 255.), # [0.0, 1.0]
#                                          rgb_context=torch.FloatTensor(rgb_context / 255.), # [0.0, 1.0]
#                                          context_regr_smooth=torch.FloatTensor(context_regr_smooth), # [0.0, 1.0]
#                                          ego_regr_smooth=torch.FloatTensor(ego_regr_smooth), # [0.0, 1.0]
#                                          data_time=torch.tensor(time.time() - start_time),
#                                          )
#
#         return data



