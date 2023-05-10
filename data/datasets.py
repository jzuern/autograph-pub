import torch
import torch_geometric.data.dataset
import os
import cv2
import numpy as np
import random
from aggregation.utils import AngleColorizer
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import pickle

get_id = lambda filename: "-".join(os.path.basename(filename).split('-')[0:4])


class SuccessorGraphDataset(torch_geometric.data.dataset.Dataset):
    def __init__(self, params, path, split):
        self.path = path
        self.params = params
        self.split = split

        # get all files
        p = Path(self.path).parents[2]

        if params.city == "all":
            params.city = "*"  # redefine for simpler city handling
        params.city = params.city.lower()

        search_pattern = "{}/successor-lgp/{}/*".format(params.city, self.split)

        print("SuccessorGraphDataset: Looking for files in {} with search pattern {}".format(str(p), search_pattern))

        filelist = [str(f) for f in p.glob(search_pattern) if f.is_file()]
        filelist = sorted(filelist)

        print("     Found {} eval graph files in total".format(len(filelist)))

        # filter files
        self.rgb_files = []
        self.graph_files = []

        for f in filelist:
            if f.endswith("-rgb.png"):
                self.rgb_files.append(f)
            elif f.endswith("-graph.gpickle"):
                self.graph_files.append(f)

        self.rgb_files = sorted(list(set(self.rgb_files)))
        self.graph_files = sorted(list(set(self.graph_files)))

        assert len(self.rgb_files) == len(self.graph_files), "Number of rgb files and graph files must be equal"

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb = cv2.imread(self.rgb_files[idx], cv2.IMREAD_UNCHANGED)
        graph_gt = self.graph_files[idx]

        # to tensor
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        return_dict = {
            'rgb': rgb,
            'graph_gt': graph_gt,
        }

        return return_dict


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
