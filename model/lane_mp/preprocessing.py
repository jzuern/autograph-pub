import numpy as np
import torch
from PIL import Image
import cv2
import torch_geometric.data.dataset
import os
import time
import argparse
from tqdm import tqdm
import torch_geometric.data

# SELECT MODEL TO BE USED
from model.lane_mp.data import TrajectoryDatasetCarla, TrajectoryDatasetIND
from model.lane_mp.utils import ParamLib, unbatch_edge_index, assign_edge_lengths, get_ego_regression_target

# For torch_geometric DataParallel training
from torch_geometric.loader import DataListLoader

import matplotlib.pyplot as plt


class Preprocessor():

    def __init__(self, export_path, params):

        self.params = params
        self.export_path = export_path
        self.params = params

    def preprocess(self, dataloader):

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            if data[0] == "empty-trajectory":
                print("Empty trajectory. Skipping sample")
                continue
            else:
                fname = self.export_path + "/{:05d}.pt".format(i)
                torch.save(data, fname)


def main():
    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, choices=["carla", "ind"], help='Dataset to preprocess', required=True)
    parser.add_argument('--export_path', type=str, default="/data/self-supervised-graph/preprocessed")

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)


    if opt.dataset == "carla":
        train_path = os.path.join(params.paths.dataroot)
        test_path = os.path.join(params.paths.dataroot)

        dataset_train = TrajectoryDatasetCarla(path=train_path, params=params)
        dataset_test = TrajectoryDatasetCarla(path=test_path, params=params)
    elif opt.dataset == "ind":

        train_path = os.path.join(params.paths.dataroot, "inD/data")
        test_path = os.path.join(params.paths.dataroot, "inD/data")

        dataset_train = TrajectoryDatasetIND(path=train_path, params=params)
        dataset_test = TrajectoryDatasetIND(path=test_path, params=params)
    else:
        raise NotImplementedError


    if params.model.dataparallel:
        dataloader_obj = DataListLoader
    else:
        dataloader_obj = torch_geometric.loader.DataLoader

    dataloader_train = dataloader_obj(dataset_train,
                                      batch_size=params.model.batch_size,
                                      num_workers=params.model.loader_workers,
                                      shuffle=True)
    dataloader_test = dataloader_obj(dataset_test,
                                     batch_size=1,
                                     num_workers=1,
                                     shuffle=False)

    preprocessor = Preprocessor(opt.export_path, params)
    preprocessor.preprocess(dataloader_train)
    # preprocessor.preprocess(dataloader_test)


if __name__ == '__main__':
    main()
