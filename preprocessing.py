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
import matplotlib.pyplot as plt
import networkx as nx

# SELECT MODEL TO BE USED
from data_old import TrajectoryDatasetCarla, TrajectoryDatasetIND
from data_av2 import TrajectoryDatasetAV2
from lanegnn.utils import ParamLib
from torch_geometric.loader import DataListLoader


class Preprocessor():

    def __init__(self, export_path, params):

        self.params = params
        self.export_path = export_path
        self.params = params

        print("Exporting to {}".format(export_path))

    def preprocess(self, dataloader):

        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            if data[0] == "empty-trajectory":
                print("Empty trajectory. Skipping sample")
                continue
            else:
                fname = self.export_path + "/{:05d}.pt".format(i)
                print(fname)
                data = data[0]
                torch.save(data, fname)

                # # Plot graph
                # fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
                # plt.tight_layout()
                # plt.axis('off')
                # G_tracklet = data["G_tracklet"]
                # ax.imshow(data["rgb"])
                # nx.draw_networkx(G_tracklet, ax=ax, pos=nx.get_node_attributes(G_tracklet, "pos"),
                #                  edge_color=np.array([255, 0, 142]) / 255.,
                #                  with_labels=False,
                #                  node_size=0,
                #                  arrowsize=5.0, )
                # plt.savefig(fname.replace(".pt", ".png"))


def main():
    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, choices=["carla", "ind", "av2"], help='Dataset to preprocess', required=True)
    parser.add_argument('--export_path', type=str, default="/data/self-supervised-graph/preprocessed/")

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)

    export_path = os.path.join(params.paths.dataroot, "preprocessed", params.paths.config_name)

    if opt.dataset == "carla":
        train_path = os.path.join(params.paths.dataroot, params.paths.config_name)
        test_path = os.path.join(params.paths.dataroot, params.paths.config_name)

        dataset_train = TrajectoryDatasetCarla(path=train_path, params=params)
        #dataset_test = TrajectoryDatasetCarla(path=test_path, params=params)

    elif opt.dataset == "ind":

        train_path = os.path.join(params.paths.dataroot, "inD/data")
        test_path = os.path.join(params.paths.dataroot, "inD/data")

        dataset_train = TrajectoryDatasetIND(path=train_path, params=params)
        #dataset_test = TrajectoryDatasetIND(path=test_path, params=params)

    elif opt.dataset == "av2":

        train_path = os.path.join(params.paths.dataroot_av2, "train")
        dataset_train = TrajectoryDatasetAV2(path=train_path, params=params)
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
    # dataloader_test = dataloader_obj(dataset_test,
    #                                  batch_size=1,
    #                                  num_workers=1,
    #                                  shuffle=False)

    preprocessor = Preprocessor(export_path, params)
    preprocessor.preprocess(dataloader_train)
    # preprocessor.preprocess(dataloader_test)


if __name__ == '__main__':
    main()
