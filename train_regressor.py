from builtins import Exception
import os
import networkx as nx
import wandb
import argparse
from tqdm import tqdm
import numpy as np
import time
import torch
import torch.utils.data
import torch_geometric.data
import matplotlib.pyplot as plt

from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
from torch_geometric.data import Batch
import cv2
from regressors.build_net import build_network
from data.data_av2 import PreprocessedDataset
from lanegnn.utils import ParamLib, assign_edge_lengths


class Trainer():

    def __init__(self, params, model, dataloader_train, dataloader_test, optimizer):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.params = params
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_step = 0

        self.figure, self.axarr = plt.subplots(1, 2)

        print(len(self.dataloader_train))

        it = iter(self.dataloader_train)
        i = 0
        while i < 1:
            i += 1
            self.one_sample_data = next(it)


    def train(self, epoch):

        self.model.train()

        train_progress = tqdm(self.dataloader_train)
        for step, data in enumerate(train_progress):

            self.optimizer.zero_grad()

            if self.params.model.dataparallel:
                data = [item.to(self.device) for item in data]
            else:
                data = data.to(self.device)

            # loss and optim
            sdf_target = data.sdf.unsqueeze(0)
            rgb = data.rgb.permute(2, 0, 1)

            # split along dim 1 into batches
            rgb = torch.split(rgb, rgb.shape[1] // self.params.model.batch_size, dim=1)
            rgb = torch.stack(rgb, dim=0)
            sdf_target = torch.split(sdf_target, sdf_target.shape[1] // self.params.model.batch_size, dim=1)
            sdf_target = torch.stack(sdf_target, dim=0)


            sdf_pred = self.model(rgb)
            sdf_pred = torch.nn.Sigmoid()(sdf_pred)


            loss_dict = {
                'loss': torch.nn.BCELoss()(sdf_pred, sdf_target),
            }

            loss = sum(loss_dict.values())
            loss.backward()

            self.optimizer.step()

            if not self.params.main.disable_wandb:
                wandb.log({"train/loss": loss.item()})

            # Visualization
            if self.total_step % 10 == 0:
                cv2.imshow("rgb", cv2.cvtColor(rgb[0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
                cv2.imshow("sdf_target", sdf_target[0, 0].cpu().numpy())
                cv2.imshow("sdf_pred", sdf_pred[0, 0].detach().cpu().numpy())
                cv2.waitKey(1)

            text = 'Epoch {} / {}, it {} / {}, it glob {}, train loss = {:03f}'.\
                format(epoch, self.params.model.num_epochs, step+1, len(self.dataloader_train), epoch * len(self.dataloader_train) + step+1, loss.item())
            train_progress.set_description(text)

            self.total_step += 1

        if not self.params.main.disable_wandb:
            wandb.log({"train/epoch": epoch})


def main():

    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)

    print("Batch size summed over all GPUs: ", params.model.batch_size)
    
    if not params.main.disable_wandb:
        wandb.login()
        wandb.init(
            entity='jannik-zuern',
            project='self_supervised_graph',
            notes='sdf',
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(params.paths)
        wandb.config.update(params.model)
        wandb.config.update(params.preprocessing)


    # -------  Model, optimizer and data initialization ------

    model = build_network(snapshot=None, backend='resnet101', num_channels=3, n_classes=1).to(params.model.device)

    # Make model parallel if available
    if params.model.dataparallel:
        print("Let's use DataParallel with", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model)
    else:
        print("Let's NOT use DataParallel with", torch.cuda.device_count(), "GPUs!")

    # Load model weights
    weights = [w for w in model.parameters() if w.requires_grad]

    optimizer = torch.optim.Adam(weights,
                                 lr=float(params.model.lr),
                                 weight_decay=float(params.model.weight_decay),
                                 betas=(params.model.beta_lo, params.model.beta_hi))

    train_path = os.path.join(params.paths.dataroot, 'preprocessed', params.paths.config_name)
    test_path = os.path.join(params.paths.dataroot, 'preprocessed', params.paths.config_name)
    dataset_train = PreprocessedDataset(path=train_path)
    dataset_test = PreprocessedDataset(path=test_path)

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

    trainer = Trainer(params, model, dataloader_train, dataloader_test, optimizer)

    for epoch in range(params.model.num_epochs):
        trainer.train(epoch)

        #if not params.main.disable_wandb:
        if epoch % 100 == 0:
            try:
                wandb_run_name = wandb.run.name
            except:
                wandb_run_name = "local_run"

            fname = 'regressor_{}_{:04d}.pth'.format(wandb_run_name, epoch)
            checkpoint_path = os.path.join(params.paths.checkpoints, fname)
            print("Saving checkpoint to {}".format(checkpoint_path))

            torch.save(model.state_dict(), checkpoint_path)

        # Evaluate
        # trainer.eval(epoch, split='test')


if __name__ == '__main__':
    main()
