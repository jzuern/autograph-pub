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
import cv2

from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader

from regressors.build_net import build_network
from data.datasets import RegressorDataset
from lanegnn.utils import ParamLib


def visualize_angles(a_x, a_y, mask):
    if mask is None:
        mask = np.ones_like(a_x, dtype=np.uint8)
    global_mask = np.concatenate([mask[..., np.newaxis], mask[..., np.newaxis], mask[..., np.newaxis]], axis=2)
    global_mask = ((global_mask > 0.5) * 255).astype(np.uint8)
    directions_hsv = np.ones([a_x.shape[0], a_x.shape[1], 3], dtype=np.uint8)

    directions_hsv[:, :, 0] = a_x * 127 + 127
    directions_hsv[:, :, 1] = a_y * 127 + 127
    directions_hsv[:, :, 2] = global_mask[:, :, 0]

    directions_hsv = directions_hsv * global_mask

    return directions_hsv


class Trainer():

    def __init__(self, params, model, dataloader_train, dataloader_val, optimizer):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
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

            # loss and optim
            sdf_target = data["sdf"].cuda()
            rgb = data["rgb"].cuda()
            angle_x_target = data["angles_x"].cuda()
            angle_y_target = data["angles_y"].cuda()


            if self.params.model.target == "sdf":
                pred = self.model(rgb)
                pred = torch.nn.Sigmoid()(pred)

                target = sdf_target.unsqueeze(1)

                loss_dict = {
                    'loss_sdf': torch.nn.BCELoss()(pred, target),
                }

            elif self.params.model.target == "angle":
                pred = self.model(rgb)
                pred = torch.nn.Tanh()(pred)

                target = torch.cat([angle_x_target.unsqueeze(1), angle_y_target.unsqueeze(1)], dim=1)

                loss_dict = {
                    'loss_angles': torch.nn.MSELoss()(pred, target),
                }

            elif self.params.model.target == "both":

                pred = self.model(rgb)
                pred_angle = torch.nn.Tanh()(pred[:, :2])
                pred_sdf = torch.nn.Sigmoid()(pred[:, 2:])

                target_sdf = sdf_target.unsqueeze(1)
                target_angle = torch.cat([angle_x_target.unsqueeze(1), angle_y_target.unsqueeze(1)], dim=1)

                loss_dict = {
                    'loss_sdf': torch.nn.BCELoss()(pred_sdf, target_sdf),
                    'loss_angles': torch.nn.MSELoss()(pred_angle, target_angle),
                }

            loss = sum(loss_dict.values())
            loss.backward()

            self.optimizer.step()

            if not self.params.main.disable_wandb:
                wandb.log({"train/loss": loss.item()})

            # Visualization
            if self.total_step % 10 == 0:
                cv2.imshow("rgb", rgb[0].cpu().numpy().transpose(1, 2, 0))

                if self.params.model.target == "sdf":
                    pred_viz = pred[0].detach().cpu().detach().numpy()
                    target_viz = sdf_target[0].cpu().detach().numpy()
                elif self.params.model.target == "angle":
                    pred_viz = visualize_angles(pred[0,0].detach().cpu().numpy(), pred[0,1].detach().cpu().numpy(), mask=None)
                    target_viz = visualize_angles(angle_x_target[0].cpu().numpy(), angle_y_target[0].cpu().numpy(), sdf_target[0].cpu().numpy())
                elif self.params.model.target == "both":
                    mask_pred = pred_sdf[0, 0].detach().cpu().detach().numpy()
                    mask_target = sdf_target[0].cpu().detach().numpy()
                    #mask_pred = None
                    pred_viz = visualize_angles(pred_angle[0,0].detach().cpu().numpy(), pred_angle[0, 1].detach().cpu().numpy(), mask=mask_pred)
                    target_viz = visualize_angles(angle_x_target[0].cpu().numpy(), angle_y_target[0].cpu().numpy(), mask=mask_target)

                    cv2.imshow("mask_pred", mask_pred)

                cv2.imshow("target", target_viz)
                cv2.imshow("pred", pred_viz)

                cv2.waitKey(1)

            text = 'Epoch {} / {}, it {} / {}, it glob {}, train loss = {:03f}'.\
                format(epoch, self.params.model.num_epochs, step+1, len(self.dataloader_train), epoch * len(self.dataloader_train) + step+1, loss.item())
            train_progress.set_description(text)

            self.total_step += 1

        if not self.params.main.disable_wandb:
            wandb.log({"train/epoch": epoch})


    def eval(self, epoch):

        self.model.eval()

        angle_accuaries = []

        eval_progress = tqdm(self.dataloader_val)
        for step, data in enumerate(eval_progress):

            self.optimizer.zero_grad()

            # loss and optim
            sdf_target = data["sdf"].cuda()
            rgb = data["rgb"].cuda()
            angle_x_target = data["angles_x"].cuda()
            angle_y_target = data["angles_y"].cuda()

            pred = self.model(rgb)
            pred = torch.nn.Tanh()(pred)

            if self.params.model.target == "sdf":
                target = sdf_target.unsqueeze(1)
            elif self.params.model.target == "angle":
                target = torch.cat([angle_x_target.unsqueeze(1), angle_y_target.unsqueeze(1)], dim=1)

            angle_accuracy = torch.nn.functional.cosine_similarity(pred, target, dim=1).mean()
            angle_accuaries.append(angle_accuracy.item())

        angle_accuracy = np.nanmean(angle_accuaries)

        print("angle_accuracy", angle_accuracy)

        if not self.params.main.disable_wandb:
            wandb.log({"eval/angle_accuracy": angle_accuracy})


def main():

    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")
    parser.add_argument('--target', type=str, choices=["sdf", "angle", "both"], help="define the target that is used")

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)
    params.model.target = opt.target

    print("Batch size summed over all GPUs: ", params.model.batch_size_reg)
    
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

    if params.model.target == "sdf":
        model = build_network(snapshot=None, backend='resnet101', num_channels=3, n_classes=1).to(params.model.device)
    elif params.model.target == "angle":
        model = build_network(snapshot=None, backend='resnet101', num_channels=3, n_classes=2).to(params.model.device)
    elif params.model.target == "both":
        model = build_network(snapshot=None, backend='resnet101', num_channels=3, n_classes=3).to(params.model.device)


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

    train_path = os.path.join(params.paths.dataroot, 'preprocessed', params.paths.config_name, "train")
    val_path = os.path.join(params.paths.dataroot, 'preprocessed', params.paths.config_name, "val")
    dataset_train = RegressorDataset(path=train_path, split='train')
    dataset_val = RegressorDataset(path=val_path, split='val')

    if params.model.dataparallel:
        dataloader_obj = DataListLoader
    else:
        dataloader_obj = torch_geometric.loader.DataLoader

    dataloader_train = dataloader_obj(dataset_train,
                                      batch_size=params.model.batch_size_reg,
                                      num_workers=params.model.loader_workers,
                                      shuffle=True)
    dataloader_val = dataloader_obj(dataset_val,
                                     batch_size=1,
                                     num_workers=1,
                                     shuffle=False)

    trainer = Trainer(params, model, dataloader_train, dataloader_val, optimizer)

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
        #trainer.eval(epoch)


if __name__ == '__main__':
    main()
