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
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import matplotlib.pyplot as plt
import cv2

from regressors.build_net import build_network
from data.datasets import RegressorDataset
from lanegnn.utils import ParamLib, visualize_angles


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)



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

        print("Training...")


        self.model.train()

        train_progress = tqdm(self.dataloader_train)
        for step, data in enumerate(train_progress):

            #data = self.one_sample_data

            self.optimizer.zero_grad()

            if self.params.model.dataparallel:
                data = [item.to(self.device) for item in data]
            else:
                data = data.to(self.device)


            # loss and optim
            sdf_target = data["sdf"].cuda()
            rgb = data["rgb"].cuda()
            angle_mask_target = data["angles_mask"].cuda()
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

                # fig, axarr = plt.subplots(1, 2)
                # axarr[0].imshow(rgb[0].cpu().numpy().transpose(1, 2, 0))
                # axarr[0].imshow(angle_x_target[0].detach().cpu().numpy(), alpha=0.5)
                # axarr[1].imshow(rgb[0].cpu().numpy().transpose(1, 2, 0))
                # axarr[1].imshow(angle_y_target[0].detach().cpu().numpy(), alpha=0.5)
                # plt.show()


                target_sdf = sdf_target.unsqueeze(1)
                target_angle = torch.cat([angle_x_target.unsqueeze(1), angle_y_target.unsqueeze(1)], dim=1)

                loss_weight = target_sdf > 0.5
                loss_weight = loss_weight.float()

                loss_dict = {
                    'loss_sdf': torch.nn.BCELoss()(pred_sdf, target_sdf),
                    'loss_angles': weighted_mse_loss(pred_angle, target_angle, loss_weight),
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
                    mask_target = angle_mask_target[0].cpu().detach().numpy()

                    pred_viz = visualize_angles(pred_angle[0,0].detach().cpu().numpy(), pred_angle[0, 1].detach().cpu().numpy(), mask=mask_pred)
                    pred_viz_nomask = visualize_angles(pred_angle[0,0].detach().cpu().numpy(), pred_angle[0, 1].detach().cpu().numpy(), mask=None)
                    target_viz = visualize_angles(angle_x_target[0].cpu().numpy(), angle_y_target[0].cpu().numpy(), mask=mask_target)

                    cv2.imshow("mask_pred", mask_pred)
                    cv2.imshow("mask_target", mask_target)


                cv2.imshow("angle target", target_viz)
                cv2.imshow("angle pred", pred_viz)
                cv2.imshow("angle pred nomask", pred_viz_nomask)

                cv2.waitKey(1)

            text = 'Epoch {} / {}, it {} / {}, it glob {}, train loss = {:03f}'.\
                format(epoch, self.params.model.num_epochs, step+1, len(self.dataloader_train), epoch * len(self.dataloader_train) + step+1, loss.item())
            train_progress.set_description(text)

            self.total_step += 1

        if not self.params.main.disable_wandb:
            wandb.log({"train/epoch": epoch})


    def eval(self, epoch):

        print("Evaluating...")

        self.model.eval()

        val_losses = []

        eval_progress = tqdm(self.dataloader_val)
        for step, data in enumerate(eval_progress):

            self.optimizer.zero_grad()

            # loss and optim
            sdf_target = data["sdf"].cuda()
            rgb = data["rgb"].cuda()
            angle_x_target = data["angles_x"].cuda()
            angle_y_target = data["angles_y"].cuda()


            pred = self.model(rgb)
            pred_angle = torch.nn.Tanh()(pred[:, :2])
            pred_sdf = torch.nn.Sigmoid()(pred[:, 2:])


            target_sdf = sdf_target.unsqueeze(1)
            target_angle = torch.cat([angle_x_target.unsqueeze(1), angle_y_target.unsqueeze(1)], dim=1)

            loss_weight = target_sdf > 0.5
            loss_weight = loss_weight.float()

            loss_dict = {
                'loss_sdf': torch.nn.BCELoss()(pred_sdf, target_sdf),
                'loss_angles': weighted_mse_loss(pred_angle, target_angle, loss_weight),
            }

            loss = sum(loss_dict.values())

            val_losses.append(loss.item())

        val_loss = np.nanmean(val_losses)

        print("eval/loss", val_loss)

        if not self.params.main.disable_wandb:
            wandb.log({"eval/loss": val_loss})


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
            notes='regressor',
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
                                 lr=1e-4,
                                 weight_decay=float(params.model.weight_decay),
                                 betas=(params.model.beta_lo, params.model.beta_hi))

    train_path = os.path.join(params.paths.dataroot, 'preprocessed', "*", "train")
    val_path =   os.path.join(params.paths.dataroot, 'preprocessed', "*", "val")
    dataset_train = RegressorDataset(path=train_path, split='train')
    dataset_val = RegressorDataset(path=val_path, split='val')

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=params.model.batch_size_reg,
                                  num_workers=params.model.loader_workers,
                                  shuffle=True)
    dataloader_val = DataLoader(dataset_val,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

    trainer = Trainer(params, model, dataloader_train, dataloader_val, optimizer)

    for epoch in range(params.model.num_epochs):

        # Evaluate
        trainer.eval(epoch)
        trainer.train(epoch)

        #if not params.main.disable_wandb:
        if epoch % 10 == 0:
            try:
                wandb_run_name = wandb.run.name
            except:
                wandb_run_name = "local_run"

            fname = 'regressor_{}_{:04d}.pth'.format(wandb_run_name, epoch)
            checkpoint_path = os.path.join(params.paths.checkpoints, fname)
            print("Saving checkpoint to {}".format(checkpoint_path))

            torch.save(model.state_dict(), checkpoint_path)



if __name__ == '__main__':
    main()
