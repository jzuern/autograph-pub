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
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
import torchvision.models as models
from torchmetrics import JaccardIndex, Precision, Recall, F1Score

from data.datasets import RegressorDataset
from lanegnn.utils import ParamLib, visualize_angles


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


# Calculate metrics according to torchmetrics
#precision = Precision(task="binary", average='none', mdmc_average='global')
#recall = Recall(task="binary", average='none', mdmc_average='global')
iou = JaccardIndex(task="binary", reduction='none', num_classes=2, ignore_index=0)
f1 = F1Score(average='none', mdmc_average='global', num_classes=2, ignore_index=0)


def calc_torchmetrics(seg_preds, seg_gts, name):

    seg_preds = torch.tensor(seg_preds)
    seg_gts = torch.tensor(seg_gts)

    #p = precision(seg_preds, seg_gts).numpy()
    #r = recall(seg_preds, seg_gts).numpy()
    i = iou(seg_preds, seg_gts).numpy()
    f = f1(seg_preds, seg_gts).numpy()

    metrics = {
        #'eval/precision_{}'.format(name): p.item(),
        #'eval/recall_{}'.format(name): r.item(),
        'eval/iou_{}'.format(name): i.item(),
        'eval/f1_{}'.format(name): f[1],
    }

    return metrics




class Trainer():

    def __init__(self, params, model, dataloader_train, dataloader_val, optimizer):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.params = params
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_step = 0
        self.threshold_sdf = 0.3
        self.threshold_angle = np.pi / 4.

        if self.params.stego:
            print("Using STEGO Loss")

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

            # loss and optim
            sdf_target = data["sdf"].cuda()
            rgb = data["rgb"].cuda()
            angle_mask_target = data["angles_mask"].cuda()
            target_angle_x = data["angles_x"].cuda()
            target_angle_y = data["angles_y"].cuda()

            target_angle = data["angles"].cuda().unsqueeze(1)

            target_sdf = sdf_target.unsqueeze(1)
            #target_angle = torch.cat([target_angle_x.unsqueeze(1), target_angle_y.unsqueeze(1)], dim=1)

            if self.params.model.target == "sdf":
                pred = self.model(rgb)
                pred = torch.nn.Sigmoid()(pred)

                loss_dict = {
                    'loss_sdf': torch.nn.BCELoss()(pred, target_sdf),
                }

            elif self.params.model.target == "angle":
                pred = self.model(rgb)
                pred = torch.nn.Tanh()(pred)

                loss_dict = {
                    'loss_angles': torch.nn.MSELoss()(pred, target_angle),
                }

            elif self.params.model.target == "both":

                (pred, features) = self.model(rgb)
                pred = torch.nn.functional.interpolate(pred, size=target_sdf.shape[2:], mode='bilinear', align_corners=True)

                #red_angle = torch.nn.Tanh()(pred[:, :2])
                #pred_sdf = torch.nn.Sigmoid()(pred[:, 2:])
                pred_angle = torch.nn.Sigmoid()(pred[:, :1])
                pred_sdf = torch.nn.Sigmoid()(pred[:, 1:])

                loss_weight = target_sdf > 0.5
                loss_weight = loss_weight.float()

                loss_dict = {
                    'loss_sdf': torch.nn.BCELoss()(pred_sdf, target_sdf),
                    #'loss_angles': weighted_mse_loss(pred_angle, target_angle, loss_weight),
                    'loss_angles': torch.nn.BCELoss(weight=loss_weight)(pred_angle, target_angle),
                }

            else:
                raise Exception("unknown target")

            if self.params.stego:
                loss_dict["stego"] = stego_loss(features, target_sdf)

            loss = sum(loss_dict.values())
            loss.backward()

            self.optimizer.step()

            if not self.params.main.disable_wandb:
                wandb.log({"train/loss": loss.item()})

            # Visualization
            if self.total_step % 10 == 0 and self.params.visualize:
                cv2.imshow("rgb", rgb[0].cpu().numpy().transpose(1, 2, 0))

                if self.params.model.target == "sdf":
                    pred_viz = pred[0].detach().cpu().detach().numpy()
                    target_viz = sdf_target[0].cpu().detach().numpy()
                elif self.params.model.target == "angle":
                    pred_viz = visualize_angles(pred[0,0].detach().cpu().numpy(), pred[0,1].detach().cpu().numpy(), mask=None)
                    target_viz = visualize_angles(target_angle_x[0].cpu().numpy(), target_angle_y[0].cpu().numpy(), sdf_target[0].cpu().numpy())
                elif self.params.model.target == "both":
                    mask_pred = pred_sdf[0, 0].detach().cpu().detach().numpy()
                    mask_target = angle_mask_target[0].cpu().detach().numpy()
                    pred_angle = pred_angle[0, 0].cpu().detach().numpy()
                    target_angle = target_angle[0, 0].cpu().detach().numpy()

                    print("pred", pred_angle.min(), pred_angle.max())
                    print("target", target_angle.min(), target_angle.max())

                    mask_pred = np.concatenate([mask_pred[..., np.newaxis], mask_pred[..., np.newaxis], mask_pred[..., np.newaxis]], axis=2)
                    mask_pred = (mask_pred > 0.3).astype(np.uint8)
                    mask_target = np.concatenate([mask_target[..., np.newaxis], mask_target[..., np.newaxis], mask_target[..., np.newaxis]], axis=2)
                    mask_target = (mask_target > 0.3).astype(np.uint8)

                    pred_viz = cv2.applyColorMap((pred_angle * 255).astype(np.uint8), cv2.COLORMAP_TURBO) * mask_pred
                    target_viz = cv2.applyColorMap((target_angle * 255).astype(np.uint8), cv2.COLORMAP_TURBO) * mask_target

                    #pred_viz_nomask = visualize_angles(pred_angle[0,0].detach().cpu().numpy(), pred_angle[0, 1].detach().cpu().numpy(), mask=None)
                    #target_viz = visualize_angles(target_angle_x[0].cpu().numpy(), target_angle_y[0].cpu().numpy(), mask=mask_target)

                    cv2.imshow("mask_pred", mask_pred * 255)
                    cv2.imshow("mask_target", mask_target * 255)

                cv2.imshow("angle target", target_viz)
                cv2.imshow("angle pred", pred_viz)
                #cv2.imshow("angle pred nomask", pred_viz_nomask)

                cv2.waitKey(1)

            text = 'Epoch {} / {}, it {} / {}, it global {}, train loss = {:03f}'.\
                format(epoch, self.params.model.num_epochs, step+1, len(self.dataloader_train), epoch * len(self.dataloader_train) + step+1, loss.item())
            train_progress.set_description(text)

            self.total_step += 1

        if not self.params.main.disable_wandb:
            wandb.log({"train/epoch": epoch})


    def eval(self, epoch):

        print("Evaluating...")

        self.model.eval()

        val_losses = []
        seg_sdf_preds = []
        seg_sdf_gts = []
        frac_correct_list = []

        eval_progress = tqdm(self.dataloader_val)
        for step, data in enumerate(eval_progress):

            self.optimizer.zero_grad()

            # loss and optim
            sdf_target = data["sdf"].cuda()
            rgb = data["rgb"].cuda()
            angle_x_target = data["angles_x"].cuda()
            angle_y_target = data["angles_y"].cuda()

            (pred, features) = self.model(rgb)
            pred = torch.nn.functional.interpolate(pred, size=sdf_target.shape[1:], mode='bilinear', align_corners=True)

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

            seg_sdf_preds.append((pred_sdf[0] > self.threshold_sdf).cpu().numpy().astype(np.uint8)[0])
            seg_sdf_gts.append((target_sdf[0] > self.threshold_sdf).cpu().numpy().astype(np.uint8).squeeze())

            target_angle = torch.atan2(angle_y_target, angle_x_target)
            pred_angle = torch.atan2(pred_angle[0, 1], pred_angle[0, 0])

            correct_angles = (torch.abs(pred_angle - target_angle)[0] < self.threshold_angle)
            correct_angles = (correct_angles * (target_sdf[0, 0] > self.threshold_sdf)).cpu().numpy().astype(np.uint8)
            sdf_thresholded = (target_sdf[0, 0].cpu().numpy() > self.threshold_sdf).astype(np.uint8)

            frac_correct_angles = np.sum(correct_angles) / np.sum(sdf_thresholded)
            frac_correct_list.append(frac_correct_angles)

            loss = sum(loss_dict.values())
            val_losses.append(loss.item())


        val_loss = np.nanmean(val_losses)

        print("eval/loss", val_loss)

        metrics_sdf = calc_torchmetrics(seg_sdf_preds, seg_sdf_gts, name="sdf")

        frac_correct_angle = np.nanmean(frac_correct_list)
        print("eval/frac_correct_angle", frac_correct_angle)

        print(metrics_sdf)

        if not self.params.main.disable_wandb:
            wandb.log({"eval/loss": val_loss,
                       "eval/frac_correct_angle": frac_correct_angle})
            wandb.log(metrics_sdf)

        return val_loss


def stego_loss(features, target):


    # Downsample target segmentation to match the size of the feature map
    target = torch.nn.functional.interpolate(target.float(), size=features.shape[2:], mode='nearest')[:, 0, :, :]

    # Downsample again due to VRAM constrints
    target = target[:, ::2, ::2]
    features = features[:, :, ::2, ::2]


    # 0: hw
    # 1: ij

    features_0 = features[0]
    features_1 = features[1]
    target_0 = target[0]
    target_1 = target[1]

    feat_height = features_0.size()[1]
    feat_width = features_0.size()[2]
    feat_depth = features_0.size()[0]

    # We exemplarily compare only image 0 and 1 in batch

    # Segmentation mask where L_hwik == 1 if l_hw == 1
    features_0 = features_0.unsqueeze(1).unsqueeze(1)
    features_1 = features_1.unsqueeze(3).unsqueeze(3)

    target_0 = target_0.unsqueeze(0).unsqueeze(0)
    target_1 = target_1.unsqueeze(0).unsqueeze(0)

    features_0 = features_0.repeat(1, feat_height, feat_width, 1, 1)
    features_1 = features_1.repeat(1, 1, 1, feat_height, feat_width)

    target_0 = target_0.repeat(feat_height, feat_width, 1, 1)
    target_1 = target_1.repeat(1, 1, feat_height, feat_width)

    # reshape to feat_depth x -1
    features_0 = features_0.view(feat_depth, -1)
    features_1 = features_1.view(feat_depth, -1)

    target_0 = target_0.view(1, -1)
    target_1 = target_1.view(1, -1)

    F_hwij = torch.nn.CosineSimilarity(dim=0, eps=1e-6)(features_0, features_1)
    F_hwij = F_hwij.view(feat_height, feat_width, feat_height, feat_width)

    L_hwij = (target_0 == target_1).int()
    L_hwij = L_hwij.view(feat_height, feat_width, feat_height, feat_width)

    # Option1: Punish F_hwij high values for L_hwij == 0
    # Option2: Encourage F_hwij low values for L_hwij != 0

    cost = 0
    cost += torch.mean((1 - F_hwij) * L_hwij)  # minimize cosine dissimiarity in places where classes are equal
    cost += torch.mean(F_hwij * (1 - L_hwij))  # minimize cosine similarity in places where classes are NOT equal

    return cost




def main():

    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")
    parser.add_argument('--target', type=str, choices=["sdf", "angle", "both"], help="define the target that is used")
    parser.add_argument('--stego', action="store_true", default=False, help="If True, applies stego loss")
    parser.add_argument('--visualize', action='store_true', help="visualize the dataset")

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)
    params.model.target = opt.target
    params.visualize = opt.visualize
    params.stego = opt.stego

    print("Batch size summed over all GPUs: ", params.model.batch_size_reg)
    
    if not params.main.disable_wandb:
        wandb.login()
        wandb.init(
            entity='jannik-zuern',
            project='autograph-regressor',
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
        #model = build_network(snapshot=None, backend='resnet101', num_channels=3, n_classes=3).to(params.model.device)
        model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=2).to(params.model.device)

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

    train_path = os.path.join(params.paths.dataroot, 'exp-08-01-23', "*", "train")
    val_path = os.path.join(params.paths.dataroot, 'exp-08-01-23', "*", "val")

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
        trainer.train(epoch)

        #if not params.main.disable_wandb:
        if epoch % 2 == 0:

            #eval_loss = trainer.eval(epoch)
            eval_loss = 0.0

            try:
                wandb_run_name = wandb.run.name
            except:
                wandb_run_name = "local_run"

            fname = 'regressor_{}_{:04d}-{:.5f}.pth'.format(wandb_run_name, epoch, eval_loss)
            checkpoint_path = os.path.join(params.paths.checkpoints, fname)
            print("Saving checkpoint to {}".format(checkpoint_path))

            torch.save(model.state_dict(), checkpoint_path)



if __name__ == '__main__':
    main()
