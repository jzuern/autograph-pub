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

from data.datasets import SuccessorRegressorDataset
from lanegnn.utils import ParamLib


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
            pos_enc = data["pos_enc"].cuda()
            rgb = data["rgb"].cuda()

            in_tensor = torch.cat([rgb, pos_enc], dim=1)

            sdf_target = sdf_target.unsqueeze(1)

            (pred, features) = self.model(in_tensor)
            pred = torch.nn.functional.interpolate(pred, size=sdf_target.shape[2:], mode='bilinear', align_corners=True)
            pred_sdf = torch.nn.Sigmoid()(pred)


            loss_dict = {
                'loss_sdf': torch.nn.BCELoss()(pred_sdf, sdf_target),
            }

            loss = sum(loss_dict.values())
            loss.backward()

            self.optimizer.step()

            if not self.params.main.disable_wandb:
                wandb.log({"train/loss": loss.item()})

            # Visualization
            if self.total_step % 10 == 0 and self.params.visualize:
                cv2.imshow("rgb", rgb[0].cpu().numpy().transpose(1, 2, 0))
                cv2.imshow("pos_enc", pos_enc[0].cpu().numpy().transpose(1, 2, 0))


                sdf_pred = pred_sdf[0, 0].detach().cpu().detach().numpy()
                sdf_target = sdf_target[0, 0].cpu().detach().numpy()

                sdf_pred = np.concatenate([sdf_pred[..., np.newaxis], sdf_pred[..., np.newaxis], sdf_pred[..., np.newaxis]], axis=2)
                sdf_target = np.concatenate([sdf_target[..., np.newaxis], sdf_target[..., np.newaxis], sdf_target[..., np.newaxis]], axis=2)
                sdf_target = (sdf_target > 0.3).astype(np.uint8) * 255

                print(sdf_pred.shape, sdf_target.shape)

                cv2.imshow("sdf_target", sdf_target)
                cv2.imshow("sdf_pred", sdf_pred)

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



    def inference(self):

        # Load model
        model_path = "checkpoints/reg_succ_local_run.pth"
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.eval()

        base_image = "/data/autograph/exp-successors/pittsburgh-pre/val/3-Pittsburgh-10300-27100-rgb.png"
        base_image = cv2.imread(base_image)

        cv2.imshow("test", np.random.randint(0, 255, (1, 1, 3)).astype(np.uint8))
        cv2.waitKey(1)

        # cv2 callback function for clicking on image
        def click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print("clicked at", x, y)
                q = [y, x]

                pos_encoding = np.zeros(base_image.shape, dtype=np.float32)
                x, y = np.meshgrid(np.arange(base_image.shape[1]), np.arange(base_image.shape[0]))
                pos_encoding[q[0], q[1], 0] = 1
                pos_encoding[..., 1] = np.abs((x - q[1])) / base_image.shape[1]
                pos_encoding[..., 2] = np.abs((y - q[0])) / base_image.shape[0]
                pos_encoding = (pos_encoding * 255).astype(np.uint8)

                cv2.imshow("base_image", base_image)
                cv2.imshow("pos_encoding", pos_encoding)
                cv2.waitKey(1)

                rgb = torch.from_numpy(base_image).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
                pos_enc = torch.from_numpy(pos_encoding).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255

                in_tensor = torch.cat([rgb, pos_enc], dim=1)

                (pred, features) = self.model(in_tensor)
                pred = torch.nn.functional.interpolate(pred, size=rgb.shape[2:], mode='bilinear', align_corners=True)
                pred_sdf = torch.nn.Sigmoid()(pred)

                pred_sdf = pred_sdf[0, 0].cpu().detach().numpy()
                pred_sdf = (pred_sdf * 255).astype(np.uint8)
                pred_sdf_viz = cv2.addWeighted(base_image, 0.5, cv2.applyColorMap(pred_sdf, cv2.COLORMAP_MAGMA), 0.5, 0)
                cv2.imshow("pred_sdf_viz", pred_sdf_viz)


        cv2.namedWindow("base_image")
        cv2.setMouseCallback("base_image", click)
        cv2.waitKey(-1)

        exit()





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
    model = DeepLabv3Plus(models.resnet101(pretrained=True), num_in_channels=6, num_classes=1).to(params.model.device)

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

    train_path = os.path.join(params.paths.dataroot, 'exp-successors', "*", "train")
    val_path = os.path.join(params.paths.dataroot, 'exp-successors', "*", "val")

    dataset_train = SuccessorRegressorDataset(path=train_path, split='train')
    dataset_val = SuccessorRegressorDataset(path=val_path, split='val')

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=params.model.batch_size_reg,
                                  num_workers=params.model.loader_workers,
                                  shuffle=True,
                                  drop_last=True)
    dataloader_val = DataLoader(dataset_val,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

    trainer = Trainer(params, model, dataloader_train, dataloader_val, optimizer)

    #trainer.inference()

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

            fname = 'reg_succ_{}.pth'.format(wandb_run_name)
            checkpoint_path = os.path.join(params.paths.checkpoints, fname)
            print("Saving checkpoint to {}".format(checkpoint_path))

            torch.save(model.state_dict(), checkpoint_path)

if __name__ == '__main__':
    main()

