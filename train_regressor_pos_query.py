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
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
import torchvision.models as models
from torchmetrics import JaccardIndex, Precision, Recall, F1Score
from collections import OrderedDict
from matplotlib import cm

from data.datasets import SuccessorRegressorDataset
from lanegnn.utils import ParamLib, make_image_grid
import glob


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


# Calculate metrics according to torchmetrics
precision = Precision(task="binary", average='none', mdmc_average='global')
recall = Recall(task="binary", average='none', mdmc_average='global')
iou = JaccardIndex(task="binary", reduction='none', num_classes=2, ignore_index=0)
f1 = F1Score(average='none', mdmc_average='global', num_classes=2, ignore_index=0)


def calc_torchmetrics(seg_preds, seg_gts, name):

    seg_preds = torch.round(torch.cat(seg_preds)).squeeze().cpu().int()
    seg_gts = torch.round(torch.cat(seg_gts)).squeeze().cpu().int()

    p = precision(seg_preds, seg_gts).numpy()
    r = recall(seg_preds, seg_gts).numpy()
    i = iou(seg_preds, seg_gts).numpy()
    f = f1(seg_preds, seg_gts).numpy()

    metrics = {
        'eval/precision_{}'.format(name): p.item(),
        'eval/recall_{}'.format(name): r.item(),
        'eval/iou_{}'.format(name): i.item(),
        'eval/f1_{}'.format(name): f[1],
    }

    return metrics




class Trainer():

    def __init__(self, params, model, dataloader_train, dataloader_val, optimizer, model_full=None):

        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.params = params
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.total_step = 0
        self.threshold_sdf = 0.3
        self.threshold_angle = np.pi / 4.
        self.model_full = model_full


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

            mask_pedestrian = data["mask_pedestrian"].cuda()
            pos_enc = data["pos_enc"].cuda()
            rgb = data["rgb"].cuda()

            if self.params.target == "full":
                in_tensor = rgb
                sdf_target = data["mask_full"].cuda()
            elif self.params.target == "successor":
                if self.model_full is not None:
                    with torch.no_grad():
                        (mask_full, _) = self.model_full(rgb)    # get from model
                    mask_full = torch.nn.functional.interpolate(mask_full, size=rgb.shape[2:], mode='bilinear',
                                                           align_corners=True)
                    mask_full = torch.nn.Sigmoid()(mask_full)
                else:
                    mask_full = data["mask_full"].unsqueeze(1)  # get from disk

                in_tensor = torch.cat([rgb, mask_full, pos_enc], dim=1)
                sdf_target = data["mask_successor"].cuda()

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

                if self.params.target == "successor":
                    cv2.imshow("pos_enc", pos_enc[0].cpu().numpy().transpose(1, 2, 0))
                    cv2.imshow("mask_full", mask_full[0, 0].cpu().numpy())

                sdf_pred = pred_sdf[0, 0].detach().cpu().detach().numpy()
                sdf_target = sdf_target[0, 0].cpu().detach().numpy()

                sdf_pred = np.concatenate([sdf_pred[..., np.newaxis], sdf_pred[..., np.newaxis], sdf_pred[..., np.newaxis]], axis=2)
                sdf_target = np.concatenate([sdf_target[..., np.newaxis], sdf_target[..., np.newaxis], sdf_target[..., np.newaxis]], axis=2)
                sdf_target = (sdf_target > 0.3).astype(np.uint8) * 255

                cv2.imshow("sdf_target", sdf_target)
                cv2.imshow("sdf_pred", sdf_pred)
                cv2.waitKey(1)

            text = 'Epoch {} / {}, it {} / {}, it global {}, train loss = {:03f}'.\
                format(epoch, self.params.model.num_epochs, step+1, len(self.dataloader_train), epoch * len(self.dataloader_train) + step+1, loss.item())
            train_progress.set_description(text)

            self.total_step += 1

        if not self.params.main.disable_wandb:
            wandb.log({"train/epoch": epoch})


    def evaluate(self, epoch):

        print("Evaluating...")

        self.model.eval()

        val_losses = []
        sdf_preds = []
        sdf_targets = []

        target_overlay_list = []
        pred_overlay_list = []

        eval_progress = tqdm(self.dataloader_val)
        for step, data in enumerate(eval_progress):

            mask_pedestrian = data["mask_pedestrian"].cuda()
            pos_enc = data["pos_enc"].cuda()
            rgb = data["rgb"].cuda()

            if self.params.target == "full":
                in_tensor = rgb
                sdf_target = data["mask_full"].cuda()
            elif self.params.target == "successor":
                if self.model_full is not None:
                    with torch.no_grad():
                        (mask_full, _) = self.model_full(rgb)    # get from model
                    mask_full = torch.nn.functional.interpolate(mask_full, size=rgb.shape[2:], mode='bilinear',
                                                           align_corners=True)
                    mask_full = torch.nn.Sigmoid()(mask_full)
                else:
                    mask_full = data["mask_full"].unsqueeze(1)  # get from disk

                in_tensor = torch.cat([rgb, mask_full, pos_enc], dim=1)
                sdf_target = data["mask_successor"].cuda()

            sdf_target = sdf_target.unsqueeze(1)

            with torch.no_grad():
                (pred, features) = self.model(in_tensor)
                pred = torch.nn.functional.interpolate(pred, size=sdf_target.shape[2:], mode='bilinear', align_corners=True)
                pred_sdf = torch.nn.Sigmoid()(pred)

            sdf_targets.append(sdf_target)
            sdf_preds.append(pred_sdf)


            # visualization
            pred_viz = (cm.plasma(pred_sdf.cpu().detach().numpy()[0, 0])[:, :, 0:3] * 255).astype(np.uint8)
            target_viz = (cm.plasma(sdf_target.cpu().detach().numpy()[0, 0])[:, :, 0:3] * 255).astype(np.uint8)

            rgb_viz = np.transpose(rgb.cpu().numpy()[0], (1, 2, 0))
            rgb_viz = (rgb_viz * 255.).astype(np.uint8)

            target_overlay = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5, np.ascontiguousarray(target_viz), 0.5, 0)
            pred_overlay = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5, np.ascontiguousarray(pred_viz), 0.5, 0)

            target_overlay_list.append(target_overlay)
            pred_overlay_list.append(pred_overlay)


            loss_dict = {
                'loss_sdf': torch.nn.BCELoss()(pred_sdf, sdf_target),
            }

            loss = sum(loss_dict.values()).cpu().numpy().item()
            val_losses.append(loss)

        val_loss = np.nanmean(val_losses)

        metrics_sdf = calc_torchmetrics(sdf_preds, sdf_targets, name="sdf")
        metrics_sdf["eval/loss"] = val_loss


        # Make grid of images
        target_overlay_grid = make_image_grid(target_overlay_list, nrow=8, ncol=8)
        pred_overlay_grid = make_image_grid(pred_overlay_list, nrow=8, ncol=8)

        if not self.params.main.disable_wandb:
            wandb.log(metrics_sdf)
            wandb.log({"Samples eval": [wandb.Image(target_overlay_grid, caption="GT"),
                                        wandb.Image(pred_overlay_grid, caption="Pred")],
                       })

        cv2.imwrite("viz/target_overlay_grid-e{}.png".format(epoch), target_overlay_grid)
        cv2.imwrite("viz/pred_overlay_grid-e{}.png".format(epoch), pred_overlay_grid)

        print(metrics_sdf)

        return metrics_sdf


    def inference(self):

        print("Inference...")

        # Load model
        model_path = "checkpoints/reg_succ_twinkling-horse-9-e267.pth"

        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)
        self.model = self.model.eval()

        base_images = glob.glob("/data/autograph/successors-pedestrians/pittsburgh-pre/train/*rgb.png")


        for base_image in base_images:
            base_image = cv2.imread(base_image)

            # cv2 callback function for clicking on image
            def click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    q = [y, x]

                    pos_encoding = np.zeros(base_image.shape, dtype=np.float32)
                    x, y = np.meshgrid(np.arange(base_image.shape[1]), np.arange(base_image.shape[0]))
                    pos_encoding[q[0], q[1], 0] = 1
                    pos_encoding[..., 1] = np.abs((x - q[1])) / base_image.shape[1]
                    pos_encoding[..., 2] = np.abs((y - q[0])) / base_image.shape[0]
                    pos_encoding = (pos_encoding * 255).astype(np.uint8)
                    pos_encoding = cv2.cvtColor(pos_encoding, cv2.COLOR_BGR2RGB)

                    rgb = torch.from_numpy(base_image).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
                    pos_enc = torch.from_numpy(pos_encoding).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255

                    in_tensor = torch.cat([rgb, pos_enc], dim=1)
                    #in_tensor = rgb
                    in_tensor = torch.cat([in_tensor, in_tensor], dim=0)

                    (pred, features) = self.model(in_tensor)
                    pred = torch.nn.functional.interpolate(pred, size=rgb.shape[2:], mode='bilinear',
                                                           align_corners=True)
                    pred_sdf = torch.nn.Sigmoid()(pred)

                    pred_sdf = pred_sdf[0, 0].cpu().detach().numpy()

                    pred_sdf = (pred_sdf * 255).astype(np.uint8)
                    pred_sdf_viz = cv2.addWeighted(base_image, 0.5, cv2.applyColorMap(pred_sdf, cv2.COLORMAP_MAGMA),
                                                   0.5, 0)
                    cv2.imshow("pred_sdf_viz", pred_sdf_viz)

            cv2.namedWindow("pred_sdf_viz")
            cv2.setMouseCallback("pred_sdf_viz", click)
            cv2.waitKey(-1)





def main():

    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")
    parser.add_argument('--stego', action="store_true", default=False, help="If True, applies stego loss")
    parser.add_argument('--visualize', action='store_true', help="visualize the dataset")
    parser.add_argument('--disable_wandb', '-d', action='store_true', help="disable wandb")
    parser.add_argument('--target', type=str, help="which target to use for training", choices=["full", "successor"])
    parser.add_argument('--inference', action='store_true', help="perform inference instead of training")
    parser.add_argument('--full-checkpoint', type=str, default=None, help="path to full checkpoint for inference")

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)
    params.visualize = opt.visualize
    params.stego = opt.stego
    params.target = opt.target

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
    if opt.target == "full":
        num_in_channels = 3  # rgb
    elif opt.target == "successor":
        num_in_channels = 7  # rgb, mask_sdf, pos_encoding
    else:
        raise ValueError("Unknown target")
    model = DeepLabv3Plus(models.resnet101(pretrained=True),
                          num_in_channels=num_in_channels,
                          num_classes=1).to(params.model.device)

    model_full = None
    if opt.full_checkpoint is not None:
        model_full = DeepLabv3Plus(models.resnet101(pretrained=True),
                                   num_in_channels=3,
                                   num_classes=1).to(params.model.device)
        model_full.load_state_dict(torch.load(opt.full_checkpoint))
        model_full.eval()


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

    train_path = os.path.join(params.paths.dataroot, 'successors-pedestrians', "*", "train")
    val_path = os.path.join(params.paths.dataroot, 'successors-pedestrians', "*", "val")

    dataset_train = SuccessorRegressorDataset(params=params, path=train_path, split='train')
    dataset_val = SuccessorRegressorDataset(params=params, path=val_path, split='val')

    dataloader_train = DataLoader(dataset_train,
                                  batch_size=params.model.batch_size_reg,
                                  num_workers=params.model.loader_workers,
                                  shuffle=True,
                                  drop_last=True)
    dataloader_val = DataLoader(dataset_val,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

    trainer = Trainer(params, model, dataloader_train, dataloader_val, optimizer, model_full=model_full)

    if opt.inference:
        trainer.inference()

    for epoch in range(params.model.num_epochs):

        # Evaluate
        trainer.train(epoch)

        #if not params.main.disable_wandb:
        if epoch % 2 == 0:

            trainer.evaluate(epoch)

            try:
                wandb_run_name = wandb.run.name
            except:
                if opt.target == "successor":
                    wandb_run_name = "local_run_successor"
                else:
                    wandb_run_name = "local_run_full"


            fname = 'reg_{}.pth'.format(wandb_run_name)
            checkpoint_path = os.path.join(params.paths.checkpoints, fname)
            print("Saving checkpoint to {}".format(checkpoint_path))

            torch.save(model.state_dict(), checkpoint_path)

if __name__ == '__main__':
    main()

