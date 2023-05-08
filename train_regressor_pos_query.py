import os
import wandb
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
import torchvision.models as models
from torchmetrics import JaccardIndex, Precision, Recall, F1Score
import matplotlib.pyplot as plt
import cv2
from collections import OrderedDict
from matplotlib import cm
from aggregation.utils import AngleColorizer
from data.datasets import SuccessorRegressorDataset
from lanegnn.utils import ParamLib, make_image_grid
import glob


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


def torch_to_cv2(pred):
    pred = pred[0].detach().cpu().numpy()
    pred = np.concatenate([pred[..., np.newaxis], pred[..., np.newaxis], pred[..., np.newaxis]], axis=2)
    pred = (pred * 255).astype(np.uint8)

    return pred


def calc_torchmetrics_mask(seg_preds, seg_gts, name):
    precision = Precision(task="binary", average='macro', mdmc_average='global')
    recall = Recall(task="binary", average='none', mdmc_average='global')
    iou = JaccardIndex(task="binary", reduction='none', num_classes=2, ignore_index=0)
    f1 = F1Score(task="binary", average='none', mdmc_average='global', num_classes=2, ignore_index=0)

    # move all to CPU
    seg_preds = [seg_pred.cpu() for seg_pred in seg_preds]
    seg_gts = [seg_gt.cpu() for seg_gt in seg_gts]

    seg_preds = torch.round(torch.cat(seg_preds)).squeeze().int()
    seg_gts = torch.round(torch.cat(seg_gts)).squeeze().int()

    p = precision(seg_preds, seg_gts).numpy()
    r = recall(seg_preds, seg_gts).numpy()
    i = iou(seg_preds, seg_gts).numpy()
    f = f1(seg_preds, seg_gts).numpy()

    metrics = {
        'eval/mask_precision_{}'.format(name): p.item(),
        'eval/mask_recall_{}'.format(name): r.item(),
        'eval/mask_iou_{}'.format(name): i.item(),
        'eval/mask_f1_{}'.format(name): f.item(),
    }

    return metrics


def calc_torchmetrics_angles(angle_preds, angle_gts, name):
    # move all to CPU
    angle_preds = [angle_pred.cpu() for angle_pred in angle_preds]
    angle_gts = [angle_gt.cpu() for angle_gt in angle_gts]

    angle_preds = torch.cat(angle_preds).squeeze().float()
    angle_gts = torch.cat(angle_gts).squeeze().float()

    angle_preds = torch.atan2(angle_preds[:, 0], angle_preds[:, 1]) + np.pi
    angle_gts = torch.atan2(angle_gts[:, 0], angle_gts[:, 1]) + np.pi
    angle_preds = angle_preds % (2 * np.pi)
    angle_gts = angle_gts % (2 * np.pi)

    # bin angles into N bins
    N_bins = 8
    angle_preds = torch.round(angle_preds / (2 * np.pi / float(N_bins))).int()
    angle_gts = torch.round(angle_gts / (2 * np.pi / float(N_bins))).int()

    # calc metrics
    precision = Precision(task="multiclass", average='macro', num_classes=N_bins + 1)
    recall = Recall(task="multiclass", average='macro', num_classes=N_bins + 1)
    iou = JaccardIndex(task="multiclass", reduction='macro', num_classes=N_bins + 1)
    f1 = F1Score(task="multiclass", average='macro', num_classes=N_bins + 1)

    p = precision(angle_preds, angle_gts).numpy()
    r = recall(angle_preds, angle_gts).numpy()
    i = iou(angle_preds, angle_gts).numpy()
    f = f1(angle_preds, angle_gts).numpy()

    metrics = {
        'eval/angles_precision_{}'.format(name): p.item(),
        'eval/angles_recall_{}'.format(name): r.item(),
        'eval/angles_iou_{}'.format(name): i.item(),
        'eval/angles_f1_{}'.format(name): f.item(),
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
        self.ac = AngleColorizer()

        if self.params.stego:
            print("Using STEGO Loss")

        self.figure, self.axarr = plt.subplots(1, 2)

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

            if torch.all(data["rgb"][0, 0] == torch.zeros([256, 256])):
                print("skip in train loop")
                continue

            # mask_pedestrian = data["mask_pedestrian"].cuda()
            # drivable_gt = data["drivable_gt"].cuda()
            # pos_enc = data["pos_enc"].cuda()
            rgb = data["rgb"].cuda()

            if self.params.target == "full":
                in_tensor = rgb
                target_drivable = data["drivable"].cuda()
                target_angles = data["angles_xy"].cuda()

                (pred, features) = self.model(in_tensor)
                pred = torch.nn.functional.interpolate(pred, size=rgb.shape[2:], mode='bilinear', align_corners=True)

                pred_angles = torch.nn.Tanh()(pred[:, 0:2, :, :])
                pred_drivable = torch.nn.Sigmoid()(pred[:, 2, :, :])


                loss_dict = {
                    'train/loss_drivable': torch.nn.BCELoss()(pred_drivable, target_drivable),
                    'train/loss_angles': torch.nn.MSELoss()(pred_angles, target_angles),
                }

                loss_total = sum(loss_dict.values())

            elif self.params.target == "successor":
                if self.model_full is not None:
                    with torch.no_grad():
                        (pred, _) = self.model_full(rgb)  # get from model
                        pred = torch.nn.functional.interpolate(pred, size=rgb.shape[2:], mode='bilinear',
                                                               align_corners=True)
                        pred_angles = torch.nn.Tanh()(pred[:, 0:2, :, :])
                        pred_drivable = torch.nn.Sigmoid()(pred[:, 2, :, :])
                else:
                    in_tensor = rgb

                if self.params.input_layers == "rgb":  # rgb [3], pos_enc [3], pred_drivable [1], pred_angles [2]
                    # in_tensor = torch.cat([rgb, pos_enc], dim=1)
                    in_tensor = rgb
                elif self.params.input_layers == "rgb+drivable":
                    # in_tensor = torch.cat([rgb, pos_enc, pred_drivable.unsqueeze(1)], dim=1)
                    in_tensor = torch.cat([rgb, pred_drivable.unsqueeze(1)], dim=1)
                elif self.params.input_layers == "rgb+drivable+angles":
                    # in_tensor = torch.cat([rgb, pos_enc, pred_drivable.unsqueeze(1), pred_angles], dim=1)
                    in_tensor = torch.cat([rgb, pred_drivable.unsqueeze(1), pred_angles], dim=1)

                target_succ = data["mask_successor"].cuda()

                (pred_succ, features) = self.model(in_tensor)

                pred_succ = torch.nn.functional.interpolate(pred_succ, size=rgb.shape[2:], mode='bilinear',
                                                            align_corners=True)
                pred_succ = torch.nn.Sigmoid()(pred_succ[:, 0, :, :])

                loss_dict = {
                    'train/loss_succ': torch.nn.BCELoss()(pred_succ, target_succ),
                }

                loss_total = sum(loss_dict.values())

            loss_total.backward()

            self.optimizer.step()

            if not self.params.main.disable_wandb:
                wandb.log({"train/loss_total": loss_total.item()})
                [wandb.log({k: v.item()}) for k, v in loss_dict.items()]

            # Visualization
            if self.total_step % 10 == 0 and self.params.visualize:
                cv2.imshow("rgb", rgb[0].cpu().numpy().transpose(1, 2, 0))

                if self.params.target == "full":
                    pred_drivable = torch_to_cv2(pred_drivable)
                    gt_drivable = torch_to_cv2(target_drivable)

                    cv2.imshow("gt_drivable", gt_drivable)
                    cv2.imshow("pred_drivable", pred_drivable)

                    angles_gt_rad = self.ac.xy_to_angle(target_angles[0].cpu().detach().numpy())
                    angles_gt_color = self.ac.angle_to_color(angles_gt_rad)

                    angles_pred_rad = self.ac.xy_to_angle(pred_angles[0].cpu().detach().numpy())
                    angles_pred_color = self.ac.angle_to_color(angles_pred_rad)

                    cv2.imshow("angles_gt", angles_gt_color)
                    cv2.imshow("angles_pred", angles_pred_color)

                if self.params.target == "successor":
                    pred_succ = torch_to_cv2(pred_succ)
                    target_succ = torch_to_cv2(target_succ)
                    cv2.imshow("pred_succ", pred_succ)
                    cv2.imshow("target_succ", target_succ)

                    angles_pred_rad = self.ac.xy_to_angle(pred_angles[0].cpu().detach().numpy())
                    angles_pred_color = self.ac.angle_to_color(angles_pred_rad)

                    pred_drivable = torch_to_cv2(pred_drivable)
                    cv2.imshow("frozen pred_drivable", pred_drivable)
                    cv2.imshow("frozen angles_pred_color", angles_pred_color)

                cv2.waitKey(1)

            text = 'Epoch {} / {}, it {} / {}, it global {}, train loss = {:03f}'. \
                format(epoch, self.params.model.num_epochs, step + 1, len(self.dataloader_train),
                       epoch * len(self.dataloader_train) + step + 1, loss_total.item())
            train_progress.set_description(text)

            self.total_step += 1

        if not self.params.main.disable_wandb:
            wandb.log({"train/epoch": epoch})

    def evaluate_full(self, epoch):

        print("evaluate_full...")

        self.model.eval()

        val_losses = []

        mask_preds = []
        mask_targets = []
        mask_gts = []

        angle_preds = []
        angle_targets = []

        target_overlay_list_mask = []
        pred_overlay_list_mask = []
        target_overlay_list_angle = []
        pred_overlay_list_angle = []

        eval_progress = tqdm(self.dataloader_val)
        for step, data in enumerate(eval_progress):

            if torch.all(data["rgb"][0, 0] == torch.zeros([256, 256])):
                print("skip in train loop")
                continue

            rgb = data["rgb"].cuda()

            in_tensor = rgb
            target_drivable = data["drivable"].cuda()
            drivable_gt = data["drivable_gt"].cuda()
            target_angles = data["angles_xy"].cuda()

            (pred, features) = self.model(in_tensor)
            pred = torch.nn.functional.interpolate(pred, size=rgb.shape[2:], mode='bilinear', align_corners=True)
            pred_angles = torch.nn.Tanh()(pred[:, 0:2, :, :])
            pred_drivable = torch.nn.Sigmoid()(pred[:, 2, :, :])

            loss_dict = {
                'loss_drivable': torch.nn.BCELoss()(pred_drivable, target_drivable),
                'loss_angles': torch.nn.MSELoss()(pred_angles, target_angles),
            }

            loss_total = sum(loss_dict.values()).item()

            mask_preds.append(pred_drivable)
            mask_targets.append(target_drivable)
            mask_gts.append(drivable_gt)

            angle_preds.append(pred_angles)
            angle_targets.append(target_angles)

            # visualization
            rgb_viz = np.transpose(rgb.cpu().numpy()[0], (1, 2, 0))
            rgb_viz = (rgb_viz * 255.).astype(np.uint8)

            pred_viz = (cm.plasma(pred_drivable.cpu().detach().numpy()[0])[:, :, 0:3] * 255).astype(np.uint8)
            target_viz = (cm.plasma(target_drivable.cpu().detach().numpy()[0])[:, :, 0:3] * 255).astype(np.uint8)

            angles_gt_rad = self.ac.xy_to_angle(target_angles[0].cpu().detach().numpy())
            angles_gt_color = self.ac.angle_to_color(angles_gt_rad)

            angles_pred_rad = self.ac.xy_to_angle(pred_angles[0].cpu().detach().numpy())
            angles_pred_color = self.ac.angle_to_color(angles_pred_rad)

            target_overlay_angles = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5,
                                                    np.ascontiguousarray(angles_gt_color), 0.5, 0)
            pred_overlay_angles = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5,
                                                  np.ascontiguousarray(angles_pred_color), 0.5, 0)

            target_overlay_list_angle.append(target_overlay_angles)
            pred_overlay_list_angle.append(pred_overlay_angles)

            target_viz = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5, np.ascontiguousarray(target_viz), 0.5, 0)
            pred_viz = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5, np.ascontiguousarray(pred_viz), 0.5, 0)

            target_overlay_list_mask.append(target_viz)
            pred_overlay_list_mask.append(pred_viz)

            # Val loss
            val_losses.append(loss_total)

        val_loss = np.nanmean(val_losses)

        # Get metrics
        metrics_tracklet_drivable = calc_torchmetrics_mask(mask_preds, mask_targets, name="drivable_tracklet")
        metrics_gt_drivable = calc_torchmetrics_mask(mask_preds, mask_gts, name="drivable_gt")
        metrics_tracklet_angles = calc_torchmetrics_angles(angle_preds, angle_targets, name="angles_tracklet")

        # Make grid of images
        target_overlay_grid_mask = make_image_grid(target_overlay_list_mask, nrow=10, ncol=10)
        pred_overlay_grid_mask = make_image_grid(pred_overlay_list_mask, nrow=10, ncol=10)

        if self.params.target == "full":
            target_overlay_grid_angles = make_image_grid(target_overlay_list_angle, nrow=10, ncol=10)
            pred_overlay_grid_angles = make_image_grid(pred_overlay_list_angle, nrow=10, ncol=10)

        # Do logging
        if not self.params.main.disable_wandb:
            wandb.log({"eval/loss_total": val_loss})
            wandb.log(metrics_tracklet_drivable)
            wandb.log(metrics_gt_drivable)
            wandb.log({"Mask": [wandb.Image(target_overlay_grid_mask, caption="GT"),
                                wandb.Image(pred_overlay_grid_mask, caption="Pred")]})

            wandb.log(metrics_tracklet_angles)
            wandb.log({"Angles": [wandb.Image(target_overlay_grid_angles, caption="GT"),
                                  wandb.Image(pred_overlay_grid_angles, caption="Pred")]})

        cv2.imwrite("viz/target_overlay_grid_mask-e{:03d}.png".format(epoch), target_overlay_grid_mask)
        cv2.imwrite("viz/pred_overlay_grid_mask-e{:03d}.png".format(epoch), pred_overlay_grid_mask)

        cv2.imwrite("viz/target_overlay_grid_angles-e{:03d}.png".format(epoch), target_overlay_grid_angles)
        cv2.imwrite("viz/pred_overlay_grid_angles-e{:03d}.png".format(epoch), pred_overlay_grid_angles)

        print(metrics_tracklet_drivable)
        print(metrics_gt_drivable)

        return metrics_tracklet_drivable

    def evaluate_succ(self, epoch):
        print("evaluate_succ...")
        self.model.eval()

        val_losses = []

        mask_preds = []
        mask_targets = []
        mask_gts = []
        angle_preds = []
        angle_targets = []

        target_overlay_list_mask = []
        pred_overlay_list_mask = []
        target_overlay_list_angle = []
        pred_overlay_list_angle = []

        eval_progress = tqdm(self.dataloader_val)
        for step, data in enumerate(eval_progress):

            if torch.all(data["rgb"][0, 0] == torch.zeros([256, 256])):
                print("skip in eval loop")
                continue

            rgb = data["rgb"].cuda()

            if self.model_full is not None:
                with torch.no_grad():
                    (pred, _) = self.model_full(rgb)  # get from model
                    pred = torch.nn.functional.interpolate(pred, size=rgb.shape[2:], mode='bilinear', align_corners=True)
                    pred_angles = torch.nn.Tanh()(pred[:, 0:2, :, :])
                    pred_drivable = torch.nn.Sigmoid()(pred[:, 2, :, :])

                if self.params.input_layers == "rgb":  # rgb [3], pos_enc [3], pred_drivable [1], pred_angles [2]
                    # in_tensor = torch.cat([rgb, pos_enc], dim=1)
                    in_tensor = rgb
                elif self.params.input_layers == "rgb+drivable":
                    # in_tensor = torch.cat([rgb, pos_enc, pred_drivable.unsqueeze(1)], dim=1)
                    in_tensor = torch.cat([rgb, pred_drivable.unsqueeze(1)], dim=1)
                elif self.params.input_layers == "rgb+drivable+angles":
                    # in_tensor = torch.cat([rgb, pos_enc, pred_drivable.unsqueeze(1), pred_angles], dim=1)
                    in_tensor = torch.cat([rgb, pred_drivable.unsqueeze(1), pred_angles], dim=1)
            else:
                in_tensor = rgb

            target_succ = data["mask_successor"].cuda()

            (pred_succ, features) = self.model(in_tensor)
            pred_succ = torch.nn.functional.interpolate(pred_succ, size=rgb.shape[2:], mode='bilinear',
                                                        align_corners=True)

            pred_succ = torch.nn.Sigmoid()(pred_succ[:, 0, :, :])
            loss_dict = {
                'loss_succ': torch.nn.BCELoss()(pred_succ, target_succ),
            }
            loss_total = sum(loss_dict.values()).item()

            mask_preds.append(pred_succ)
            mask_targets.append(target_succ)

            # visualization
            rgb_viz = np.transpose(rgb.cpu().numpy()[0], (1, 2, 0))
            rgb_viz = (rgb_viz * 255.).astype(np.uint8)

            pred_viz = (cm.plasma(pred_succ.cpu().detach().numpy()[0])[:, :, 0:3] * 255).astype(np.uint8)
            target_viz = (cm.plasma(target_succ.cpu().detach().numpy()[0])[:, :, 0:3] * 255).astype(np.uint8)

            target_viz = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5, np.ascontiguousarray(target_viz), 0.5, 0)
            pred_viz = cv2.addWeighted(np.ascontiguousarray(rgb_viz), 0.5, np.ascontiguousarray(pred_viz), 0.5, 0)

            target_overlay_list_mask.append(target_viz)
            pred_overlay_list_mask.append(pred_viz)

            # Val loss
            val_losses.append(loss_total)

        val_loss = np.nanmean(val_losses)

        # Get metrics
        metrics_tracklet_succ = calc_torchmetrics_mask(mask_preds, mask_targets, name="successor_tracklet")

        # Make grid of images
        target_overlay_grid_mask = make_image_grid(target_overlay_list_mask, nrow=10, ncol=10)
        pred_overlay_grid_mask = make_image_grid(pred_overlay_list_mask, nrow=10, ncol=10)

        # Do logging
        if not self.params.main.disable_wandb:
            wandb.log({"eval/loss_total": val_loss})
            wandb.log(metrics_tracklet_succ)
            wandb.log({"Mask": [wandb.Image(target_overlay_grid_mask, caption="GT"),
                                wandb.Image(pred_overlay_grid_mask, caption="Pred")]})

        cv2.imwrite("viz/target_overlay_grid_succ-e{}.png".format(epoch), target_overlay_grid_mask)
        cv2.imwrite("viz/pred_overlay_grid_succ-e{}.png".format(epoch), pred_overlay_grid_mask)

        print(metrics_tracklet_succ)

        return metrics_tracklet_succ

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
        self.model.eval()

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
                    # in_tensor = rgb
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

    parser.add_argument('--dataset_name', type=str, help="which dataset to use for training")
    parser.add_argument('--target', type=str, help="which target to use for training", choices=["full", "successor"])
    parser.add_argument('--input_layers', type=str, help="which input layers to use for training",
                        choices=["rgb", "rgb+drivable", "rgb+drivable+angles"])
    parser.add_argument('--inference', action='store_true', help="perform inference instead of training")
    parser.add_argument('--full-checkpoint', type=str, default=None, help="path to full checkpoint for inference")
    parser.add_argument('--city', type=str, default="all", help="city to use for training")

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)
    params.visualize = opt.visualize
    params.stego = opt.stego
    params.target = opt.target
    params.input_layers = opt.input_layers
    params.dataset_name = opt.dataset_name
    params.city = opt.city

    print("Batch size summed over all GPUs: ", params.model.batch_size_reg)

    print("Params: ", params)

    if not params.main.disable_wandb:
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
        num_out_channels = 3  # drivable, angles
    elif opt.target == "successor":
        if opt.full_checkpoint is not None:
            if opt.input_layers == "rgb":   # rgb [3], pos_enc [3], pred_drivable [1], pred_angles [2]
                # num_in_channels = 6
                num_in_channels = 3
            elif opt.input_layers == "rgb+drivable":
                # num_in_channels = 7
                num_in_channels = 4
            elif opt.input_layers == "rgb+drivable+angles":
                # num_in_channels = 9
                num_in_channels = 6
            else:
                raise ValueError("Unknown input layers: ", opt.input_layers)
        else:
            # num_in_channels = 6  # rgb, pos_encoding
            num_in_channels = 3  # rgb
        num_out_channels = 1  # drivable
    else:
        raise ValueError("Unknown target")

    model = DeepLabv3Plus(models.resnet101(pretrained=True),
                          num_in_channels=num_in_channels,
                          num_classes=num_out_channels).to(params.model.device)

    model_full = None
    if opt.full_checkpoint is not None:
        model_full = DeepLabv3Plus(models.resnet101(pretrained=True),
                                   num_in_channels=3,
                                   num_classes=3).to(params.model.device)

        state_dict = torch.load(opt.full_checkpoint)
        try:
            model_full.load_state_dict(state_dict)
        except:
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module'
                new_state_dict[name] = v
            model_full.load_state_dict(new_state_dict)

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

    train_path = os.path.join(params.paths.dataroot, opt.dataset_name, "*", "train", "*")  # .../exp-name/city/split/branch-straight/*
    val_path = os.path.join(params.paths.dataroot, opt.dataset_name, "*", "eval", "*")
    #test_path = os.path.join(params.paths.dataroot, opt.dataset_name, "*", "test", "*")

    dataset_train = SuccessorRegressorDataset(params=params,
                                              path=train_path,
                                              split='train',
                                              max_num_samples=10000000)
    dataset_val = SuccessorRegressorDataset(params=params,
                                            path=val_path,
                                            split='eval',
                                            max_num_samples=1000)
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

        if epoch % 2 == 0:
            with torch.no_grad():
                if opt.target == "successor":
                    trainer.evaluate_succ(epoch)
                else:
                    trainer.evaluate_full(epoch)
            try:
                wandb_run_name = wandb.run.name
            except:
                if opt.target == "successor":
                    wandb_run_name = "local_run_successor"
                else:
                    wandb_run_name = "local_run_full"

            checkpoint_path = os.path.join(params.paths.checkpoints, wandb_run_name)

            os.makedirs(checkpoint_path, exist_ok=True)

            fname = 'e-{:03d}.pth'.format(epoch)
            checkpoint_name = os.path.join(checkpoint_path, fname)

            print("Saving checkpoint to {}".format(checkpoint_name))
            torch.save(model.state_dict(), checkpoint_name)


if __name__ == '__main__':
    main()

