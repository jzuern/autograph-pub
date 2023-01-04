import torch
import argparse
import numpy as np
import torch.nn as nn
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
import torchvision.models as models
from data.datasets import RegressorDataset
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Segmentation Training')
parser.add_argument('--path', help="path to datasets")

args = parser.parse_args()


def get_correlation(features_0, features_1):

    feat_height = features_0.size()[1]
    feat_width = features_0.size()[2]
    feat_depth = features_0.size()[0]

    # We exemplarily compare only image 0 and 1 in batch

    # Segmentation mask where L_hwik == 1 if l_hw == 1
    features_0 = features_0.unsqueeze(1).unsqueeze(1)
    features_1 = features_1.unsqueeze(3).unsqueeze(3)

    features_0 = features_0.repeat(1, feat_height, feat_width, 1, 1)
    features_1 = features_1.repeat(1, 1, 1, feat_height, feat_width)

    # reshape to feat_depth x -1
    features_0 = features_0.view(feat_depth, -1)
    features_1 = features_1.view(feat_depth, -1)

    F_hwij = torch.nn.CosineSimilarity(dim=0, eps=1e-6)(features_0, features_1)
    F_hwij = F_hwij.view(feat_height, feat_width, feat_height, feat_width)

    return F_hwij

# Load model

model = DeepLabv3Plus(models.resnet101(pretrained=False), num_classes=1).cuda()

# load checkpoint
#checkpoint = torch.load('checkpoints/no_logging/checkpoint_002.pth.tar')
#model.load_state_dict(checkpoint['state_dict'])


# Load image
train_dataset = RegressorDataset(path="/data/autograph/sparse/pittsburgh-post/train/", split='train')


train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=2,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=False,
                                           drop_last=False)


for i, data in enumerate(train_loader):

    # loss and optim
    sdf_target = data["sdf"].cuda()
    rgb = data["rgb"].cuda()
    angle_mask_target = data["angles_mask"].cuda()
    target_angle_x = data["angles_x"].cuda()
    target_angle_y = data["angles_y"].cuda()

    target_sdf = sdf_target.unsqueeze(1)
    target_angle = torch.cat([target_angle_x.unsqueeze(1), target_angle_y.unsqueeze(1)], dim=1)

    (_, features) = model(rgb)

    cv2.imshow("test", np.random.randint(0, 255, (256, 256, 3)).astype(np.uint8))
    cv2.waitKey(0)

    f0 = features[0]
    f1 = features[0]

    F_hwij = get_correlation(f0, f1)

    # Now we visualize image, and get mouse coordinates where clicked with opencv
    image_0 = rgb[0:1]
    image_1 = rgb[1:2]

    image_0 = np.ascontiguousarray(np.transpose(image_0[0].detach().cpu().numpy(), (1, 2, 0)))
    image_1 = np.ascontiguousarray(np.transpose(image_1[0].detach().cpu().numpy(), (1, 2, 0)))


    def viz(event, x, y, flags, param):
        global mouseX, mouseY
        if event == cv2.EVENT_LBUTTONDOWN:
            mouseX, mouseY = x, y

            # downscale the mouse coordinates to feature positions
            df = image_0.shape[0] / F_hwij.shape[0]
            df = int(df)
            mouseX = int(mouseX / df)
            mouseY = int(mouseY / df)
            correlation = F_hwij[mouseY, mouseX].cpu().detach().numpy()

            correlation = (correlation * 255).astype(np.uint8)
            correlation = cv2.applyColorMap(correlation, cv2.COLORMAP_JET)
            correlation = cv2.resize(correlation, (image_0.shape[1], image_0.shape[0]), interpolation=cv2.INTER_CUBIC)

            cv2.imshow('Correlation', correlation)
            cv2.waitKey(1)

    print("test")
    cv2.namedWindow('image_0')
    cv2.setMouseCallback('image_0', viz)
    cv2.imshow('image_0', image_0)
    cv2.imshow('image_1', image_1)
    cv2.waitKey(1)





