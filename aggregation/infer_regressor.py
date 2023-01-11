import os.path

import torch
import cv2
from regressors.build_net import build_network
from lanegnn.utils import visualize_angles
from glob import glob
from PIL import Image
import argparse
from tqdm import tqdm
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np


def get_id(filename):
    return '-'.join(os.path.basename(filename).split('-')[0:3])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path_root', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/regressor-newest.pth')
    args = parser.parse_args()

    #regressor = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=3).cuda()
    regressor = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=2).cuda()

    state_dict = torch.load(args.checkpoint)

    regressor.load_state_dict(state_dict)
    regressor.eval()

    sat_images = sorted(glob(os.path.join(args.out_path_root, "train", "*-rgb.png")) +
                        glob(os.path.join(args.out_path_root, "val", "*-rgb.png")))

    for sat_image_f in tqdm(sat_images):
        rgb = cv2.imread(sat_image_f)
        #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.FloatTensor(rgb).permute(2, 0, 1).unsqueeze(0).cuda() / 255.

        out_path = os.path.dirname(sat_image_f)
        sample_id = get_id(sat_image_f)

        (pred, features) = regressor(rgb_tensor)

        pred = torch.nn.functional.interpolate(pred, size=rgb_tensor.shape[2:], mode='bilinear', align_corners=True)

        #sdf = torch.nn.Sigmoid()(pred[0, 2]).detach().cpu().numpy()
        sdf = torch.nn.Sigmoid()(pred[0, 0]).detach().cpu().numpy()
        angles = torch.nn.Sigmoid()(pred[0, 1]).detach().cpu().numpy()

        angles_viz = visualize_angles(np.cos(angles),
                                      np.sin(angles),
                                      mask=sdf)

        plt.imshow(rgb)
        plt.imshow(sdf, alpha=0.5)
        plt.show()

        Image.fromarray(angles_viz).save("{}/{}-angles-reg.png".format(out_path, sample_id))
        Image.fromarray(sdf * 255.).convert("L").save("{}/{}-sdf-reg.png".format(out_path, sample_id))