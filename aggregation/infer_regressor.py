import os.path

import torch
import cv2
from regressors.build_net import build_network
from aggregate_av2 import visualize_angles
from glob import glob
from PIL import Image
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path_root', type=str, default='')
    args = parser.parse_args()

    regressor = build_network(snapshot=None, backend='resnet101', num_channels=3, n_classes=3).cuda()
    regressor.load_state_dict(torch.load("../checkpoints/regressor-newest.pth"))
    regressor.train()

    sat_images = sorted(glob(os.path.join(args.out_path_root, "train", "*.png")) + glob(os.path.join(args.out_path_root, "val", "*.png")))

    for sat_image in sat_images:
        sat_image = cv2.imread(sat_image)

        # Apply filter to remove edges with low probability
        rgb = cv2.cvtColor(sat_image, cv2.COLOR_RGB2BGR)
        pred = regressor(torch.FloatTensor(rgb).permute(2, 0, 1).unsqueeze(0).cuda() / 255.)
        sdf = torch.nn.Sigmoid()(pred[0, 2]).detach().cpu().numpy()
        angles = torch.nn.Tanh()(pred[0, 0:2]).detach().cpu().numpy()

        angles_viz = visualize_angles(angles[0], angles[1], mask=sdf)

        Image.fromarray(angles_viz).save("{}/{}-angles-reg.png".format(out_path, sample_id))
        Image.fromarray(sdf * 255.).convert("L").save("{}/{}-sdf-reg.png".format(out_path, sample_id))