import cv2
import torch
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
from collections import OrderedDict
import torchvision.models as models
import numpy as np
from pynput.keyboard import Key, Listener
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2334477275000
import matplotlib.pyplot as plt


def colorize(mask):

    # normalize mask
    mask = np.log(mask + 1e-8)
    mask = mask - np.min(mask)
    mask = mask / np.max(mask)

    mask = (mask * 255.).astype(np.uint8)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_MAGMA)
    return mask


class SatelliteDriver(object):
    def __init__(self):
        self.satellite = None
        self.pose = np.array([2871.6, 1594.9, -11.60])
        self.current_crop = None
        self.model = None
        #self.crop_shape = (256, 256)
        self.crop_shape = (512, 512)
        self.canvas_log_odds = None
        self.pose_history = np.array([self.pose])

    def load_model(self, model_path, type=None):

        state_dict = torch.load(model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        if type == "full":
            self.model_full = DeepLabv3Plus(models.resnet101(pretrained=True),
                                           num_in_channels=3,
                                           num_classes=1).cuda()
            self.model_full.load_state_dict(new_state_dict)
            self.model_full.eval()

        elif type == "successor":
            self.model_succ = DeepLabv3Plus(models.resnet101(pretrained=True),
                                           num_in_channels=7,
                                           num_classes=1).cuda()
            self.model_succ.load_state_dict(new_state_dict)
            self.model_succ.eval()

        print("Model loaded")

    def load_satellite(self, path):
        self.satellite = np.asarray(Image.open(path)).astype(np.uint8)
        self.satellite = cv2.cvtColor(self.satellite, cv2.COLOR_BGR2RGB)

        # Crop
        top_left = [27000, 17000]  # vertical, horizontal
        delta = [5000, 5000]
        self.satellite = self.satellite[top_left[0]:top_left[0] + delta[0], top_left[1]:top_left[1] + delta[1], :]

        self.canvas_log_odds = np.ones([self.satellite.shape[0], self.satellite.shape[1]], dtype=np.float32)

        print("Satellite loaded")

    def generate_pos_encoding(self):
        q = [255, 127]

        pos_encoding = np.zeros([self.crop_shape[0], self.crop_shape[1], 3], dtype=np.float32)
        x, y = np.meshgrid(np.arange(self.crop_shape[1]), np.arange(self.crop_shape[0]))
        pos_encoding[q[0], q[1], 0] = 1
        pos_encoding[..., 1] = np.abs((x - q[1])) / self.crop_shape[1]
        pos_encoding[..., 2] = np.abs((y - q[0])) / self.crop_shape[0]
        pos_encoding = (pos_encoding * 255).astype(np.uint8)
        pos_encoding = cv2.cvtColor(pos_encoding, cv2.COLOR_BGR2RGB)

        return pos_encoding


    def add_pred_to_canvas(self, pred, pose):


        x, y, yaw = pose

        csize = self.crop_shape[0]
        half_csize = csize // 2

        # For bottom centered
        dst_pts = np.array([[-half_csize, 0],
                            [-half_csize, -csize+1],
                            [half_csize-1, -csize+1],
                            [half_csize-1, 0]])

        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])

        center = np.array([x, y]) - np.array([csize, csize])

        # Rotate dst points
        dst_pts = (np.matmul(R, dst_pts.T).T + center).astype(np.float32)

        # source points are simply the corner points
        src_pts = np.array([[0, csize - 1],
                            [0, 0],
                            [csize - 1, 0],
                            [csize - 1, csize - 1]],
                           dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        pred_roi = (np.ones_like(pred) * 255).astype(np.uint8)

        warped_pred = cv2.warpPerspective(pred, M,
                                          (self.canvas_log_odds.shape[0], self.canvas_log_odds.shape[1]),
                                          cv2.INTER_LINEAR)
        warped_roi = cv2.warpPerspective(pred_roi, M,
                                          (self.canvas_log_odds.shape[0], self.canvas_log_odds.shape[1]),
                                          cv2.INTER_NEAREST)

        warped_roi = warped_roi.astype(np.float32) / 255.  # 1 for valid, 0 for invalid
        warped_roi[warped_roi < 0.5] = 0.5
        warped_roi[warped_roi >= 0.5] = 1
        # TODO: make this proper bayesian

        self.canvas_log_odds += warped_pred

        # resize to smaller
        df = self.canvas_log_odds.shape[0] / 1000
        img1 = cv2.resize(colorize(self.canvas_log_odds), (1000, 1000))
        img2 = cv2.resize(self.satellite, (1000, 1000))
        canvas_viz = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        for i in range(1, len(self.pose_history)-1):
            x_0, y_0, _ = self.pose_history[i-1]
            x_1, y_1, _ = self.pose_history[i]

            x_0 -= csize
            x_1 -= csize
            y_0 -= csize
            y_1 -= csize

            x_0 = int(x_0 / df)
            x_1 = int(x_1 / df)
            y_0 = int(y_0 / df)
            y_1 = int(y_1 / df)

            cv2.line(canvas_viz, (x_0, y_0), (x_1, y_1), (0, 145, 0), 1, cv2.LINE_AA)

        cv2.imshow("Aggregation", canvas_viz)

        # also visualize the canvas around current pose
        width = 500
        x_1, y_1, _ = self.pose_history[-1]
        x_1 -= csize
        y_1 -= csize
        x_1 = int(x_1)
        y_1 = int(y_1)

        canvas_roi = self.canvas_log_odds[y_1 - width//2 : y_1+width//2, x_1-width//2 : x_1+width//2]
        canvas_roi = colorize(canvas_roi)
        satellite_roi = self.satellite[y_1 - width//2 : y_1+width//2, x_1-width//2 : x_1+width//2, :]
        viz_roi = cv2.addWeighted(canvas_roi, 0.5, satellite_roi, 0.5, 0)

        cv2.imshow("Aggregation_cropped", viz_roi)


        # warped_log_odds = np.log(warped_pred / (1 - warped_pred))
        # warped_log_odds += warped_roi
        #
        # self.canvas_log_odds += warped_log_odds
        #
        # canvas_odds = np.exp(self.canvas_log_odds)
        # canvas_probs = canvas_odds / (1 + canvas_odds)
        #
        # cv2.imshow("warped_pred", colorize(warped_pred))
        # cv2.imshow("canvas", colorize(canvas_probs))
        #
        #
        #
        # canvas = 1 / (1 + np.exp(-self.canvas_log_odds))
        #
        # canvas_viz = colorize(canvas)
        # canvas_viz = cv2.addWeighted(self.satellite, 0.5, canvas_viz, 0.5, 0)
        #
        # cv2.imshow("Aggregation", canvas_viz)



    def crop_satellite_at_pose(self, pose):

        x, y, yaw = pose

        csize = self.crop_shape[0]
        csize_half = csize // 2

        satellite_image = self.satellite[int(y - csize * 2): int(y + csize * 2),
                                         int(x - csize * 2): int(x + csize * 2)].copy()

        # For bottom centered
        src_pts = np.array([[-csize_half, 0],
                            [-csize_half, -csize+1],
                            [csize_half-1, -csize+1],
                            [csize_half-1, 0]])

        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])

        center = np.array([csize, csize])

        # Rotate source points
        src_pts = (np.matmul(R, src_pts.T).T + center).astype(np.float32)

        # Destination points are simply the corner points
        dst_pts = np.array([[0, csize - 1],
                            [0, 0],
                            [csize - 1, 0],
                            [csize - 1, csize - 1]],
                           dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        try:
            rgb = cv2.warpPerspective(satellite_image, M, (csize, csize), cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        except:
            print("Error in warpPerspective. Resetting position")
            self.pose = np.array([2871.6, 1594.9, -11.60])
            rgb = self.crop_satellite_at_pose(self.pose)

        return rgb


    def drive_step(self,key):

        print("Pose x, y, yaw: {:.1f}, {:.1f}, {:.2f}".format(self.pose[0], self.pose[1], self.pose[2]))

        if self.pose[2] > 2 * np.pi:
            self.pose[2] -= 2 * np.pi
        if self.pose[2] < -2 * np.pi:
            self.pose[2] += 2 * np.pi

        # alter pose based on which arrow key is pressed
        s = 20

        forward_vector = np.array([np.cos(self.pose[2]),
                                  -np.sin(self.pose[2])])

        # arrow key pressed
        if key == Key.up:
            # go forward
            delta = s * forward_vector
            self.pose[0:2] -= np.array([delta[1], delta[0]])
        elif key == Key.down:
            # go backward
            delta = s * forward_vector
            self.pose[0:2] += np.array([delta[1], delta[0]])
        elif key == Key.left:
            # rotate left
            self.pose[2] -= 0.2
        elif key == Key.right:
            self.pose[2] += 0.2

        self.pose_history = np.concatenate([self.pose_history, [self.pose]])

        pos_encoding = self.generate_pos_encoding()
        rgb = self.crop_satellite_at_pose(self.pose)

        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
        pos_encoding_torch = torch.from_numpy(pos_encoding).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255

        if self.model_full is not None:
            with torch.no_grad():
                (tracklet_image_torch, _) = self.model_full(rgb_torch)
                tracklet_image_torch = torch.nn.functional.interpolate(tracklet_image_torch,
                                                                       size=rgb_torch.shape[2:],
                                                                       mode='bilinear',
                                                                       align_corners=True)
                tracklet_image_torch = torch.nn.Sigmoid()(tracklet_image_torch)

        in_tensor = torch.cat([rgb_torch, tracklet_image_torch, pos_encoding_torch], dim=1)
        in_tensor = torch.cat([in_tensor, in_tensor], dim=0)

        (pred, features) = self.model_succ(in_tensor)
        pred = torch.nn.functional.interpolate(pred,
                                               size=rgb_torch.shape[2:],
                                               mode='bilinear',
                                               align_corners=True)
        pred_sdf = torch.nn.Sigmoid()(pred)
        pred_sdf = pred_sdf[0, 0].cpu().detach().numpy()

        self.add_pred_to_canvas(pred_sdf, self.pose)

        pred_sdf = (pred_sdf * 255).astype(np.uint8)
        pred_sdf_viz = cv2.addWeighted(rgb, 0.5, cv2.applyColorMap(pred_sdf, cv2.COLORMAP_MAGMA), 0.5, 0)
        cv2.imshow("pred_sdf_viz", pred_sdf_viz)
        cv2.waitKey(1)


if __name__ == "__main__":
    driver = SatelliteDriver()
    driver.load_satellite(path="/data/lanegraph/woven-data/Pittsburgh.png")
    #driver.load_model(model_path="checkpoints/reg_major-rain-24.pth", type="full")
    #driver.load_model(model_path="checkpoints/reg_misty-resonance-25.pth", type="successor")
    driver.load_model(model_path="checkpoints/resilient-bush-32/e-150.pth", type="full")
    driver.load_model(model_path="checkpoints/noble-bee-34/model.pth", type="successor")

    print("Press arrow keys to drive")

    def on_press(key):
        driver.drive_step(key)

    def on_release(key):
        if key == Key.esc:
            return False

    # Collect events until released
    with Listener(on_press=on_press) as listener:
        listener.join()

