import cv2
import torch
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
from collections import OrderedDict
import torchvision.models as models
import numpy as np
from pynput.keyboard import Key, Listener, Controller
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
from tqdm import tqdm
import time
import pickle
import glob
import os
from aggregation.utils import similarity_check, out_of_bounds_check, AngleColorizer
from utils import aggregate, colorize, skeleton_to_graph, skeletonize_prediction, roundify_skeleton_graph
import sys
import json

keyboard = Controller()


# SETTINGS

skeleton_threshold = 0.08  # 0.08  # threshold for skeletonization
edge_start_idx = 10        # start index for selecting edge as future pose
edge_end_idx = 50          # end index for selecting edge as future pose
write_every = 1           # write to disk every n steps
waitkey_ms = 1
step_size = 40           # step size along ego edge in pixels
max_edge_length = 100    # max length of any graph edge in pixels
succ_graph_weight_threshold = 50  # threshold for successor graph edge weight below which we abandon branch


# CVPR graph aggregation
threshold_px = 30
threshold_rad = 0.2
closest_lat_thresh = 30

init_poses = json.load(open('starting_poses.json', 'r'))

def move_graph_nodes(g, delta):
    g_ = g.copy(as_view=False)
    for node in g_.nodes():
        g_.nodes[node]['pos'] += delta
    return g_


class AerialDriver(object):
    def __init__(self, debug=False, input_layers=None, tile_id=None, data_source=None):
        self.aerial_image = None

        my_init_poses = np.array(init_poses[tile_id])
        my_init_poses = my_init_poses + np.array([500, 500, 0])
        self.init_pose = my_init_poses[0]

        self.future_poses = []

        if len(init_poses[tile_id]) > 1:
            self.future_poses = list(my_init_poses[1:])

        self.future_poses_destinations = {}

        self.pose = self.init_pose.copy()

        self.current_branch_age = 0

        self.current_crop = None
        self.model = None
        self.time = time.time()
        self.debug = debug

        self.input_layers = input_layers
        self.crop_shape = (256, 256)
        self.graphs = []  # list of graphs from each step

        self.canvas_log_odds = None
        self.canvas_angles = None

        self.pose_history = np.array([self.pose])
        self.ac = AngleColorizer()

        self.step = 0
        self.graph_skeleton = None

        self.data_source = data_source

        self.G_agg_naive = nx.DiGraph()

        self.done = False # flag to indicate end of episode



    def load_model(self, model_path, type=None, input_layers="rgb"):

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
                                            num_classes=3).cuda()
            self.model_full.load_state_dict(new_state_dict)
            self.model_full.eval()

        elif type == "successor":
            if input_layers == "rgb":  # rgb [3], pos_enc [3], pred_drivable [1], pred_angles [2]
                num_in_channels = 3
            elif input_layers == "rgb+drivable":
                num_in_channels = 4
            elif input_layers == "rgb+drivable+angles":
                num_in_channels = 6
            else:
                raise NotImplementedError

            self.model_succ = DeepLabv3Plus(models.resnet101(pretrained=True),
                                            num_in_channels=num_in_channels,
                                            num_classes=1).cuda()
            self.model_succ.load_state_dict(new_state_dict)
            self.model_succ.eval()

        print("Model {} loaded".format(model_path))

    def load_satellite(self, impath):
        print("Loading aerial image {}".format(impath))
        self.aerial_image = np.asarray(Image.open(impath)).astype(np.uint8)
        self.tile_id = impath.split("/")[-1].split(".")[0]
        self.city_name = self.tile_id.split("_")[0]
        print("Tile ID: {}".format(self.tile_id))
        print("City: {}".format(self.city_name))

        self.dumpdir = "/data/autograph/evaluations/G_agg/{}/{}".format(self.data_source, self.tile_id)
        os.makedirs(self.dumpdir, exist_ok=True)
        os.makedirs(self.dumpdir + "/debug/", exist_ok=True)


        if os.path.exists("{}/G_agg_naive_cleanup.pickle".format(self.dumpdir)):
            print("G_agg_naive_cleanup already exists. Exiting.")
            sys.exit(0)

        # Embed the aerial image into a larger image to avoid edge effects
        self.aerial_image = np.pad(self.aerial_image, ((500, 500), (500, 500),
                                                       (0, 0)), mode="constant", constant_values=0)

        print("Aerial image shape: {}".format(self.aerial_image.shape))

        self.aerial_image = cv2.cvtColor(self.aerial_image, cv2.COLOR_BGR2RGB)
        self.canvas_log_odds = np.ones([self.aerial_image.shape[0], self.aerial_image.shape[1]], dtype=np.float32)
        self.canvas_angles = np.zeros([self.aerial_image.shape[0], self.aerial_image.shape[1], 3], dtype=np.uint8)

        self.visualize_write_G_agg(self.G_agg_naive, "G_agg_naive")


    def generate_pos_encoding(self):
        q = [self.crop_shape[0]-1,
             self.crop_shape[1]//2 - 1]

        pos_encoding = np.zeros([self.crop_shape[0], self.crop_shape[1], 3], dtype=np.float32)
        x, y = np.meshgrid(np.arange(self.crop_shape[1]), np.arange(self.crop_shape[0]))
        pos_encoding[q[0], q[1], 0] = 1
        pos_encoding[..., 1] = np.abs((x - q[1])) / self.crop_shape[1]
        pos_encoding[..., 2] = np.abs((y - q[0])) / self.crop_shape[0]
        pos_encoding = (pos_encoding * 255).astype(np.uint8)
        pos_encoding = cv2.cvtColor(pos_encoding, cv2.COLOR_BGR2RGB)

        return pos_encoding

    def pose_to_transform(self):

        x, y, yaw = self.pose

        csize = self.crop_shape[0]
        csize_half = csize // 2

        # For bottom centered
        src_pts = np.array([[-csize_half, 0],
                            [-csize_half, -csize+1],
                            [csize_half-1, -csize+1],
                            [csize_half-1, 0]])

        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw), np.cos(yaw)]])

        center = np.array([x, y])

        # Rotate source points
        src_pts = (np.matmul(R, src_pts.T).T + center).astype(np.float32)

        # Destination points are simply the corner points
        dst_pts = np.array([[0, csize - 1],
                            [0, 0],
                            [csize - 1, 0],
                            [csize - 1, csize - 1]],
                           dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        return M


    def add_pred_to_canvas(self, pred):

        M = np.linalg.inv(self.pose_to_transform())

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


        self.canvas_log_odds += warped_pred

        # resize to smaller
        df = self.canvas_log_odds.shape[0] / 1500
        img1 = cv2.resize(colorize(self.canvas_log_odds), (1500, 1500))
        img2 = cv2.resize(self.aerial_image, (1500, 1500))
        canvas_viz = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        for iter, p in enumerate(self.pose_history):
            x_0, y_0, _ = p
            x_0 = int(x_0 / df)
            y_0 = int(y_0 / df)
            cv2.circle(canvas_viz, (x_0, y_0), 3, (0, 255, 0), -1)

    def crop_satellite_at_pose(self, pose):

        M = self.pose_to_transform()
        aerial_image = self.aerial_image.copy()

        try:
            rgb = cv2.warpPerspective(aerial_image, M, (self.crop_shape[0], self.crop_shape[1]),
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        except:
            print("Error in warpPerspective. Resetting position")
            self.pose = self.init_pose
            rgb = self.crop_satellite_at_pose(self.pose)

        self.current_crop = rgb

        return rgb


    def add_graph_to_angle_canvas(self):

        g = self.graph_skeleton

        angle_canvas_cropped = np.zeros(self.crop_shape).astype(np.float32)
        angle_indicator = np.zeros(self.crop_shape).astype(np.float32)

        # fill angle canvas with graph g
        for (s, e) in g.edges():
            edge_points = g[s][e]['pts']

            for i in range(len(edge_points) - 1):
                x1 = edge_points[i][1]
                y1 = edge_points[i][0]
                x2 = edge_points[i + 1][1]
                y2 = edge_points[i + 1][0]
                angle = np.arctan2(y2 - y1, x2 - x1) + np.pi
                angle = angle + self.pose[2]
                angle = angle % (2 * np.pi)
                cv2.line(angle_canvas_cropped, (int(x1), int(y1)), (int(x2), int(y2)), angle, thickness=10, lineType=cv2.LINE_8)
                cv2.line(angle_indicator, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness=10,  lineType=cv2.LINE_8)

        M = np.linalg.inv(self.pose_to_transform())

        angle_canvas_cropped_c = self.ac.angle_to_color(angle_canvas_cropped)
        angle_canvas_cropped_c = angle_canvas_cropped_c * np.expand_dims(angle_indicator, axis=2)

        # cv2.imshow("angles_colorized", angle_canvas_cropped_c)
        warped_angles = cv2.warpPerspective(angle_canvas_cropped_c, M,
                                            (self.canvas_angles.shape[0], self.canvas_angles.shape[1]),
                                            cv2.INTER_LINEAR)

        info_available = np.sum(warped_angles, axis=2) > 0

        self.canvas_angles[info_available] = warped_angles[info_available]


    def render_poses_in_aerial(self):
        rgb_pose_viz = self.aerial_image.copy()
        arrow_len = 40

        for pose in self.pose_history:
            # render pose as arrow
            y = pose[0] + 1
            x = pose[1] + 1
            theta = pose[2]
            x2 = x - arrow_len * np.cos(theta)
            y2 = y + arrow_len * np.sin(theta)
            cv2.arrowedLine(rgb_pose_viz, (int(y), int(x)), (int(y2), int(x2)), (255, 0, 0), 1, cv2.LINE_AA)

        for pose in self.future_poses:
            # render pose as arrow
            y = pose[0] - 1
            x = pose[1] - 1
            theta = pose[2]
            x2 = x - arrow_len * np.cos(theta)
            y2 = y + arrow_len * np.sin(theta)
            cv2.arrowedLine(rgb_pose_viz, (int(y), int(x)), (int(y2), int(x2)), (255, 255, 0), 1, cv2.LINE_AA)

        # crop around ego pose
        x = int(self.pose[0])
        y = int(self.pose[1])

        x1 = x - 500
        x2 = x + 500
        y1 = y - 500
        y2 = y + 500

        rgb_pose_viz = rgb_pose_viz[y1:y2, x1:x2]

        # cv2.imshow("rgb_pose_viz", rgb_pose_viz)
        cv2.imwrite("/data/autograph/evaluations/other/{}/{}-{:04d}_rgb_pose_viz.png".format(self.data_source, self.tile_id, self.step), rgb_pose_viz)

    def make_step(self):

        """Run one step of the driving loop."""

        self.pose_history = np.concatenate([self.pose_history, [self.pose]])
        rgb = self.crop_satellite_at_pose(self.pose)
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255

        with torch.no_grad():
            (pred, _) = self.model_full(rgb_torch)
            pred = torch.nn.functional.interpolate(pred,
                                                   size=rgb_torch.shape[2:],
                                                   mode='bilinear',
                                                   align_corners=True)
            pred_angles = torch.nn.Tanh()(pred[0:1, 0:2, :, :])
            pred_drivable = torch.nn.Sigmoid()(pred[0:1, 2:3, :, :])


        if self.input_layers == "rgb":  # rgb [3], pos_enc [3], pred_drivable [1], pred_angles [2]
            in_tensor = rgb_torch
        elif self.input_layers == "rgb+drivable":
            in_tensor = torch.cat([rgb_torch, pred_drivable], dim=1)
        elif self.input_layers == "rgb+drivable+angles":
            in_tensor = torch.cat([rgb_torch, pred_drivable, pred_angles], dim=1)
        else:
            raise ValueError("Unknown input layers: ", self.input_layers)

        (pred_succ, features) = self.model_succ(in_tensor)
        pred_succ = torch.nn.functional.interpolate(pred_succ,
                                                    size=rgb_torch.shape[2:],
                                                    mode='bilinear',
                                                    align_corners=True)

        pred_succ = torch.nn.Sigmoid()(pred_succ)
        pred_succ = pred_succ[0, 0].cpu().detach().numpy()
        pred_drivable = pred_drivable[0, 0].cpu().detach().numpy()

        cv2.imshow("pred_succ", pred_succ)

        skeleton = skeletonize_prediction(pred_succ, threshold=skeleton_threshold)

        self.skeleton = skeleton

        self.pred_succ = pred_succ
        self.pred_drivable = pred_drivable
        self.graph_skeleton = skeleton_to_graph(skeleton)

        for edge in self.graph_skeleton.edges():
            self.graph_skeleton.edges[edge]['pts'] = self.graph_skeleton.edges[edge]['pts'][:, ::-1]


        # make skeleton fatter
        skeleton = skeleton.astype(np.uint8) * 255
        skeleton = cv2.dilate(skeleton, np.ones((3, 3), np.uint8), iterations=1)
        skeleton = (skeleton / 255.0).astype(np.float32)

        pred_angles = self.ac.xy_to_angle(pred_angles[0].cpu().detach().numpy())
        pred_angles_succ_color = self.ac.angle_to_color(pred_angles, mask=pred_succ > skeleton_threshold)
        pred_angles_color = self.ac.angle_to_color(pred_angles, mask=pred_drivable > 0.3)

        cv2.imshow("skeleton", skeleton)
        cv2.imshow("pred_angles_color", pred_angles_color)
        cv2.imshow("rgb", rgb)

        self.add_pred_to_canvas(skeleton)

        pred_succ = (pred_succ * 255).astype(np.uint8)
        pred_succ_viz = cv2.addWeighted(rgb, 0.5, cv2.applyColorMap(pred_succ, cv2.COLORMAP_MAGMA), 0.5, 0)

        # draw edges by pts
        for (s, e) in self.graph_skeleton.edges():
            ps = self.graph_skeleton[s][e]['pts']
            for i in range(len(ps) - 1):
                cv2.arrowedLine(pred_succ_viz, (int(ps[i][0]), int(ps[i][1])), (int(ps[i + 1][0]), int(ps[i + 1][1])), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.arrowedLine(pred_succ_viz, (int(ps[0][0]), int(ps[0][1])), (int(ps[-1][0]), int(ps[-1][1])), (255, 0, 255), 1, cv2.LINE_AA)

        # draw nodes
        nodes = self.graph_skeleton.nodes()
        node_positions = np.array([nodes[i]['o'] for i in nodes])
        [cv2.circle(pred_succ_viz, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1) for p in node_positions]


        skeleton_drivable_weight = np.sum(skeleton * pred_drivable)
        skeleton_succ_weight = np.sum(skeleton * pred_succ / 255.)

        if self.debug:
            fig, axarr = plt.subplots(1, 5, figsize=(20, 5), sharex=True, sharey=True)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            [axi.set_axis_off() for axi in axarr.ravel()]

            axarr[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            axarr[0].title.set_text('RGB')
            axarr[1].imshow(pred_drivable, cmap='gray')
            axarr[1].title.set_text('Drivable')
            axarr[2].imshow(pred_angles_color)
            axarr[2].title.set_text('Angles')
            axarr[3].imshow(pred_succ, cmap='gray')
            axarr[3].title.set_text('Successor Heatmap')
            axarr[4].imshow(skeleton, cmap='gray')
            axarr[4].title.set_text('Successor Skeleton')
            plt.savefig("/data/autograph/evaluations/G_agg/{}/{}/debug/{:04d}.png".format(self.data_source,
                                                                                          self.tile_id,
                                                                                          self.step))
            plt.close(fig)

        # cv2.imwrite("/data/autograph/evaluations/debug/{}-{:04d}_pred_succ_viz.png".format(self.tile_id, self.step), pred_succ_viz)

        self.step += 1

    def aggregate_graphs(self, graphs):


        # relabel nodes according to a global counter
        graphs_relabel = []
        global_counter = 0
        for G in graphs:
            relabel_dict = {}
            for n in G.nodes():
                relabel_dict[n] = global_counter
                global_counter += 1
            G = nx.relabel_nodes(G, relabel_dict)
            graphs_relabel.append(G)

        graphs = graphs_relabel

        #
        # fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
        #
        # [visualize_graph(G, ax=ax[0]) for G in graphs]
        # ax[0].set_title("Graphs to aggregate")

        # Aggregate all graphs
        G_pred_agg = nx.DiGraph()
        for pred_agg_idx, G in tqdm(enumerate(graphs), total=len(graphs), desc="Aggregating graphs"):
            G_pred_agg, merging_map = aggregate(G_pred_agg, G,
                                                visited_edges=[],
                                                threshold_px=threshold_px,
                                                threshold_rad=threshold_rad,
                                                closest_lat_thresh=closest_lat_thresh,
                                                w_decay=False,
                                                remove=False)

        # visualize_graph(G_pred_agg, ax=ax[1])
        # ax[1].set_title("Aggregated Graph")
        # plt.show()

        return G_pred_agg



    def yaw_check(self, yaw):
        if yaw > 2 * np.pi:
            yaw -= 2 * np.pi
        if yaw < 0:
            yaw += 2 * np.pi
        return yaw


    def visualize_write_G_single(self, graphs, name="G"):

        G_agg_viz = self.aerial_image.copy()
        G_agg_viz = G_agg_viz // 2

        # history colors linearly interpolated
        colors = matplotlib.cm.get_cmap('jet')(np.linspace(0, 1, len(graphs)))
        colors = (colors[:, 0:3] * 255).astype(np.uint8)
        colors = [tuple(color.tolist()) for color in colors]

        for i, graph in enumerate(graphs):

            if len(graph.edges) == 0:
                continue

            for edge in graph.edges:
                # edge as arrow
                start = graph.nodes[edge[0]]["pos"]
                end = graph.nodes[edge[1]]["pos"]
                start = (int(start[0]), int(start[1]))
                end = (int(end[0]), int(end[1]))
                cv2.arrowedLine(G_agg_viz, start, end, color=colors[i], thickness=1, line_type=cv2.LINE_AA)

            pos = (int(self.pose_history[i, 0]), int(self.pose_history[i, 1]) - 10)
            cv2.putText(G_agg_viz, "{} - {:.0f}".format(i, graph.graph["succ_graph_weight"]), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite("/data/autograph/evaluations/G_agg/{}/{}/{:04d}_{}_viz.png".format(self.data_source, self.tile_id, self.step, name), G_agg_viz)


    def visualize_write_G_agg(self, G_agg, name="G_agg"):

        G_agg_viz = self.aerial_image.copy()

        # make darker
        G_agg_viz = G_agg_viz // 4

        # if len(G_agg.edges) == 0:
        #     return

        # history colors linearly interpolated
        colors = matplotlib.cm.get_cmap('jet')(np.linspace(0, 1, len(list(G_agg.edges))))
        colors = (colors[:, 0:3] * 255).astype(np.uint8)
        colors = [tuple(color.tolist()) for color in colors]

        for i, edge in enumerate(G_agg.edges):
            # edge as arrow
            start = G_agg.nodes[edge[0]]["pos"]
            end = G_agg.nodes[edge[1]]["pos"]
            start = (int(start[0]), int(start[1]))
            end = (int(end[0]), int(end[1]))

            cv2.arrowedLine(G_agg_viz, start, end, color=colors[i], thickness=4, line_type=cv2.LINE_AA)

        if len(self.graphs) > 1:
            curr_graph = self.graphs[-1].copy()
            for edge in curr_graph.edges:
                start = curr_graph.nodes[edge[0]]["pos"]
                end = curr_graph.nodes[edge[1]]["pos"]
                start = (int(start[0]), int(start[1]))
                end = (int(end[0]), int(end[1]))
                cv2.arrowedLine(G_agg_viz, start, end, color=(255, 255, 255), thickness=4, line_type=cv2.LINE_AA)


        for p in self.pose_history:
            x_0, y_0, _ = p
            x_0 = int(x_0)
            y_0 = int(y_0)
            cv2.circle(G_agg_viz, (x_0, y_0), 2, (0, 255, 0), -1)

        # # also visualize queued poses
        # arrow_length = 30
        # for p in self.future_poses:
        #     x_0, y_0, yaw = p
        #     x_0 = int(x_0)
        #     y_0 = int(y_0)
        #     start = (x_0, y_0)
        #     end = (x_0 + arrow_length * np.sin(yaw),
        #            y_0 - arrow_length * np.cos(yaw))
        #     start = (int(start[0]), int(start[1]))
        #     end = (int(end[0]), int(end[1]))
        #
        #     cv2.arrowedLine(G_agg_viz, start, end, color=(0, 0, 255), thickness=3, line_type=cv2.LINE_AA)

        cv2.imwrite("/data/autograph/evaluations/G_agg/{}/{}/{:04d}_{}_viz.png".format(self.data_source,
                                                                                       self.tile_id,
                                                                                       self.step, name), G_agg_viz)

        margin = 400
        G_agg_viz = G_agg_viz[int(self.pose[1]) - margin:int(self.pose[1]) + margin,
                    int(self.pose[0]) - margin:int(self.pose[0]) + margin]

        cv2.imshow("G_agg_viz", G_agg_viz)

        # serialize graph
        # pickle.dump(G_agg, open("/data/autograph/evaluations/G_agg/{}/{:04d}_{}.pickle".format(self.tile_id, self.step, name), "wb"))


    def drive_keyboard(self, key):

        print("Pose x, y, yaw: {:.1f}, {:.1f}, {:.2f}".format(self.pose[0], self.pose[1], self.pose[2]))

        if self.pose[2] > 2 * np.pi:
            self.pose[2] -= 2 * np.pi
        if self.pose[2] < -2 * np.pi:
            self.pose[2] += 2 * np.pi

        # alter pose based on which arrow key is pressed
        s = 50

        forward_vector = np.array([np.cos(self.pose[2]),
                                   -np.sin(self.pose[2])])
        sideways_vector = np.array([np.cos(self.pose[2] + np.pi / 2),
                                    -np.sin(self.pose[2] + np.pi / 2)])

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
        elif key == Key.page_up:
            delta = s/2. * sideways_vector
            self.pose[0:2] += np.array([delta[1], delta[0]])
        elif key == Key.page_down:
            delta = s/2. * sideways_vector
            self.pose[0:2] -= np.array([delta[1], delta[0]])

        self.make_step()
        cv2.waitKey(0)



    def crop_coordintates_to_global(self, pose, pos_local):
        """
        :param pose:
        :param pos_local: local position in the image frame (origin is top left), shape (2,)
        :return:
        """

        squeeze = False
        if len(pos_local.shape) == 1:
            squeeze = True
            pos_local = np.expand_dims(pos_local, axis=0)

        pos_local = np.array([[256, 128]]) - pos_local

        pos_global = np.zeros_like(pos_local)
        pos_global[:, 0] = pose[0] - pos_local[:, 1] * np.cos(pose[2]) + pos_local[:, 0] * np.sin(pose[2])
        pos_global[:, 1] = pose[1] - pos_local[:, 1] * np.sin(pose[2]) - pos_local[:, 0] * np.cos(pose[2])

        if squeeze:
            pos_global = np.squeeze(pos_global, axis=0)

        return pos_global


    def drive_freely(self):

        # if self.step > 50:
        #     self.done = True
        #     return

        fps = 1 / (time.time() - self.time)
        self.time = time.time()

        print("Step: {} | FPS = {:.1f} | Pose: {:.0f}, {:.0f}, {:.1f} | {}".format(self.step, fps, self.pose[0],
                                                                                   self.pose[1], self.pose[2],
                                                                                   self.tile_id))

        self.visualize_write_G_agg(self.G_agg_naive, "G_agg_naive")

        if self.graph_skeleton is None:
            self.make_step()
            return

        G_current_local = self.graph_skeleton.copy()

        # succ_graph_weight = np.sum(self.skeleton * self.pred_succ)
        succ_graph_weight = np.sum(self.skeleton * self.pred_drivable)

        # do branch_alive check
        branch_alive = True
        if succ_graph_weight < succ_graph_weight_threshold:
            print("     Successor Graph too weak ({:.1f}), aborting branch".format(succ_graph_weight))
            branch_alive = False

        if branch_alive:
            G_current_global = nx.DiGraph()

            # add nodes and edges from self.graph_skeleton and transform to global coordinates (for aggregation)
            for node in G_current_local.nodes:
                # transform pos_start to global coordinates
                pos_local = nx.get_node_attributes(G_current_local, "pts")[node][0].astype(np.float32)
                pos_global = self.crop_coordintates_to_global(self.pose, pos_local)

                G_current_global.add_node(node,
                                          pos=pos_global,
                                          weight=1.0,
                                          score=1.0,)

            for edge in G_current_local.edges:
                edge_points = G_current_local.edges[edge]["pts"]
                edge_points = self.crop_coordintates_to_global(self.pose, edge_points)
                G_current_global.add_edge(edge[0], edge[1], pts=edge_points)

            # convert to smooth graph
            G_current_global_dense = roundify_skeleton_graph(G_current_global)
            G_current_global_dense.graph["succ_graph_weight"] = succ_graph_weight
            self.graphs.append(G_current_global_dense)

            self.add_graph_to_angle_canvas()

            successor_points = []
            for node in G_current_global.nodes:
                if len(list(G_current_global.successors(node))) >= 1:
                    successor_points.append(node)

            succ_edges = []
            for successor_point in successor_points:
                succ = list(G_current_global.successors(successor_point))
                for successor in succ:
                    succ_edges.append(G_current_global.edges[successor_point, successor])

            print("     Number of successor edges: {}".format(len(succ_edges)))

            self.future_poses_destinations = []

            node_current_pose = (int(self.pose[0]), int(self.pose[1]))

            # loop over all successor edges to find future poses
            for edge in succ_edges:

                num_points_in_edge = len(edge["pts"])
                if num_points_in_edge < edge_end_idx+1:
                    continue

                succ_edge_start = np.array([edge["pts"][edge_start_idx][0],
                                            edge["pts"][edge_start_idx][1]])
                succ_edge_end = np.array([edge["pts"][edge_end_idx][0],
                                          edge["pts"][edge_end_idx][1]])

                node_succ_edge_start = (int(succ_edge_start[0]), int(succ_edge_start[1]))
                node_succ_edge_end = (int(succ_edge_end[0]), int(succ_edge_end[1]))

                edge_delta = succ_edge_end - succ_edge_start
                angle_global = np.arctan2(edge_delta[0], -edge_delta[1])

                # define future pose
                future_pose_global = np.zeros(3)
                diff = step_size * (succ_edge_end - succ_edge_start) / np.linalg.norm(succ_edge_end - succ_edge_start)
                future_pose_global[0:2] = succ_edge_start + diff
                future_pose_global[2] = self.yaw_check(angle_global)

                # put future pose in queue if not yet visited
                is_already_in_queue, matches_future = similarity_check(future_pose_global,
                                                                       self.future_poses,
                                                                       min_dist=30,
                                                                       min_angle=np.pi/4)

                if is_already_in_queue:
                    print("     future pose already in queue. Connecting to current pose.")
                    node_current_pose = (int(self.pose[0]), int(self.pose[1]))

                    for match_future in matches_future:
                        try:
                            future_pose = self.future_poses[match_future][np.newaxis, 0:2]
                        except:
                            continue

                        future_pose_node = (int(future_pose[0, 0]), int(future_pose[0, 1]))

                        # now add edge from ego pose to ego edge start
                        # and edge from ego edge start to future pose
                        self.G_agg_naive.add_node(node_current_pose, pos=self.pose[0:2])
                        self.G_agg_naive.add_node(node_succ_edge_start, pos=succ_edge_start)
                        self.G_agg_naive.add_node(future_pose_node, pos=future_pose[0, 0:2])

                        self.G_agg_naive.add_edge(node_current_pose, node_succ_edge_start, step=self.step)
                        self.G_agg_naive.add_edge(node_succ_edge_start, future_pose_node, step=self.step)
                        print("     ! Added edge to already queued pose: {} -> {}".format(node_current_pose,
                                                                                          node_succ_edge_start))
                    continue


                was_visited, matches_visited = similarity_check(future_pose_global,
                                                                self.pose_history,
                                                                min_dist=30,
                                                                min_angle=np.pi/4)

                if was_visited:
                    print("     future pose was already visited. Connecting to current pose.")
                    for match_visited in matches_visited:
                        try:
                            future_pose = self.pose_history[match_visited][np.newaxis, 0:2]
                        except:
                            continue

                        future_pose_node = (int(future_pose[0, 0]), int(future_pose[0, 1]))

                        # now add edge from ego pose to ego edge start
                        # and edge from ego edge start to future pose
                        self.G_agg_naive.add_node(node_current_pose, pos=self.pose[0:2])
                        self.G_agg_naive.add_node(node_succ_edge_start, pos=succ_edge_start)
                        self.G_agg_naive.add_node(future_pose_node, pos=future_pose[0, 0:2])

                        self.G_agg_naive.add_edge(node_current_pose, node_succ_edge_start, step=self.step)
                        self.G_agg_naive.add_edge(node_succ_edge_start, future_pose_node, step=self.step)
                        print("     ! Added edge to already visited pose: {} -> {}".format(node_current_pose,
                                                                                           node_succ_edge_start))
                    continue


                # define future pose 2
                future_pose_global2 = np.zeros(3)
                diff2 = step_size / 2 * (succ_edge_end - succ_edge_start) / np.linalg.norm(succ_edge_end - succ_edge_start)
                future_pose_global2[0:2] = succ_edge_start + diff2
                future_pose_global2[2] = self.yaw_check(angle_global)

                was_visited2, matches_visited2 = similarity_check(future_pose_global2,
                                                                  self.pose_history,
                                                                  min_dist=30,
                                                                  min_angle=np.pi/4)


                if was_visited2:
                    print("     future pose was already visited2. Connecting to current pose.")
                    for match_visited2 in matches_visited2:
                        try:
                            future_pose2 = self.pose_history[match_visited2][np.newaxis, 0:2]
                        except Exception as e:
                            print(e)
                            continue
                        future_pose_node2 = (int(future_pose2[0, 0]), int(future_pose2[0, 1]))

                        # now add edge from ego pose to ego edge start
                        # and edge from ego edge start to future pose
                        self.G_agg_naive.add_node(node_current_pose, pos=self.pose[0:2])
                        self.G_agg_naive.add_node(node_succ_edge_start, pos=succ_edge_start)
                        self.G_agg_naive.add_node(future_pose_node2, pos=future_pose2[0, 0:2])

                        self.G_agg_naive.add_edge(node_current_pose, node_succ_edge_start)
                        self.G_agg_naive.add_edge(node_succ_edge_start, future_pose_node2)
                        print("     ! Added edge to already visited pose: {} -> {}".format(node_current_pose,
                                                                                           node_succ_edge_start))
                    continue

                if not was_visited and not is_already_in_queue:

                    self.future_poses.append(future_pose_global)
                    self.future_poses_destinations.append(future_pose_global)

                    # add G_agg-edge from edge start to edge end
                    self.G_agg_naive.add_node(node_succ_edge_start, pos=succ_edge_start)
                    self.G_agg_naive.add_node(node_succ_edge_end, pos=succ_edge_end)

            # add future poses to graph
            for dest in self.future_poses_destinations:
                node_dest = (int(dest[0]), int(dest[1]))
                self.G_agg_naive.add_node(node_current_pose, pos=self.pose[0:2])
                self.G_agg_naive.add_node(node_dest, pos=dest[0:2])
                self.G_agg_naive.add_edge(node_current_pose, node_dest)
                print("     added edge from current pose to future pose: {} -> {}".format(node_current_pose, node_dest))


            G_agg_naive_copy = self.G_agg_naive.copy(as_view=False)
            # also remove all edges longer than 80 px
            for edge in G_agg_naive_copy.edges():
                node_1_pos = np.array(self.G_agg_naive.nodes[edge[0]]["pos"])
                node_2_pos = np.array(self.G_agg_naive.nodes[edge[1]]["pos"])
                if np.linalg.norm(node_1_pos - node_2_pos) > max_edge_length:
                    self.G_agg_naive.remove_edge(edge[0], edge[1])
                    print("     removed edge: {} -> {}".format(edge[0], edge[1]))


            if self.step % write_every == 0:
                self.render_poses_in_aerial()
                self.visualize_write_G_agg(self.G_agg_naive, "G_agg_naive")
                # self.visualize_write_G_single(self.graphs, "G_single")

                # cv2.imwrite("/data/autograph/evaluations/G_agg/{}/{:04d}_angle_canvas.png".format(self.tile_id, self.step), self.canvas_angles)

                # G_agg_cvpr = driver.aggregate_graphs(self.graphs)
                #
                # fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
                # [ax.set_aspect('equal') for ax in axarr]
                # [ax.invert_yaxis() for ax in axarr]
                # axarr[0].set_title("g single")
                # axarr[1].set_title("G_agg_naive")
                # axarr[2].set_title("G_agg_cvpr")
                # [visualize_graph(g, axarr[0], node_color="g", edge_color="g") for g in self.graphs]
                # visualize_graph(self.G_agg_naive, axarr[1], node_color="b", edge_color="b")
                # visualize_graph(G_agg_cvpr, axarr[2], node_color="r", edge_color="r")
                # plt.show()

            print("     Pose queue size: {}".format(len(self.future_poses)))

        if len(self.future_poses) == 0:
            print("future_poses empty. Exiting.")
            self.done = True
            return

        # reorder queue based on distance to current pose
        self.future_poses.sort(key=lambda x: np.linalg.norm(x[0:2] - self.pose[0:2]))

        self.pose = self.future_poses.pop(0)
        while out_of_bounds_check(self.pose, self.aerial_image.shape, oob_margin=500):
            print("     pose out of bounds. removing from queue")
            if len(self.future_poses) == 0:
                print("future_poses empty. Exiting.")
                self.done = True
                break
            self.pose = self.future_poses.pop(0)

        print("     get pose from queue: {:.0f}, {:.0f}, {:.1f}".format(self.pose[0], self.pose[1], self.pose[2]))

        self.pose[2] = self.yaw_check(self.pose[2])

        self.make_step()
        cv2.waitKey(waitkey_ms)

    def cleanup(self):
        cv2.destroyAllWindows()

        # write self.graphs to disk
        os.makedirs(self.dumpdir, exist_ok=True)

        with open("{}/G_agg_naive_raw.pickle".format(self.dumpdir), "wb") as f:
            pickle.dump(self.G_agg_naive, f)

        G_agg = self.G_agg_naive.copy(as_view=False)

        # remove node if it has no successors:
        G_agg_copy = G_agg.copy(as_view=False)
        for node in G_agg_copy.nodes():
            if len(list(G_agg_copy.successors(node))) == 0:
                G_agg.remove_node(node)

        print("Number of nodes: {}, Number of edges: {}".format(len(G_agg.nodes()), len(G_agg.edges())))

        G_agg_copy = G_agg.copy(as_view=False)
        for node in G_agg_copy.nodes():
            if len(list(G_agg_copy.predecessors(node))) == 0:
                G_agg.remove_node(node)

        print("Number of nodes: {}, Number of edges: {}".format(len(G_agg.nodes()), len(G_agg.edges())))

        # connect edges that are close according to their step attribute

        # sample edges more densly:
        G_agg_copy = G_agg.copy(as_view=False)
        for edge in G_agg_copy.edges():
            # place midpoint between two nodes
            node_1_pos = G_agg_copy.nodes[edge[0]]["pos"]
            node_2_pos = G_agg_copy.nodes[edge[1]]["pos"]
            midpoint = (node_1_pos + node_2_pos) / 2
            midpoint_node = (int(midpoint[0]), int(midpoint[1]))
            G_agg.add_node(midpoint_node, pos=midpoint)
            G_agg.remove_edge(edge[0], edge[1])
            # add edge between midpoint and both nodes
            G_agg.add_edge(edge[0], midpoint_node)
            G_agg.add_edge(midpoint_node, edge[1])

        # visualize
        self.visualize_write_G_agg(self.G_agg_naive, "G_agg_naive_final_raw")
        self.visualize_write_G_agg(G_agg, "G_agg_naive_final_cleanup")

        with open("{}/G_agg_naive_raw.pickle".format(self.dumpdir), "wb") as f:
            pickle.dump(self.G_agg_naive, f)

        with open("{}/G_agg_naive_cleanup.pickle".format(self.dumpdir), "wb") as f:
            pickle.dump(G_agg, f)

        with open("{}/graphs_all.pickle".format(self.dumpdir), "wb") as f:
            pickle.dump(self.graphs, f)


if __name__ == "__main__":

    tile_ids = glob.glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/tiles/*/*.png")
    tile_ids = [os.path.basename(t).split(".")[0] for t in tile_ids]

    if len(sys.argv) > 1:
        tile_ids = [sys.argv[1]]

    if len(sys.argv) > 2:
        data_source = sys.argv[2]
    else:
        data_source = "tracklets"

    # tile_ids = ['freiburg_8057_7995']  # freiburg

    for tile_id in tile_ids:

        print("Driving on tile {}".format(tile_id))

        driver = AerialDriver(debug=True,
                              input_layers="rgb",
                              tile_id=tile_id,
                              data_source=data_source)

        # Tracklets_joint
        if data_source == "tracklets":
            print("Using tracklets model")
            driver.load_model(model_path="/data/autograph/checkpoints/serene-voice-204/e-016.pth",  # (all-3004)
                              type="full")
            driver.load_model(model_path="/data/autograph/checkpoints/visionary-voice-212/e-030.pth",  # (all-3004)
                              type="successor",
                              input_layers="rgb")
        # Lanegraph
        elif data_source == "lanegraph":
            print("Using lanegraph model")
            driver.load_model(model_path="/data/autograph/checkpoints/dulcet-water-210/e-058.pth",  # (all-3004)
                              type="full")
            driver.load_model(model_path="/data/autograph/checkpoints/fallen-oath-217/e-050.pth",  # (all-3004)
                              type="successor",
                              input_layers="rgb")
        else:
            raise ValueError("Unknown data_source {}".format(data_source))


        if "freiburg" in tile_id:
            driver.load_satellite(impath="/data/lanegraph/freiburg/{}.png".format(tile_id))
        else:
            driver.load_satellite(
                impath=glob.glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/tiles/*/{}.png".format(tile_id))[0])

        while True:
            driver.drive_freely()
            if driver.done:
                driver.cleanup()
                break


        # # load files from disk
        # with open("/data/autograph/evaluations/G_agg/{}/graphs_all.pickle".format(driver.tile_id, driver.tile_id), "rb") as f:
        #     graphs = pickle.load(f)
        # with open("/data/autograph/evaluations/G_agg/{}/G_agg_naive_all.pickle".format(driver.tile_id, driver.tile_id), "rb") as f:
        #     G_agg_naive = pickle.load(f)

        # G_agg_cvpr = driver.aggregate_graphs(graphs)
        # driver.visualize_write_G_agg(G_agg_cvpr, "G_agg_cvpr")
        # driver.visualize_write_G_agg(G_agg_naive, "G_agg_naive")

        # fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        # img = cv2.cvtColor(driver.aerial_image, cv2.COLOR_BGR2RGB)
        # [ax.imshow(img) for ax in axarr]
        # axarr[0].set_title("g single")
        # axarr[1].set_title("G_agg_naive")
        # axarr[2].set_title("G_agg_cvpr")
        # [visualize_graph(g, axarr[0], node_color=np.random.rand(3), edge_color=np.random.rand(3)) for g in graphs]
        # visualize_graph(driver.G_agg_naive, axarr[1], node_color="b", edge_color="b")
        # visualize_graph(G_agg_cvpr, axarr[2], node_color="r", edge_color="r")
        # plt.show()

        continue

        print("Press arrow keys to drive")

        def on_press(key):
            driver.drive_keyboard(key)

        def on_release(key):
            if key == Key.esc:
                return False

        # Collect events until released
        with Listener(on_press=on_press) as listener:
            listener.join()

