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

from aggregation.utils import similarity_check, out_of_bounds_check, visualize_graph, AngleColorizer
from utils import aggregate, colorize, skeleton_to_graph, skeletonize_prediction, roundify_skeleton_graph


keyboard = Controller()


# SETTINGS

skeleton_threshold = 0.05  # threshold for skeletonization
edge_start_idx = 10        # start index for selecting edge as future pose
edge_end_idx = 50          # end index for selecting edge as future pose
write_every = 10            # write to disk every n steps



# CVPR graph aggregation
threshold_px = 30
threshold_rad = 0.2
closest_lat_thresh = 30



class AerialDriver(object):
    def __init__(self, debug=False):
        self.aerial_image = None
        self.init_pose = np.array([1163, 2982, -2.69])
        self.pose = self.init_pose.copy()
        self.current_crop = None
        self.model = None
        self.time = time.time()
        self.debug = debug

        self.crop_shape = (256, 256)
        self.graphs = []  # list of graphs from each step

        self.canvas_log_odds = None
        self.canvas_angles = None

        self.pose_history = np.array([self.pose])
        self.ac = AngleColorizer()

        self.future_poses = []
        self.step = 0
        self.graph_skeleton = None

        self.G_agg_naive = nx.DiGraph()

        self.done = False # flag to indicate end of episode


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
                                            num_classes=3).cuda()
            self.model_full.load_state_dict(new_state_dict)
            self.model_full.eval()

        elif type == "successor":
            self.model_succ = DeepLabv3Plus(models.resnet101(pretrained=True),
                                            num_in_channels=9,
                                            num_classes=1).cuda()
            self.model_succ.load_state_dict(new_state_dict)
            self.model_succ.eval()

        print("Model {} loaded".format(model_path))

    def load_satellite(self, impath):
        # if not os.path.exists("aerial_image.png"):
        print("Loading aerial image {}".format(impath))
        self.aerial_image = np.asarray(Image.open(impath)).astype(np.uint8)
        self.tile_id = impath.split("/")[-1].split(".")[0]
        self.city_name = self.tile_id.split("_")[0]
        print("Tile ID: {}".format(self.tile_id))
        print("City: {}".format(self.city_name))

        #     # Crop (horizontal, vertical)
        #
        #     # PaloAlto - 1
        #     # center = [18082, 29236]
        #     # delta = [4000, 4000]
        #     # Austin - 1
        #     center = [43200, 21872]  # horizontal, vertical
        #     delta = [4000, 4000]
        #     self.aerial_image = self.aerial_image[center[1] - delta[1]//2:center[1] + delta[1]//2,
        #                                           center[0] - delta[0]//2:center[0] + delta[0]//2, :]

        # Image.fromarray(self.aerial_image).save("aerial_image.png")
        #
        # self.aerial_image = np.asarray(Image.open("aerial_image.png")).astype(np.uint8)

        self.aerial_image = cv2.cvtColor(self.aerial_image, cv2.COLOR_BGR2RGB)
        self.canvas_log_odds = np.ones([self.aerial_image.shape[0], self.aerial_image.shape[1]], dtype=np.float32)
        self.canvas_angles = np.zeros([self.aerial_image.shape[0], self.aerial_image.shape[1], 3], dtype=np.uint8)

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

        for p in self.pose_history:
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
                cv2.line(angle_canvas_cropped, (int(x1), int(y1)), (int(x2), int(y2)), angle, thickness=2)
                cv2.line(angle_indicator, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness=2)

        # angle_canvas_cropped = angle_canvas_cropped % (2 * np.pi)

        M = np.linalg.inv(self.pose_to_transform())

        angle_canvas_cropped_c = self.ac.angle_to_color(angle_canvas_cropped)
        angle_canvas_cropped_c = angle_canvas_cropped_c * np.expand_dims(angle_indicator, axis=2)

        # cv2.imshow("angles_colorized", angle_canvas_cropped_c)
        warped_angles = cv2.warpPerspective(angle_canvas_cropped_c, M,
                                           (self.canvas_angles.shape[0], self.canvas_angles.shape[1]),
                                           cv2.INTER_LINEAR)

        info_available = np.sum(warped_angles, axis=2) > 0

        self.canvas_angles[info_available] = warped_angles[info_available]

        canvas_viz = cv2.addWeighted(self.aerial_image, 0.5, self.canvas_angles, 0.5, 0)
        # cv2.imwrite("/home/zuern/Desktop/tmp/other/{:04d}_angle_canvas.png".format(self.step), canvas_viz)

        # resize to smaller
        # canvas_viz = cv2.resize(canvas_viz, (1000, 1000))
        #
        # cv2.imshow("Aggregation angles", canvas_viz)

    def render_poses_in_aerial(self):
        rgb_pose_viz = self.aerial_image.copy()
        arrow_len = 60

        for pose in self.pose_history:
            # render pose as arrow
            y = pose[0]
            x = pose[1]
            theta = pose[2]
            x2 = x - arrow_len * np.cos(theta)
            y2 = y + arrow_len * np.sin(theta)
            cv2.arrowedLine(rgb_pose_viz, (int(y), int(x)), (int(y2), int(x2)), (255, 0, 0), 1, cv2.LINE_AA)

        for pose in self.future_poses:
            # render pose as arrow
            y = pose[0]
            x = pose[1]
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
        cv2.imwrite("/home/zuern/Desktop/tmp/other/{:04d}_rgb_pose_viz.png".format(self.step), rgb_pose_viz)

    def make_step(self):

        """Run one step of the driving loop."""

        self.pose_history = np.concatenate([self.pose_history, [self.pose]])

        pos_encoding = self.generate_pos_encoding()
        rgb = self.crop_satellite_at_pose(self.pose)

        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
        pos_encoding_torch = torch.from_numpy(pos_encoding).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255

        with torch.no_grad():
            (pred, _) = self.model_full(rgb_torch)
            pred = torch.nn.functional.interpolate(pred,
                                                   size=rgb_torch.shape[2:],
                                                   mode='bilinear',
                                                   align_corners=True)
            pred_angles = torch.nn.Tanh()(pred[0:1, 0:2, :, :])
            pred_drivable = torch.nn.Sigmoid()(pred[0:1, 2:3, :, :])


        in_tensor = torch.cat([rgb_torch, pos_encoding_torch, pred_drivable, pred_angles], dim=1)
        in_tensor = torch.cat([in_tensor, in_tensor], dim=0)

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

        # pred_angles_color
        # cv2.imshow("pred_angles_color", pred_angles_color)
        # cv2.imshow("rgb", rgb)

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

        # cv2.imshow("pred_succ_viz", pred_succ_viz)
        # cv2.imshow("pred_drivable", pred_drivable)
        # cv2.imshow("pred_angles_succ_color", pred_angles_succ_color)

        if self.debug:
            fig, axarr = plt.subplots(1, 6, figsize=(20, 20), sharex=True, sharey=True)
            axarr[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            axarr[0].title.set_text('rgb')
            axarr[1].imshow(pred_drivable)
            axarr[1].title.set_text('pred_drivable')
            axarr[2].imshow(pred_succ)
            axarr[2].title.set_text('pred_succ')
            axarr[3].imshow(pred_angles_color)
            axarr[3].title.set_text('pred_angles_color')
            axarr[4].imshow(pred_angles_succ_color)
            axarr[4].title.set_text('pred_angles_succ_color')
            axarr[5].imshow(skeleton)
            axarr[5].title.set_text('skeleton')
            plt.show()

        cv2.imwrite("/home/zuern/Desktop/tmp/other/{:04d}_pred_succ_viz.png".format(self.step), pred_succ_viz)

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
        for pred_agg_idx, G in tqdm(enumerate(graphs), total=len(graphs)):
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


    def visualize_write_G_agg(self, G_agg, name="G_agg"):

        G_agg_viz = self.aerial_image.copy()
        G_agg_viz = G_agg_viz // 2

        if len(G_agg.edges) == 0:
            return

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

            cv2.arrowedLine(G_agg_viz, start, end, color=colors[i], thickness=1, line_type=cv2.LINE_AA)

        for p in self.pose_history:
            x_0, y_0, _ = p
            x_0 = int(x_0)
            y_0 = int(y_0)
            cv2.circle(G_agg_viz, (x_0, y_0), 2, (0, 255, 0), -1)

        # also visualize queued poses
        arrow_length = 30
        for p in self.future_poses:
            x_0, y_0, yaw = p
            x_0 = int(x_0)
            y_0 = int(y_0)
            start = (x_0, y_0)
            end = (x_0 + arrow_length * np.sin(yaw),
                   y_0 - arrow_length * np.cos(yaw))
            start = (int(start[0]), int(start[1]))
            end = (int(end[0]), int(end[1]))

            cv2.arrowedLine(G_agg_viz, start, end, color=(0, 0, 255), thickness=3, line_type=cv2.LINE_AA)

        cv2.imwrite("/home/zuern/Desktop/autograph/tmp/G_agg/{:04d}_{}_viz.png".format(self.step, name), G_agg_viz)

        margin = 400
        G_agg_viz = G_agg_viz[int(self.pose[1]) - margin:int(self.pose[1]) + margin,
                              int(self.pose[0]) - margin:int(self.pose[0]) + margin]

        cv2.imshow("G_agg_viz", G_agg_viz)

        # serialize graph
        pickle.dump(self.G_agg_naive, open("/home/zuern/Desktop/autograph/tmp/G_agg/{:04d}_{}.pickle".format(self.step, name), "wb"))


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

        if self.step > 10:
            self.done = True
            return

        fps = 1 / (time.time() - self.time)
        self.time = time.time()

        print("Step: {} | FPS = {:.1f} | Current pose: {:.0f}, {:.0f}, {:.1f}".format(self.step, fps, self.pose[0], self.pose[1], self.pose[2]))

        if self.graph_skeleton is None:
            self.make_step()
            return

        # OLD DRIVING CODE

        # G_current_local = self.graph_skeleton.copy()
        #
        # G_current_local_pruned = self.graph_skeleton.copy()
        #
        #
        # # shorten all edges to 50 pixels
        # for edge in G_current_local_pruned.edges:
        #     if len(list(G_current_local_pruned.successors(edge[1]))) > 0:
        #         continue
        #     pts = G_current_local_pruned.edges[edge]['pts']
        #     pts = pts[0:10, :]
        #     G_current_local_pruned.edges[edge]['pts'] = pts
        #
        #     # also adjust node position
        #     G_current_local_pruned.nodes[edge[1]]['pos'] = pts[-1, :]
        #
        #
        # G_current_global_pruned = nx.DiGraph()
        #
        # # add nodes and edges from self.graph_skeleton and transform to global coordinates (for aggregation)
        # for node in G_current_local_pruned.nodes:
        #     # transform pos_start to global coordinates
        #     pos_local = nx.get_node_attributes(G_current_local_pruned, "pts")[node][0].astype(np.float32)
        #     pos_global = self.crop_coordintates_to_global(self.pose, pos_local)
        #
        #     G_current_global_pruned.add_node(node,
        #                               pos=pos_global,
        #                               weight=1.0,
        #                               score=1.0,)
        #
        # for edge in G_current_local_pruned.edges:
        #     edge_points = G_current_local_pruned.edges[edge]["pts"]
        #     edge_points = self.crop_coordintates_to_global(self.pose, edge_points)
        #     G_current_global_pruned.add_edge(edge[0], edge[1], pts=edge_points)
        #
        # self.graphs.append(G_current_global_pruned)
        #
        #
        # successor_points = []
        # for node in G_current_local.nodes:
        #     if len(list(G_current_local.successors(node))) >= 1:
        #         successor_points.append(node)
        #
        # succ_edges = []
        # for successor_point in successor_points:
        #     succ = list(G_current_local.successors(successor_point))
        #     for successor in succ:
        #         succ_edges.append(G_current_local.edges[successor_point, successor])
        #
        # if len(succ_edges) == 0:
        #     print("     No successor edges found.")
        #
        # for edge in succ_edges:
        #
        #     num_points_in_edge = len(edge["pts"])
        #     if num_points_in_edge < edge_end_idx+1:
        #         continue
        #
        #     pos_start_local = np.array([edge["pts"][edge_start_idx][1],
        #                                 edge["pts"][edge_start_idx][0]])
        #     pos_end_local = np.array([edge["pts"][edge_end_idx][1],
        #                               edge["pts"][edge_end_idx][0]])
        #
        #     edge_start_global = self.crop_coordintates_to_global(self.pose, pos_start_local)
        #     edge_end_global = self.crop_coordintates_to_global(self.pose, pos_end_local)
        #
        #
        #     edge_local = pos_end_local - pos_start_local
        #     angle_local = np.arctan2(edge_local[1], -edge_local[0])
        #     angle_global = self.pose[2] + angle_local
        #
        #     # step_sizes = [20, 40, 60] # number of pixels to move forward along edge
        #     step_sizes = [40]  # number of pixels to move forward along edge
        #
        #     for step_size in step_sizes:
        #
        #
        #         # define future pose
        #         future_pose_global = np.zeros(3)
        #         diff = step_size * (edge_end_global - edge_start_global) / np.linalg.norm(edge_end_global - edge_start_global)
        #         future_pose_global[0:2] = edge_start_global + diff
        #         future_pose_global[2] = self.yaw_check(angle_global)
        #
        #
        #         # put future pose in queue if not yet visited
        #         was_visited = similarity_check(future_pose_global, self.pose_history, min_dist=20, min_angle=np.pi/4)
        #         is_already_in_queue = similarity_check(future_pose_global, self.future_poses, min_dist=20, min_angle=np.pi/4)
        #
        #         if not was_visited and not is_already_in_queue:
        #             self.future_poses.append(future_pose_global)
        #             print("     put pose in queue: {:.0f}, {:.0f}, {:.1f} (step size: {})".format(future_pose_global[0],
        #                                                                                           future_pose_global[1],
        #                                                                                           future_pose_global[2],
        #                                                                                           step_size))
        #
        #             # add edge to aggregated graph
        #             pointlist_local = np.array(edge["pts"][edge_start_idx:edge_end_idx])
        #
        #             pointlist_global = self.crop_coordintates_to_global(self.pose, pointlist_local)
        #
        #             node_edge_start = (int(edge_start_global[0]), int(edge_start_global[1]))
        #             node_edge_end = (int(edge_end_global[0]), int(edge_end_global[1]))
        #
        #             # add G_agg-edge from edge start to edge end
        #             self.G_agg_naive.add_node(node_edge_start, pos=edge_start_global)
        #             self.G_agg_naive.add_node(node_edge_end, pos=edge_end_global)
        #             self.G_agg_naive.add_edge(node_edge_start, node_edge_end, pts=pointlist_global)
        #
        #             break


        # NEW DRIVING CODE



        G_current_local = self.graph_skeleton.copy()
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
        self.graphs.append(G_current_global_dense)


        successor_points = []
        for node in G_current_global.nodes:
            if len(list(G_current_global.successors(node))) >= 1:
                successor_points.append(node)

        succ_edges = []
        for successor_point in successor_points:
            succ = list(G_current_global.successors(successor_point))
            for successor in succ:
                succ_edges.append(G_current_global.edges[successor_point, successor])

        if len(succ_edges) == 0:
            print("     No successor edges found.")


        for edge in succ_edges:

            num_points_in_edge = len(edge["pts"])
            if num_points_in_edge < edge_end_idx+1:
                continue

            pos_start = np.array([edge["pts"][edge_start_idx][0],
                                  edge["pts"][edge_start_idx][1]])
            pos_end = np.array([edge["pts"][edge_end_idx][0],
                                edge["pts"][edge_end_idx][1]])

            edge_delta = pos_end - pos_start
            angle_global = np.arctan2(edge_delta[0], -edge_delta[1])


            # step_sizes = [20, 40, 60] # number of pixels to move forward along edge
            step_sizes = [40]  # number of pixels to move forward along edge

            for step_size in step_sizes:

                # define future pose
                future_pose_global = np.zeros(3)
                diff = step_size * (pos_end - pos_start) / np.linalg.norm(pos_end - pos_start)
                future_pose_global[0:2] = pos_start + diff
                future_pose_global[2] = self.yaw_check(angle_global)

                # put future pose in queue if not yet visited
                was_visited = similarity_check(future_pose_global,
                                               self.pose_history,
                                               min_dist=20,
                                               min_angle=np.pi/4)
                is_already_in_queue = similarity_check(future_pose_global,
                                                       self.future_poses,
                                                       min_dist=20,
                                                       min_angle=np.pi/4)

                if not was_visited and not is_already_in_queue:

                    self.future_poses.append(future_pose_global)
                    print("     put pose in queue: {:.0f}, {:.0f}, {:.1f} (step size: {})".format(future_pose_global[0],
                                                                                                  future_pose_global[1],
                                                                                                  future_pose_global[2],
                                                                                                  step_size))

                    # add edge to aggregated graph
                    pointlist = np.array(edge["pts"][edge_start_idx:edge_end_idx])

                    node_edge_start = (int(pos_start[0]), int(pos_start[1]))
                    node_edge_end = (int(pos_end[0]), int(pos_end[1]))

                    # add G_agg-edge from edge start to edge end
                    self.G_agg_naive.add_node(node_edge_start, pos=pos_start)
                    self.G_agg_naive.add_node(node_edge_end, pos=pos_end)
                    self.G_agg_naive.add_edge(node_edge_start, node_edge_end, pts=pointlist)

                    break
























        if self.step % write_every == 0:
            self.render_poses_in_aerial()
            self.visualize_write_G_agg(self.G_agg_naive, "G_agg_naive")

            # G_agg_cvpr = driver.aggregate_graphs(self.graphs)

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
        else:

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
        cv2.waitKey(1)

    def cleanup(self):
        cv2.destroyAllWindows()

        # write self.graphs to disk
        with open("/home/zuern/Desktop/autograph/tmp/G_agg/graphs_all.pickle", "wb") as f:
            pickle.dump(self.graphs, f)

        with open("/home/zuern/Desktop/autograph/tmp/G_agg/G_agg_naive_all.pickle", "wb") as f:
            pickle.dump(self.G_agg_naive, f)


if __name__ == "__main__":
    driver = AerialDriver(debug=False)

    driver.load_model(model_path="/data/autograph/checkpoints/clean-hill-97/e-014.pth",  # (Austin only, BIG)
                      type="full")
    driver.load_model(model_path="/data/autograph/checkpoints/smart-rain-99/e-023.pth",  # (Austin only, BIG)
                      type="successor")
    driver.load_satellite(impath="/data/lanegraph/urbanlanegraph-dataset-dev/Austin/tiles/eval/Austin_83_34021_46605.png")

    while True:
        driver.drive_freely()
        if driver.done:
            driver.cleanup()
            break



    # load files from disk
    with open("/home/zuern/Desktop/autograph/tmp/G_agg/graphs_all.pickle", "rb") as f:
        graphs = pickle.load(f)
    with open("/home/zuern/Desktop/autograph/tmp/G_agg/G_agg_naive_all.pickle", "rb") as f:
        G_agg_naive = pickle.load(f)

    G_agg_cvpr = driver.aggregate_graphs(graphs)
    driver.visualize_write_G_agg(G_agg_cvpr, "G_agg_cvpr")
    driver.visualize_write_G_agg(G_agg_naive, "G_agg_naive")


    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    img = cv2.cvtColor(driver.aerial_image, cv2.COLOR_BGR2RGB)
    [ax.imshow(img) for ax in axarr]
    axarr[0].set_title("g single")
    axarr[1].set_title("G_agg_naive")
    axarr[2].set_title("G_agg_cvpr")
    [visualize_graph(g, axarr[0], node_color="w", edge_color="w") for g in driver.graphs]
    visualize_graph(driver.G_agg_naive, axarr[1], node_color="b", edge_color="b")
    visualize_graph(G_agg_cvpr, axarr[2], node_color="r", edge_color="r")
    plt.show()

    exit()

    print("Press arrow keys to drive")

    def on_press(key):
        driver.drive_keyboard(key)


    def on_release(key):
        if key == Key.esc:
            return False

    # Collect events until released
    with Listener(on_press=on_press) as listener:
        listener.join()

