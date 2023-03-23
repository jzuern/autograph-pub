import cv2
import torch
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
from collections import OrderedDict
import torchvision.models as models
import numpy as np
from pynput.keyboard import Key, Listener, Controller
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2334477275000
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from aggregation.utils import AngleColorizer
import sknw
import networkx as nx
import time
from aggregation.utils import smooth_trajectory, similarity_check, out_of_bounds_check

keyboard = Controller()


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
        self.init_pose = np.array([1600, 2641, 2*np.pi*0.98])
        self.pose = self.init_pose.copy()
        self.current_crop = None
        self.model = None

        self.crop_shape = (256, 256)

        self.canvas_log_odds = None
        self.canvas_angles = None

        self.pose_history = np.array([self.pose])
        self.ac = AngleColorizer()

        self.future_poses = []
        self.step = 0
        self.graph_skeleton = None


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

        print("Model loaded")

    def load_satellite(self, path):
        print("Loading satellite...")
        # self.satellite = np.asarray(Image.open(path)).astype(np.uint8)
        # self.satellite = cv2.cvtColor(self.satellite, cv2.COLOR_BGR2RGB)

        # # Crop
        # top_left = [36000, 26000]  # vertical, horizontal
        # delta = [3000, 3000]
        # self.satellite = self.satellite[top_left[0]:top_left[0] + delta[0], top_left[1]:top_left[1] + delta[1], :]
        #
        #
        # cv2.imwrite("satellite.png", cv2.cvtColor(self.satellite, cv2.COLOR_BGR2RGB))
        # exit()
        self.satellite = np.asarray(Image.open("satellite.png")).astype(np.uint8)


        self.canvas_log_odds = np.ones([self.satellite.shape[0], self.satellite.shape[1]], dtype=np.float32)
        self.canvas_angles = np.zeros([self.satellite.shape[0], self.satellite.shape[1], 3], dtype=np.uint8)

        print("Satellite loaded")

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


    def skeletonize_prediction(self, pred, threshold=0.5):

        # first, convert to binary
        pred_thrshld = (pred > threshold).astype(np.uint8)

        # then, skeletonize
        skeleton = skeletonize(pred_thrshld)

        # cut away top and sides by N pixels
        N = 30
        skeleton[:N,  :] = 0
        skeleton[: , :N] = 0
        skeleton[:, -N:] = 0


        return skeleton


    def pose_to_transform_1(self):

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

        M = np.linalg.inv(self.pose_to_transform_1())

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
        df = self.canvas_log_odds.shape[0] / 1500
        img1 = cv2.resize(colorize(self.canvas_log_odds), (1500, 1500))
        img2 = cv2.resize(self.satellite, (1500, 1500))
        canvas_viz = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        for i in range(1, len(self.pose_history)-1):
            x_0, y_0, _ = self.pose_history[i-1]
            x_1, y_1, _ = self.pose_history[i]

            x_0 -= self.crop_shape[0]
            x_1 -= self.crop_shape[0]
            y_0 -= self.crop_shape[0]
            y_1 -= self.crop_shape[0]

            x_0 = int(x_0 / df)
            x_1 = int(x_1 / df)
            y_0 = int(y_0 / df)
            y_1 = int(y_1 / df)

            cv2.line(canvas_viz, (x_0, y_0), (x_1, y_1), (0, 145, 0), 1, cv2.LINE_AA)

        # cv2.imshow("Aggregation", canvas_viz)



        # also visualize the canvas around current pose
        width = 800
        x_1, y_1, _ = self.pose_history[-1]
        x_1 = int(x_1)
        y_1 = int(y_1)

        canvas_roi = self.canvas_log_odds[y_1 - width//2:y_1+width//2, x_1-width//2 : x_1+width//2]
        canvas_roi = colorize(canvas_roi)
        satellite_roi = self.satellite[y_1 - width//2 : y_1+width//2, x_1-width//2 : x_1+width//2, :]
        viz_roi = cv2.addWeighted(canvas_roi, 0.5, satellite_roi, 0.5, 0)

        cv2.imshow("Aggregation_cropped", viz_roi)
        cv2.imwrite("/home/zuern/Desktop/tmp/{:04d}_Aggregation_cropped.png".format(self.step), viz_roi)


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

        M = self.pose_to_transform_1()
        satellite_image = self.satellite.copy()

        try:
            rgb = cv2.warpPerspective(satellite_image, M, (self.crop_shape[0], self.crop_shape[1]),
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        except:
            print("Error in warpPerspective. Resetting position")
            self.pose = self.init_pose
            rgb = self.crop_satellite_at_pose(self.pose)

        self.current_crop = rgb

        return rgb


    def skeleton_to_graph(self, skeleton, pred_angles, pred_succ):
        """Convert skeleton to graph"""

        # build graph from skeleton
        graph = sknw.build_sknw(skeleton)

        # smooth edges
        for (s, e) in graph.edges():
            graph[s][e]['pts'] = smooth_trajectory(graph[s][e]['pts'])

        # add node positions
        node_positions = np.array([graph.nodes[n]['o'] for n in graph.nodes()])

        if len(node_positions) == 0:
            return graph

        start_node = np.argmin(np.linalg.norm(node_positions - np.array([128, 255]), axis=1))

        # remove all edges that are not connected to closest node
        connected = nx.node_connected_component(graph, start_node)  # nodes of component that contains node 0
        graph.remove_nodes_from([n for n in graph if n not in connected])

        graph = graph.to_directed()

        # now let every edge face away from the closest node
        edge_order = nx.dfs_edges(graph, source=start_node, depth_limit=None)
        edge_order = list(edge_order)

        # print("edge_order", edge_order)
        # print("graph.edges()", graph.edges())

        for i, (s, e) in enumerate(edge_order):
            if graph.has_edge(s, e):
                if graph.has_edge(e, s):
                    graph[s][e]['pts'] = np.flip(graph[e][s]['pts'], axis=0)
                    graph.remove_edge(e, s)

        # print("graph.edges()", graph.edges())


        # fig, ax = plt.subplots()
        # ax.set_aspect('equal')
        # ax.imshow(self.current_crop)
        # for (s, e) in graph.edges():
        #     ps = graph[s][e]['pts']
        #     for i in range(len(ps) - 1):
        #         ax.arrow(ps[i][1], ps[i][0], ps[i + 1][1] - ps[i][1], ps[i + 1][0] - ps[i][0], head_width=1, head_length=1, fc='g', ec='g')
        #     ax.arrow(ps[0][1], ps[0][0], ps[-1][1] - ps[0][1], ps[-1][0] - ps[0][0], head_width=5, head_length=5, fc='k', ec='k')
        # plt.scatter(node_positions[:, 1], node_positions[:, 0], c='r')
        # plt.show()


        graphviz = np.repeat(pred_succ[:, :, np.newaxis], 3, axis=2)

        # draw edges by pts
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            for i in range(len(ps) - 1):
                cv2.arrowedLine(graphviz, (int(ps[i][1]), int(ps[i][0])), (int(ps[i + 1][1]), int(ps[i + 1][0])), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.arrowedLine(graphviz, (int(ps[0][1]), int(ps[0][0])), (int(ps[-1][1]), int(ps[-1][0])), (255, 0, 255), 1, cv2.LINE_AA)

        # draw nodes
        nodes = graph.nodes()
        node_positions = np.array([nodes[i]['o'] for i in nodes])
        [cv2.circle(graphviz, (int(p[1]), int(p[0])), 4, (0, 255, 0), -1) for p in node_positions]

        # cv2.imshow("graphviz", graphviz)
        # cv2.imwrite("/home/zuern/Desktop/tmp/{:04d}_graphviz.png".format(self.step), graphviz)


        return graph



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

        # plt.imshow(angle_canvas_cropped)
        # plt.show()

        # angle_canvas_cropped = angle_canvas_cropped % (2 * np.pi)

        M = np.linalg.inv(self.pose_to_transform_1())

        angle_canvas_cropped_c = self.ac.angle_to_color(angle_canvas_cropped)
        angle_canvas_cropped_c = angle_canvas_cropped_c * np.expand_dims(angle_indicator, axis=2)

        # cv2.imshow("angles_colorized", angle_canvas_cropped_c)

        warped_angles = cv2.warpPerspective(angle_canvas_cropped_c, M,
                                           (self.canvas_angles.shape[0], self.canvas_angles.shape[1]),
                                           cv2.INTER_LINEAR)

        info_available = np.sum(warped_angles, axis=2) > 0

        self.canvas_angles[info_available] = warped_angles[info_available]

        # resize to smaller
        img1 = cv2.resize(self.canvas_angles, (1000, 1000))
        img2 = cv2.resize(self.satellite, (1000, 1000))
        canvas_viz = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        cv2.imshow("Aggregation angles", canvas_viz)


    def render_poses_in_aerial(self):
        rgb_pose_viz = self.satellite.copy()
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

        # fig, ax = plt.subplots()
        # plt.imshow(rgb_pose_viz)
        # ax.set_xlim([x1, x2])
        # ax.set_ylim([y2, y1])
        # plt.show()

        rgb_pose_viz = rgb_pose_viz[y1:y2, x1:x2]


        # cv2.imshow("rgb_pose_viz", rgb_pose_viz)
        cv2.imwrite("/home/zuern/Desktop/tmp/{:04d}_rgb_pose_viz.png".format(self.step), rgb_pose_viz)


    def make_step(self):

        """Run one step of the driving loop."""

        self.pose_history = np.concatenate([self.pose_history, [self.pose]])

        pos_encoding = self.generate_pos_encoding()
        rgb = self.crop_satellite_at_pose(self.pose)

        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255
        pos_encoding_torch = torch.from_numpy(pos_encoding).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255

        if self.model_full is not None:
            with torch.no_grad():
                (pred, _) = self.model_full(torch.cat([rgb_torch, rgb_torch], dim=0))
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

        pred_angles = self.ac.xy_to_angle(pred_angles[0].cpu().detach().numpy())
        pred_angles_color = self.ac.angle_to_color(pred_angles)

        skeleton = self.skeletonize_prediction(pred_succ, threshold=0.1)
        self.graph_skeleton = self.skeleton_to_graph(skeleton, pred_angles, pred_succ)

        # make skeleton fatter
        skeleton = skeleton.astype(np.uint8) * 255
        skeleton = cv2.dilate(skeleton, np.ones((3, 3), np.uint8), iterations=1)
        skeleton = (skeleton / 255.0).astype(np.float32)


        # self.add_pred_to_canvas(sdf_succ, self.pose)
        self.add_pred_to_canvas(skeleton)
        self.add_graph_to_angle_canvas()

        # skeleton = (skeleton * 255).astype(np.uint8)
        # skeleton_viz = cv2.addWeighted(rgb, 0.5, cv2.applyColorMap(skeleton, cv2.COLORMAP_MAGMA), 0.5, 0)
        # cv2.imshow("skeleton", skeleton_viz)

        pred_succ = (pred_succ * 255).astype(np.uint8)
        pred_drivable = (pred_drivable * 255).astype(np.uint8)
        pred_succ_viz = cv2.addWeighted(rgb, 0.5, cv2.applyColorMap(pred_succ, cv2.COLORMAP_MAGMA), 0.5, 0)
        # pred_drivable_viz = cv2.addWeighted(rgb, 0.5, cv2.applyColorMap(pred_drivable, cv2.COLORMAP_MAGMA), 0.5, 0)
        # pred_angles_color_viz = cv2.addWeighted(rgb, 0.5, pred_angles_color, 0.5, 0)

        # cv2.imshow("pred_succ", pred_succ_viz)
        cv2.imwrite("/home/zuern/Desktop/tmp/{:04d}_pred_succ_viz.png".format(self.step), pred_succ_viz)


        # cv2.imshow("pred_drivable", pred_drivable_viz)
        # cv2.imshow("pred_angles_color", pred_angles_color_viz)

        self.step += 1

    def yaw_check(self, yaw):
        if yaw > 2 * np.pi:
            yaw -= 2 * np.pi
        if yaw < 0:
            yaw += 2 * np.pi
        return yaw



    def drive_keyboard(self, key):

        print("drive_step")

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

    def drive_freely(self):

        print("Current pose: {:.0f}, {:.0f}, {:.1f}".format(self.pose[0], self.pose[1], self.pose[2]))

        if self.graph_skeleton is None:
            self.make_step()
            return

        g = self.graph_skeleton
        successor_points = []
        for node in g.nodes:
            if len(list(g.successors(node))) >= 1:
                successor_points.append(node)

        succ_edges = []
        for successor_point in successor_points:
            succ = list(g.successors(successor_point))
            for successor in succ:
                succ_edges.append(g.edges[successor_point, successor])

        print("checking {} edges".format(len(succ_edges)))
        for edge in succ_edges:
            start_idx = 10
            end_idx = 40

            num_points_in_edge = len(edge["pts"])
            if num_points_in_edge < end_idx+1:
                continue

            pos_start_local = np.array([edge["pts"][start_idx][1], edge["pts"][start_idx][0]])
            pos_end_local = np.array([edge["pts"][end_idx][1], edge["pts"][end_idx][0]])
            pos_start_local = np.array([128, 256]) - pos_start_local
            pos_end_local = np.array([128, 256]) - pos_end_local

            edge_local = pos_end_local - pos_start_local
            angle_local = np.arctan2(-edge_local[0], edge_local[1])

            print("pos_start", pos_start_local)
            print("pos_end", pos_end_local)
            print("edge_local", edge_local)
            print("edge_angle_local", angle_local)

            # transform pos_start to global coordinates
            pos_start_global = np.zeros(2)
            pos_start_global[0] = self.pose[0] - pos_start_local[0] * np.cos(self.pose[2]) + pos_start_local[1] * np.sin(self.pose[2])
            pos_start_global[1] = self.pose[1] - pos_start_local[0] * np.sin(self.pose[2]) - pos_start_local[1] * np.cos(self.pose[2])

            # add to current pose
            future_pose_global = np.zeros(3)
            future_pose_global[0:2] = pos_start_global
            future_pose_global[2] = self.pose[2] + angle_local
            future_pose_global[2] = self.yaw_check(future_pose_global[2])

            # put in queue if not yet visited
            was_visited = similarity_check(future_pose_global, self.pose_history)
            is_already_in_queue = similarity_check(future_pose_global, self.future_poses)

            if not was_visited and not is_already_in_queue:
                self.future_poses.append(future_pose_global)
                print("put pose in queue: {:.0f}, {:.0f}, {:.1f}".format(future_pose_global[0],
                                                                         future_pose_global[1],
                                                                         future_pose_global[2]))
        self.render_poses_in_aerial()

        if len(self.future_poses) == 0:
            print("future_pose_queuefuture_poses empty. Exiting.")
            exit()
        else:

            # reorder queue based on distance to current pose
            self.future_poses.sort(key=lambda x: np.linalg.norm(x[0:2] - self.pose[0:2]))

            self.pose = self.future_poses.pop(0)
            while out_of_bounds_check(self.pose, self.satellite.shape):
                print("pose out of bounds. removing from queue")
                self.pose = self.future_poses.pop(0)

            print("get pose from queue: {:.0f}, {:.0f}, {:.1f}".format(self.pose[0], self.pose[1], self.pose[2]))

        self.pose[2] = self.yaw_check(self.pose[2])

        self.make_step()

        cv2.waitKey(1)
        time.sleep(0.1)



if __name__ == "__main__":
    driver = SatelliteDriver()

    driver.load_model(model_path="/data/autograph/checkpoints/soft-river-75/e-028.pth",
                      type="full")
    driver.load_model(model_path="/data/autograph/checkpoints/smart-river-76/e-030.pth",
                      type="successor")
    driver.load_satellite(path="/data/lanegraph/woven-data/Austin.png")

    while True:
        driver.drive_freely()


    # print("Press arrow keys to drive")
    #
    # def on_press(key):
    #     driver.drive_keyboard(key)
    #
    #
    # def on_release(key):
    #     if key == Key.esc:
    #         return False
    #
    # # Collect events until released
    # with Listener(on_press=on_press) as listener:
    #     listener.join()

