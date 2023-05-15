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
from aggregation.utils import similarity_check, out_of_bounds_check, visualize_graph, AngleColorizer, laplacian_smoothing
from utils import aggregate, colorize, skeleton_to_graph, skeletonize_prediction, roundify_skeleton_graph

keyboard = Controller()


# SETTINGS

skeleton_threshold = 0.08  # threshold for skeletonization
edge_start_idx = 10        # start index for selecting edge as future pose
edge_end_idx = 50          # end index for selecting edge as future pose
write_every = 10            # write to disk every n steps
waitkey_ms = 1


# CVPR graph aggregation
threshold_px = 30
threshold_rad = 0.2
closest_lat_thresh = 30


# init_poses = {'paloalto_43_25359_23592': [[2925, 1557, 0.5317240672588055],
#                                           [3132, 1479, -1.0303768265243125],
#                                           [2853, 1341, 2.2236429578956707],
#                                           [3060, 1269, 3.7670776938290222]],
#               'washington_46_36634_59625': [[2565, 798, 1.6092389168160846],
#                                             [3015, 603, 3.2610215796081317],
#                                             [3036, 990, 0.04164257909858837],
#                                             [3285, 777, -1.5102643070127897],
#                                             [1272, 2466, 1.6672701019774834],
#                                             [1776, 2247, 3.104572537715863],
#                                             [1959, 2466, 4.71238898038469],
#                                             [2157, 2490, 1.6563016204731011],
#                                             [2667, 2478, -1.4940244355251187],
#                                             [2397, 2733, 0.0],
#                                             [2370, 2277, 3.0916342578678506]]}



init_poses = {'austin_40_14021_51605': [[339, 4353, 0.3347368373168147],
                                        [231, 1422, 1.7894652726688385],
                                        [2907, 30, 2.958481836327309],
                                        [4623, 405, -1.5152978215491797],
                                        [4239, 3807, 0.4506613260806336],
                                        [1266, 4116, -0.18534794999569493],
                                        [429, 4053, 0.291456794477867],
                                        [1707, 2637, 0.7853981633974483],
                                        [3234, 1902, -1.4994888620096063],
                                        [4752, 2082, -0.551654982528547]],
              'austin_83_34021_46605': [[33, 1917, 2.0106389096106327],
                                        [4239, 105, 3.638015406994675],
                                        [4968, 1455, -1.2490457723982544],
                                        [30, 3543, 0.244978663126864],
                                        [1794, 3894, 0.372987721800061],
                                        [24, 495, 2.0647376957144776],
                                        [57, 927, 1.9890206563741257],
                                        [1935, 2658, 1.8391625377008034],
                                        [1785, 1356, 3.5380339368082345],
                                        [4818, 3702, -1.1597317794050324],
                                        [3873, 2994, 0.39479111969976155]],
              'detroit_136_10700_30709': [[33, 4005, 1.1441688336680205],
                                          [45, 636, 1.7004988639508087],
                                          [4221, 36, 4.19683997292571],
                                          [4701, 807, -0.3109982806055409],
                                          [4023, 3891, -0.2965458090697006],
                                          [4590, 3804, -0.3612037554936818],
                                          [1746, 3702, -0.5619215622568152],
                                          [315, 3837, 1.2627435457711202]],
              'detroit_165_25700_30709': [[3606, 2313, 4.209545769456829],
                                          [3516, 2418, 1.1659045405098132],
                                          [243, 4140, 1.2793395323170293],
                                          [6, 3387, 2.761086276477428],
                                          [216, 4560, -0.7853981633974483],
                                          [201, 1692, 2.3996453855838755],
                                          [27, 2244, 0.9544993893988253],
                                          [2364, 1725, 4.14386443265365],
                                          [2526, 189, 2.626043646130814],
                                          [3342, 39, 2.5625183842220625],
                                          [4962, 309, 4.124386376837123],
                                          [4251, 513, 2.709184878019255],
                                          [4974, 1782, -0.6078019961139605]],
              'miami_185_41863_18400': [[45, 1887, 1.729451588981298],
                                        [1515, 1272, 3.064820762320015],
                                        [153, 39, 3.173839536025047],
                                        [3465, 69, 3.1145722044025286],
                                        [4740, 990, 4.71238898038469],
                                        [4716, 3822, -1.5152978215491797],
                                        [33, 3324, 1.6352231662204502],
                                        [2793, 3849, -0.09347678115858948],
                                        [1221, 615, 2.2655346029915995],
                                        [4488, 2211, -1.1164942401015803]],
              'miami_194_46863_3400': [[66, 2319, 1.5707963267948966],
                                       [99, 273, 1.5363272257953886],
                                       [3465, 36, 3.1638112189165124],
                                       [4878, 288, 4.691558660348473],
                                       [4416, 3879, 0.0739390376579403],
                                       [2583, 3000, 0.0],
                                       [75, 1656, 1.4056476493802696],
                                       [2532, 1035, 1.5971060440478189]],
              'paloalto_43_25359_23592': [[780, 3663, 0.6202494859828214],
                                          [1875, 4116, -0.9481255380378295],
                                          [210, 1407, 2.4061528859142878],
                                          [2841, 1302, -0.9827937232473292],
                                          [3498, 1941, 0.6435011087932844],
                                          [4362, 4332, 0.7188299996216244],
                                          [3357, 4056, -1.0074800653029286],
                                          [2163, 222, 3.6299266046461987]],
              'paloalto_62_35359_38592': [[327, 1740, 0.39852244566642026],
                                          [228, 1431, 2.525295716193722],
                                          [408, 4206, 0.5763752205911836],
                                          [4533, 4068, 0.6799088548723576],
                                          [3693, 4257, -1.0617254383725134],
                                          [4875, 1587, -0.9357695914045827],
                                          [3180, 33, 3.719494590552039],
                                          [4398, 402, 3.7618421395726145],
                                          [51, 273, 2.0647376957144776],
                                          [2067, 291, 3.593746515875569]],
              'pittsburgh_36_27706_11407': [[9, 3219, 1.396124127786657],
                                            [24, 2358, 1.4141944498128811],
                                            [21, 1455, 1.2490457723982544],
                                            [516, 39, 2.356194490192345],
                                            [2148, 33, 2.245537269018449],
                                            [3225, 255, 2.5213431676069717],
                                            [4983, 249, 4.514993420534809],
                                            [4989, 909, -0.7853981633974483],
                                            [4974, 2103, 4.508371000792142],
                                            [4239, 1674, 4.4674103172578254]],
              'pittsburgh_5_2706_31407': [[3561, 2730, 0.44441920990109884],
                                          [18, 537, 2.0344439357957027],
                                          [1413, 180, 0.48995732625372823],
                                          [4692, 318, 3.653982113900531],
                                          [4614, 1470, -1.0014831356942349],
                                          [3054, 768, 2.0899424410414196],
                                          [2910, 1530, 0.4266274931268761],
                                          [4845, 2991, -0.41822432957922917],
                                          [4542, 2979, 2.173083672929861]],
              'washington_46_36634_59625': [[447, 2487, 1.6107750139181867],
                                            [33, 3042, 3.0940095503128098],
                                            [2067, 3426, 4.71238898038469],
                                            [1800, 534, 3.141592653589793],
                                            [3018, 528, 3.141592653589793],
                                            [3057, 1053, 0.03123983343026815],
                                            [4923, 690, 3.9540112661745064],
                                            [4785, 2526, -1.4711276743037347],
                                            [4302, 3723, 0.043450895391530686],
                                            [2970, 4398, 0.0688564893010446]],
              'washington_55_41634_69625': [[57, 255, 1.6052654277944045],
                                            [90, 1098, 1.4659193880646626],
                                            [216, 4113, 1.6078164426688266],
                                            [1758, 3996, 0.04542327942157698],
                                            [3627, 3951, 0.07878396098914364],
                                            [4842, 2775, 4.412217809554578],
                                            [4752, 1974, -1.5422326689561365],
                                            [4770, 1476, -0.9900399732272263],
                                            [4602, 318, 4.680142097949435],
                                            [2538, 1260, 3.1749136494680403],
                                            [1800, 2403, -0.08314123188844125]],
              'austin_41_14021_56605': [[4959, 1341, -1.0552473193359178],
                                        [4305, 54, 3.618938035963465],
                                        [516, 3990, 0.11065722117389565],
                                        [2712, 3996, 0.25367409613864256],
                                        [4308, 3303, 0.0],
                                        [213, 1350, 0.448723344010721],
                                        [1977, 297, 3.4633432079864352],
                                        [9, 3474, 2.0344439357957027],
                                        [411, 1722, 2.256525837701183],
                                        [3405, 2652, 0.5937496667107711],
                                        [3723, 2034, 3.792669375034273],
                                        [1734, 1017, 3.589112628746963]],
              'austin_72_29021_46605': [[192, 33, 1.800028260071892],
                                        [3402, 39, 3.5284683713208214],
                                        [15, 726, 1.9862884227357873],
                                        [1902, 3996, 0.47646741947370663],
                                        [927, 3984, 0.30970294454245617],
                                        [4497, 3984, -1.2397002500907646],
                                        [4251, 3102, -1.1987285679367763],
                                        [4977, 2598, -1.3382393849899876],
                                        [4932, 1839, -1.3258176636680323],
                                        [2562, 2301, 0.46364760900080615],
                                        [774, 2544, 1.9990604796715514],
                                        [234, 1572, 2.0131705497716412],
                                        [78, 3678, 0.30092023436042514],
                                        [1623, 1335, 2.0647376957144776]],
              'detroit_204_45700_25709': [[24, 2163, 1.0636978224025597],
                                          [21, 618, 1.4288992721907325],
                                          [336, 1038, 1.1801892830972098],
                                          [1740, 1290, 1.1071487177940904],
                                          [3294, 747, -0.46364760900080615],
                                          [2514, 1851, 1.5707963267948966]],
              'miami_143_21863_48400': [[2745, 9, 3.2129001183750834],
                                        [3522, 12, 3.1915510493117356],
                                        [4953, 120, 4.509343763131225],
                                        [4686, 3978, -0.8139618212362083],
                                        [3456, 2586, -0.8050034942546529],
                                        [3009, 3984, 0.09347678115858948],
                                        [252, 3951, 0.0],
                                        [30, 3060, 1.5707963267948966],
                                        [24, 900, 1.1441688336680205],
                                        [2943, 2640, 0.0],
                                        [2142, 3000, 1.5011417530663285]],
              'miami_94_1863_43400': [[3708, 3978, 0.05875582271572255],
                                      [3636, 1473, -0.09966865249116208],
                                      [3471, 87, 3.141592653589793],
                                      [3, 2877, 1.5707963267948966],
                                      [741, 231, 3.141592653589793],
                                      [4977, 3378, 4.63022145802299],
                                      [4644, 1179, 4.71238898038469],
                                      [4620, 636, 4.71238898038469],
                                      [1290, 1893, 1.4157995848709555],
                                      [2589, 3453, 1.2637505947761059],
                                      [918, 3168, 0.0]],
              'paloalto_24_15359_8592': [[1083, 4182, 0.6610431688506868],
                                         [204, 3750, 0.7216548508647612],
                                         [15, 2349, 2.2367655641740063],
                                         [27, 906, 2.181522291184105],
                                         [2631, 39, 2.2264919530364327],
                                         [3855, 609, 3.9724140964088184],
                                         [4926, 1791, -0.9440534255838497],
                                         [4773, 3090, -0.8728935041998396],
                                         [4431, 4278, -0.9119902906774207]],
              'paloalto_49_30359_13592': [[15, 489, 1.3415643935179011],
                                          [957, 252, 2.356194490192345],
                                          [3084, 87, 3.8885482269660536],
                                          [3624, 477, 3.8088987885681473],
                                          [3972, 2160, 3.8744077553763],
                                          [4260, 2433, 3.7083218711132995],
                                          [2136, 4257, 0.8441539861131709],
                                          [51, 4065, 0.702256931509007]],
              'pittsburgh_19_12706_31407': [[24, 3432, 1.5707963267948966],
                                            [21, 1899, 1.6447353644528369],
                                            [15, 495, 1.7561442767905913],
                                            [2784, 18, 3.4198923125949046],
                                            [4950, 222, 4.71238898038469],
                                            [4977, 1410, -1.3009471708564275],
                                            [4959, 2631, 4.67536886451076],
                                            [2529, 1905, 1.8076450877418169],
                                            [1938, 1032, 3.2564692590066926],
                                            [1848, 3081, 0.2252767792140553],
                                            [3249, 1533, 0.18822150530477066],
                                            [903, 921, 3.141592653589793]],
              'pittsburgh_67_47706_26407': [[57, 1194, 1.5707963267948966],
                                            [516, 45, 2.7291822119924056],
                                            [3858, 51, 2.8501358591119264],
                                            [4980, 1083, 4.446136931233765],
                                            [4986, 2673, 4.53397848103365],
                                            [4794, 3972, -0.1651486774146269],
                                            [2760, 3930, 0.0],
                                            [1482, 3882, 0.0739390376579403],
                                            [627, 3462, 1.4968572891369563],
                                            [123, 2763, 1.5707963267948966],
                                            [2499, 2247, -0.3455555805817121],
                                            [3612, 1275, -1.3886280163221332],
                                            [4530, 2478, -0.27300870308671077]],
              'washington_48_36634_69625': [[15, 1875, 2.3036114285814033],
                                            [495, 1782, 1.5707963267948966],
                                            [1563, 1221, 3.2112472273183617],
                                            [2178, 1230, 3.1093457711545396],
                                            [1578, 12, 3.1941757152007346],
                                            [2205, 18, 3.3726833207856903],
                                            [2814, 33, 3.141592653589793],
                                            [4131, 12, 3.189175756866777],
                                            [4977, 1053, 4.71238898038469],
                                            [4542, 4197, -0.962107019467485],
                                            [1527, 4215, 0.03123983343026815],
                                            [2766, 4197, 0.0],
                                            [663, 4161, 0.04164257909858837]]}



def move_graph_nodes(g, delta):
    g_ = g.copy(as_view=False)
    for node in g_.nodes():
        g_.nodes[node]['pos'] += delta
    return g_


class AerialDriver(object):
    def __init__(self, debug=False, input_layers=None, tile_id=None):
        self.aerial_image = None

        my_init_poses = np.array(init_poses[tile_id])
        my_init_poses = my_init_poses + np.array([500, 500, 0])
        self.init_pose = my_init_poses[0]

        self.future_poses = []

        if len(init_poses[tile_id]) > 1:
            self.future_poses = list(my_init_poses[1:])


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

        dumpdir = "/home/zuern/Desktop/autograph/G_agg/{}".format(self.tile_id)
        if not os.path.exists(dumpdir):
            os.makedirs(dumpdir)

        # Embed the aerial image into a larger image to avoid edge effects
        self.aerial_image = np.pad(self.aerial_image, ((500, 500), (500, 500),
                                                       (0, 0)), mode="constant", constant_values=0)

        print("Aerial image shape: {}".format(self.aerial_image.shape))

        self.pose += np.array([500, 500, 0])

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
        cv2.imwrite("/home/zuern/Desktop/other/{}-{:04d}_rgb_pose_viz.png".format(self.tile_id, self.step), rgb_pose_viz)

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

        #cv2.waitKey(1000000)

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
            fig, axarr = plt.subplots(1, 6, figsize=(20, 5), sharex=True, sharey=True)
            axarr[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            axarr[0].title.set_text('rgb')
            axarr[1].imshow(pred_drivable)
            axarr[1].title.set_text('pred_drivable - {:.0f}'.format(skeleton_drivable_weight))
            axarr[2].imshow(pred_succ)
            axarr[2].title.set_text('pred_succ - {:.0f}'.format(skeleton_succ_weight))
            axarr[3].imshow(pred_angles_color)
            axarr[3].title.set_text('pred_angles_color')
            axarr[4].imshow(pred_angles_succ_color)
            axarr[4].title.set_text('pred_angles_succ_color')
            axarr[5].imshow(skeleton)
            axarr[5].title.set_text('skeleton')
            plt.savefig("/home/zuern/Desktop/autograph/debug/{}-{:04d}_matplotlib.png".format(self.tile_id, self.step))
            plt.close(fig)

        # cv2.imwrite("/home/zuern/Desktop/autograph/debug/{}-{:04d}_pred_succ_viz.png".format(self.tile_id, self.step), pred_succ_viz)

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

        cv2.imwrite("/home/zuern/Desktop/autograph/G_agg/{}/{:04d}_{}_viz.png".format(self.tile_id, self.step, name), G_agg_viz)

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

        cv2.imwrite("/home/zuern/Desktop/autograph/G_agg/{}/{:04d}_{}_viz.png".format(self.tile_id, self.step, name), G_agg_viz)

        margin = 400
        G_agg_viz = G_agg_viz[int(self.pose[1]) - margin:int(self.pose[1]) + margin,
                    int(self.pose[0]) - margin:int(self.pose[0]) + margin]

        cv2.imshow("G_agg_viz", G_agg_viz)

        # serialize graph
        #
        pickle.dump(G_agg, open("/home/zuern/Desktop/autograph/G_agg/{}/{:04d}_{}.pickle".format(self.tile_id, self.step, name), "wb"))


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

        # if self.step > 4:
        #     self.done = True
        #     return

        fps = 1 / (time.time() - self.time)
        self.time = time.time()

        print("Step: {} | FPS = {:.1f} | Current pose: {:.0f}, {:.0f}, {:.1f}".format(self.step, fps, self.pose[0], self.pose[1], self.pose[2]))

        if self.graph_skeleton is None:
            self.make_step()
            return

        G_current_local = self.graph_skeleton.copy()

        # # calculate the edge weights of the current graph
        # succ_graph_weight = 0
        # for edge in G_current_local.edges:
        #     edges_u = G_current_local.edges[edge]['pts'][:, 0].astype(np.uint8)
        #     edges_v = G_current_local.edges[edge]['pts'][:, 1].astype(np.uint8)
        #     # edge_len = np.linalg.norm(G_current_local.edges[edge]['pts'][0] - G_current_local.edges[edge]['pts'][-1])
        #     # succ_graph_weight += np.sum(self.pred_succ[edges_u, edges_v])
        #     succ_graph_weight += np.sum(self.pred_succ[edges_u, edges_v])

        succ_graph_weight = np.sum(self.skeleton * self.pred_drivable)

        # do branch_alive check
        branch_alive = True
        if succ_graph_weight < 50:
            print("     Successor Graph too weak, aborting branch")
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

            if len(succ_edges) == 0:
                print("     No successor edges found.")


            # loop over all successor edges to find future poses
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
                step_sizes = [40]

                for step_size in step_sizes:

                    # define future pose
                    future_pose_global = np.zeros(3)
                    diff = step_size * (pos_end - pos_start) / np.linalg.norm(pos_end - pos_start)
                    future_pose_global[0:2] = pos_start + diff
                    future_pose_global[2] = self.yaw_check(angle_global)

                    # put future pose in queue if not yet visited
                    was_visited, matches = similarity_check(future_pose_global,
                                                            self.pose_history,
                                                            min_dist=30,
                                                            min_angle=np.pi/4)

                    is_already_in_queue, _ = similarity_check(future_pose_global,
                                                              self.future_poses,
                                                              min_dist=30,
                                                              min_angle=np.pi/4)

                    # print("     Current branch age: {}".format(self.current_branch_age))
                    if was_visited:
                        node_positions = nx.get_node_attributes(self.G_agg_naive, "pos")
                        node_positions = np.array(list(node_positions.values()))
                        nodes_list = np.array(list(self.G_agg_naive.nodes()))

                        for match in matches:
                            visited_pose = self.pose_history[match][np.newaxis, 0:2]
                            # get node in G_agg_naive that corresponds to visited pose
                            closest_node = np.argmin(np.linalg.norm(node_positions - visited_pose, axis=1))
                            edge_end_id = nodes_list[closest_node]
                            edge_end_id = (int(edge_end_id[0]), int(edge_end_id[1]))

                            # now add edge from future_pose_global to node_id
                            try:

                                node_ego = (int(pos_start[0]), int(pos_start[1]))
                                self.G_agg_naive.add_node(node_ego, pos=pos_start)

                                edge_start_pos = future_pose_global[0:2]
                                edge_start_id = (int(edge_start_pos[0]), int(edge_start_pos[1]))

                                self.G_agg_naive.add_node(node_ego, pos=pos_start)
                                self.G_agg_naive.add_node(edge_start_id, pos=edge_start_pos)
                                self.G_agg_naive.add_edge(edge_start_id, edge_end_id)
                                print("     ! Added edge to visited pose: {} -> {}".format(edge_start_id, edge_end_id))

                            except Exception as e:
                                print(e)
                                continue




                    if not was_visited and not is_already_in_queue:

                        self.future_poses.append(future_pose_global)
                        # print("     put pose in queue: {:.0f}, {:.0f}, {:.1f} (step size: {})".format(future_pose_global[0],
                        #                                                                               future_pose_global[1],
                        #                                                                               future_pose_global[2],
                        #                                                                               step_size))

                        # add edge to aggregated graph
                        pointlist = np.array(edge["pts"][edge_start_idx:edge_end_idx])

                        node_edge_start = (int(pos_start[0]), int(pos_start[1]))
                        node_edge_end = (int(pos_end[0]), int(pos_end[1]))

                        # add G_agg-edge from edge start to edge end
                        self.G_agg_naive.add_node(node_edge_start, pos=pos_start)
                        self.G_agg_naive.add_node(node_edge_end, pos=pos_end)
                        self.G_agg_naive.add_edge(node_edge_start, node_edge_end, pts=pointlist)

                        # add G_agg-edge from current pose to edge start
                        if np.linalg.norm(pos_start - self.pose[0:2]) < 50:
                            node_current_pose = (int(self.pose[0]), int(self.pose[1]))
                            self.G_agg_naive.add_node(node_current_pose, pos=self.pose[0:2])
                            self.G_agg_naive.add_edge(node_current_pose, node_edge_start)

                        # add G_agg-edge from edge end to future pose start
                        closest_distance = 100000
                        closest_edge = None
                        for inner_edge in succ_edges:
                            distance = np.linalg.norm(edge["pts"][0] - inner_edge["pts"][-1])
                            if distance < 1e-3: # same edge
                                continue
                            if distance < closest_distance and distance < 100:
                                closest_distance = distance
                                closest_edge = inner_edge

                        if closest_edge is not None:
                            if len(closest_edge["pts"]) > edge_end_idx:
                                print("     adding edge from edge end to future pose start")
                                pos_start = closest_edge["pts"][edge_end_idx]
                                node_start = (int(pos_start[0]), int(pos_start[1]))
                                self.G_agg_naive.add_node(node_start, pos=pos_start)
                                self.G_agg_naive.add_edge(node_start, node_edge_start)

                                self.current_branch_age += 1
                        break

            if self.step % write_every == 0:
                self.render_poses_in_aerial()
                self.visualize_write_G_agg(self.G_agg_naive, "G_agg_naive")
                self.visualize_write_G_single(self.graphs, "G_single")

                cv2.imwrite("/home/zuern/Desktop/autograph/G_agg/{}/{:04d}_angle_canvas.png".format(self.tile_id, self.step), self.canvas_angles)

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

        dumpdir = "/home/zuern/Desktop/autograph/G_agg/{}".format(self.tile_id)

        # write self.graphs to disk
        if not os.path.exists(dumpdir):
            os.makedirs(dumpdir)

        # move to correct position
        G_agg_naive = move_graph_nodes(self.G_agg_naive, [500, 500])

        # smooth graph
        G_agg_naive = laplacian_smoothing(G_agg_naive, gamma=0.2, iterations=1)

        with open("{}/G_agg_naive_all.pickle".format(dumpdir), "wb") as f:
            pickle.dump(G_agg_naive, f)

        # also move all graphs to correct position
        graphs_all = [move_graph_nodes(g, [500, 500]) for g in self.graphs]

        with open("{}/graphs_all.pickle".format(dumpdir), "wb") as f:
            pickle.dump(graphs_all, f)


if __name__ == "__main__":

    input_layers = "rgb+drivable+angles"

    tile_ids = glob.glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/tiles/test/*.png")
    tile_ids = [os.path.basename(t).split(".")[0] for t in tile_ids]

    # tile_ids = ['washington_46_36634_59625']


    for tile_id in tile_ids:

        print("Driving on tile {}".format(tile_id))

        driver = AerialDriver(debug=True, input_layers=input_layers, tile_id=tile_id)

        driver.load_model(model_path="/data/autograph/checkpoints/civilized-bothan-187/e-150.pth",  # (all-3004)
                          type="full")
        driver.load_model(model_path="/data/autograph/checkpoints/jumping-spaceship-188/e-040.pth",  # (all-3004)
                          type="successor",
                          input_layers=input_layers,
                          )

        driver.load_satellite(impath=glob.glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/tiles/*/{}.png".format(tile_id))[0])

        while True:
            driver.drive_freely()
            if driver.done:
                driver.cleanup()
                break

        #
        # # load files from disk
        # with open("/home/zuern/Desktop/autograph/G_agg/{}/graphs_all.pickle".format(driver.tile_id, driver.tile_id), "rb") as f:
        #     graphs = pickle.load(f)
        # with open("/home/zuern/Desktop/autograph/G_agg/{}/G_agg_naive_all.pickle".format(driver.tile_id, driver.tile_id), "rb") as f:
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

