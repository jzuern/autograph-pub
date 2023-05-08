import cv2
import numpy as np
import torch
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
from collections import OrderedDict
import torchvision.models as models
from glob import glob
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import networkx as nx
import pickle
import os
from evaluate import evaluate
from tqdm import tqdm
import pprint
from driving.utils import aggregate, colorize, skeleton_to_graph, skeletonize_prediction, roundify_skeleton_graph


class FormatPrinter(pprint.PrettyPrinter):

    def __init__(self, formats):
        super(FormatPrinter, self).__init__()
        self.formats = formats

    def format(self, obj, ctx, maxlvl, lvl):
        if type(obj) in self.formats:
            return self.formats[type(obj)] % obj, 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl)


def visualize_graph(G, ax, aerial_image, node_color=np.array([255, 0, 142])/255., edge_color=np.array([255, 0, 142])/255.):
    if aerial_image is not None:
        ax.imshow(aerial_image)

    nx.draw_networkx(G, ax=ax, pos=nx.get_node_attributes(G, "pos"),
                     edge_color=node_color,
                     node_color=edge_color,
                     with_labels=False,
                     node_size=5,
                     arrowsize=15.0, )


def load_full_model(model_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    model_full = DeepLabv3Plus(models.resnet101(pretrained=True),
                               num_in_channels=3,
                               num_classes=3).cuda()
    model_full.load_state_dict(new_state_dict)
    model_full.eval()

    print("Model {} loaded".format(model_path))

    return model_full


def load_succ_model(model_path, full_model=False, input_layers="rgb+drivable+angles"):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    if full_model is True:
        if input_layers == "rgb":   # rgb [3], pos_enc [3], pred_drivable [1], pred_angles [2]
            num_in_channels = 3
        elif input_layers == "rgb+drivable":
            num_in_channels = 4
        elif input_layers == "rgb+drivable+angles":
            num_in_channels = 6
        else:
            raise ValueError("Unknown input layers: ", input_layers)
    else:
        num_in_channels = 3  # rgb

    model_succ = DeepLabv3Plus(models.resnet101(pretrained=True),
                               num_in_channels=num_in_channels,
                               num_classes=1).cuda()
    model_succ.load_state_dict(new_state_dict)
    model_succ.eval()

    print("Model {} loaded".format(model_path))

    return model_succ


def run_successor_lgp(picklefile):

    split = "eval"

    # Image folder
    test_images = sorted(glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/successor-lgp/{}/*-rgb.png".format(split)))
    test_graphs = sorted(glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/successor-lgp/{}/*.gpickle".format(split)))

    # full model
    full_model_pth = "/data/autograph/checkpoints/civilized-bothan-187/e-150.pth"     # full model tracklets
    #full_model_pth = "/data/autograph/checkpoints/civilized-bothan-187/e-150.pth"     # full model lanegraph

    # succ model
    succ_model = "/data/autograph/checkpoints/cosmic-feather-189/e-010.pth"         # tracklets_joint rgb
    input_layers = "rgb"

    # succ_model = "/data/autograph/checkpoints/jumping-spaceship-188/e-030.pth"      # tracklets_joint rgb+drivable+angles
    # input_layers = "rgb+drivable+angles"


    # Load model
    model_full = load_full_model(model_path=full_model_pth)
    model_succ = load_succ_model(model_path=succ_model,
                                 full_model=True,
                                 input_layers=input_layers)

    pred_dict = {}

    for test_image, test_graph in tqdm(zip(test_images, test_graphs), total=len(test_images), desc="Testing samples"):

        # print("Loading sample: {}".format(test_image))

        sample_id = os.path.basename(test_image).replace("-rgb.png", "")

        city_name = test_image.split("/")[-4]

        if city_name not in pred_dict:
            pred_dict[city_name] = {}

        if split not in pred_dict[city_name]:
            pred_dict[city_name][split] = {}

        img = Image.open(test_image)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gt_graph = pickle.load(open(test_graph, "rb"))

        # Run model
        with torch.no_grad():
            rgb_torch = torch.from_numpy(img).permute(2, 0, 1).float().cuda() / 255.
            rgb_torch = rgb_torch.unsqueeze(0)

            (pred, _) = model_full(rgb_torch)
            pred = torch.nn.functional.interpolate(pred,
                                                   size=rgb_torch.shape[2:],
                                                   mode='bilinear',
                                                   align_corners=True)
            pred_angles = torch.nn.Tanh()(pred[0:1, 0:2, :, :])
            pred_drivable = torch.nn.Sigmoid()(pred[0:1, 2:3, :, :])


            if input_layers == "rgb":  # rgb [3], pos_enc [3], pred_drivable [1], pred_angles [2]
                in_tensor = rgb_torch
            elif input_layers == "rgb+drivable":
                in_tensor = torch.cat([rgb_torch, pred_drivable], dim=1)
            elif input_layers == "rgb+drivable+angles":
                in_tensor = torch.cat([rgb_torch, pred_drivable, pred_angles], dim=1)
            else:
                raise ValueError("Unknown input layers: ", input_layers)

            (pred_succ, features) = model_succ(in_tensor)
            pred_succ = torch.nn.functional.interpolate(pred_succ,
                                                        size=rgb_torch.shape[2:],
                                                        mode='bilinear',
                                                        align_corners=True)

        pred_succ = torch.nn.Sigmoid()(pred_succ)
        pred_succ = pred_succ[0, 0].cpu().detach().numpy()

        skeleton = skeletonize_prediction(pred_succ, threshold=0.05)
        succ_graph = skeleton_to_graph(skeleton)
        succ_graph = roundify_skeleton_graph(succ_graph)


        # relabel nodes
        mapping = {n: i for i, n in enumerate(succ_graph.nodes)}
        succ_graph = nx.relabel_nodes(succ_graph, mapping)

        pred_dict[city_name][split][sample_id] = succ_graph

        # # Visualize
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # visualize_graph(gt_graph, ax[0], aerial_image=img, node_color='white', edge_color='white')
        # visualize_graph(succ_graph, ax[1], aerial_image=img)
        # visualize_graph(gt_graph, ax[2], aerial_image=img, node_color='white', edge_color='white')
        # visualize_graph(succ_graph, ax[2], aerial_image=img)
        # ax[1].imshow(pred_succ, vmin=0, alpha=0.5)
        # plt.show()

    pickle.dump(pred_dict, open(picklefile, "wb"))


if __name__ == "__main__":

    split = "eval"

    predictions_file = 'succ_lgp_eval_autograph.pickle'
    run_successor_lgp(picklefile=predictions_file)

    results_dict = evaluate(annotation_file="/home/zuern/lanegnn-dev/urbanlanegraph_evaluator/annotations_successor_lgp_eval.pickle",
                            user_submission_file=predictions_file,
                            phase_codename="phase_successor_lgp")

    print("austin")
    for k,v in results_dict['submission_result']["austin"][split]["avg"].items():
        print("     {}: {:.3f}".format(k, v))

    print("detroit")
    for k,v in results_dict['submission_result']["detroit"][split]["avg"].items():
        print("     {}: {:.3f}".format(k, v))

    print("miami")
    for k,v in results_dict['submission_result']["miami"][split]["avg"].items():
        print("     {}: {:.3f}".format(k, v))

    print("paloalto")
    for k,v in results_dict['submission_result']["paloalto"][split]["avg"].items():
        print("     {}: {:.3f}".format(k, v))

    print("pittsburgh")
    for k,v in results_dict['submission_result']["pittsburgh"][split]["avg"].items():
        print("     {}: {:.3f}".format(k, v))

    print("washington")
    for k,v in results_dict['submission_result']["washington"][split]["avg"].items():
        print("     {}: {:.3f}".format(k, v))



    print("avg")
    for k,v in results_dict['submission_result'][split]["avg"].items():
        print("     {}: {:.3f}".format(k, v))
