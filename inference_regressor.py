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
import matplotlib.pyplot as plt
import pprint
from driving.utils import skeleton_to_graph, skeletonize_prediction, roundify_skeleton_graph
from random import shuffle
import pandas as pd


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
                     node_size=3,
                     arrowsize=8.0, )


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


def run_successor_lgp(full_model_pth, succ_model_pth, input_layers, picklefile, split):

    # Image folder
    test_images = sorted(glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/successor-lgp/{}/*-rgb.png".format(split)))
    test_graphs = sorted(glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/successor-lgp/{}/*.gpickle".format(split)))
    # test_images = sorted(glob("//data/autograph/all-3004/lanegraph/pittsburgh/test/branching/*-rgb.png"))
    # test_graphs = sorted(glob("/data/lanegraph/urbanlanegraph-dataset-dev/*/successor-lgp/{}/*.gpickle".format(split)))

    # shuffle(test_images)

    # # jointly shuffle them
    joint = list(zip(test_images, test_graphs))
    np.random.shuffle(joint)
    test_images, test_graphs = zip(*joint)

    # Load model
    model_full = load_full_model(model_path=full_model_pth)
    model_succ = load_succ_model(model_path=succ_model_pth,
                                 full_model=True,
                                 input_layers=input_layers)

    pred_dict = {}

    images = []
    images_succ = []
    graphs_pred = []
    graphs_gt = []


    for image_counter, (test_image, test_graph) in tqdm(enumerate(zip(test_images, test_graphs)),
                                                        total=len(test_images),
                                                        desc="Inference on samples"):

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

            if input_layers == "rgb":
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

        skeleton = skeletonize_prediction(pred_succ, threshold=0.15)
        succ_graph = skeleton_to_graph(skeleton)

        succ_graph = roundify_skeleton_graph(succ_graph)


        # relabel nodes
        mapping = {n: i for i, n in enumerate(succ_graph.nodes)}
        succ_graph = nx.relabel_nodes(succ_graph, mapping)

        pred_dict[city_name][split][sample_id] = succ_graph


        images.append(img)
        images_succ.append(np.digitize(pred_succ, np.arange(0, 1.1, 0.1)))
        graphs_pred.append(succ_graph)
        graphs_gt.append(gt_graph)

        # Visualize
        print(sample_id)
        plot_every = 10
        if image_counter % plot_every == 0 and image_counter > 0:
            fig, ax = plt.subplots(plot_every, 4, sharex=True, sharey=True, figsize=(10, 30))
            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            for i in range(plot_every):
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")
                ax[i, 2].axis("off")
                ax[i, 3].axis("off")
                ax[i, 0].set_title(sample_id)
                img = cv2.cvtColor(images[image_counter-i], cv2.COLOR_BGR2RGB)
                visualize_graph(graphs_gt[image_counter-i], ax[i, 0], aerial_image=img, node_color='white', edge_color='white')
                visualize_graph(graphs_pred[image_counter-i], ax[i, 1], aerial_image=img)
                visualize_graph(graphs_gt[image_counter-i], ax[i, 2], aerial_image=img, node_color='white', edge_color='white')
                visualize_graph(graphs_pred[image_counter-i], ax[i, 2], aerial_image=img)
                ax[i, 3].imshow(images_succ[image_counter-i], vmin=1.1, cmap="jet")
            plt.savefig("/home/zuern/Desktop/autograph/eval_succ/viz/{:04d}.svg".format(image_counter))
            # exit()

    pickle.dump(pred_dict, open(picklefile, "wb"))


if __name__ == "__main__":

    split = "test"

    # full model
    full_model_pth = "/data/autograph/checkpoints/civilized-bothan-187/e-150.pth"     # full model tracklets
    #full_model_pth = "/data/autograph/checkpoints/civilized-bothan-187/e-150.pth"     # full model lanegraph

    # succ model dict
    model_dicts = [
        {"model_path": "/data/autograph/checkpoints/jumping-spaceship-188/e-040.pth",
         "model_notes": "tracklets_joint|successor|rgb+drivable+angles",
         "input_layers": "rgb+drivable+angles"},

        {"model_path": "/data/autograph/checkpoints/tough-blaze-198/e-008.pth",
         "model_notes": "tracklets_joint|successor|rgb",
         "input_layers": "rgb"},

        {"model_path": "/data/autograph/checkpoints/hardy-frog-198/e-012.pth",
            "model_notes": "tracklets_joint|successor|rgb+drivable",
            "input_layers": "rgb+drivable"},

        {"model_path": "/data/autograph/checkpoints/splendid-breeze-198/e-010.pth",
            "model_notes": "tracklets_joint|successor|rgb+drivable+angles",
            "input_layers": "rgb+drivable+angles"},

        {"model_path": "/data/autograph/checkpoints/dandy-cherry-199/e-032.pth",
            "model_notes": "tracklets_raw|successor|rgb",
            "input_layers": "rgb"},
    ]

    results_df = pd.DataFrame(columns=["model_name", "model_notes", "split", "iou", "apls", "geo_precision",
                                       "geo_recall","topo_precision","topo_recall","sda@20","sda@50"])

    for model_dict in model_dicts:
        succ_model_pth = model_dict["model_path"]
        model_notes = model_dict["model_notes"]
        input_layers = model_dict["input_layers"]


        model_name = succ_model_pth.split("/")[-2:]
        model_name = "_".join(model_name)
        model_identifier = model_name + "_" + model_notes + "_" + split

        predictions_file = '/home/zuern/Desktop/autograph/eval_succ/{}_predictions.pickle'.format(model_identifier)
        run_successor_lgp(full_model_pth=full_model_pth,
                          succ_model_pth=succ_model_pth,
                          input_layers=input_layers,
                          picklefile=predictions_file,
                          split=split)

        results_dict = evaluate(annotation_file="/home/zuern/lanegnn-dev/urbanlanegraph_evaluator/annotations_successor_lgp_{}.pickle".format(split),
                                user_submission_file=predictions_file,
                                phase_codename="phase_successor_lgp",
                                split=split,)

        print("avg")
        for k, v in results_dict['submission_result'][split]["avg"].items():
            print("     {}: {:.3f}".format(k, v))

        # save dict
        pickle.dump(results_dict, open("/home/zuern/Desktop/autograph/eval_succ/{}_results_dict.pickle".format(model_identifier), "wb"))

        # save results
        results_df = results_df.append({"model_name": model_name,
                                        "model_notes": model_notes,
                                        "split": split,
                                        "iou": results_dict['submission_result'][split]["avg"]["Graph IoU"],
                                        "apls": results_dict['submission_result'][split]["avg"]["APLS"],
                                        "geo_precision": results_dict['submission_result'][split]["avg"]["GEO Precision"],
                                        "geo_recall": results_dict['submission_result'][split]["avg"]["GEO Recall"],
                                        "topo_precision": results_dict['submission_result'][split]["avg"]["TOPO Precision"],
                                        "topo_recall": results_dict['submission_result'][split]["avg"]["TOPO Recall"],
                                        "sda@20": results_dict['submission_result'][split]["avg"]["SDA20"],
                                        "sda@50": results_dict['submission_result'][split]["avg"]["SDA50"]
                                        },
                                        ignore_index=True)

    results_df.to_csv("/home/zuern/Desktop/autograph/eval_succ/results_all.csv", index=False)


