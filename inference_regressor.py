import cv2
import numpy as np
import torch
from regressors.reco.deeplabv3.deeplabv3 import DeepLabv3Plus
from collections import OrderedDict
import torchvision.models as models
from glob import glob
import pprint
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2334477275000
import networkx as nx
import pickle
import matplotlib.pyplot as plt

from driving.utils import aggregate, colorize, skeleton_to_graph, skeletonize_prediction, roundify_skeleton_graph


def generate_pos_encoding(crop_shape=(256, 256)):
    q = [crop_shape[0] - 1,
         crop_shape[1] // 2 - 1]

    pos_encoding = np.zeros([crop_shape[0], crop_shape[1], 3], dtype=np.float32)
    x, y = np.meshgrid(np.arange(crop_shape[1]), np.arange(crop_shape[0]))
    pos_encoding[q[0], q[1], 0] = 1
    pos_encoding[..., 1] = np.abs((x - q[1])) / crop_shape[1]
    pos_encoding[..., 2] = np.abs((y - q[0])) / crop_shape[0]
    pos_encoding = (pos_encoding * 255).astype(np.uint8)
    pos_encoding = cv2.cvtColor(pos_encoding, cv2.COLOR_BGR2RGB)

    return pos_encoding



def evaluate(annotation_file, user_submission_file, phase_codename):
    print("Starting Evaluation.....")

    with open(annotation_file, 'rb') as f:
        graphs_gt = pickle.load(f)

    with open(user_submission_file, 'rb') as f:
        graphs_pred = pickle.load(f)

    output = {}
    if phase_codename == "phase_successor_lgp":
        print("%%%%%%%%%%%%%%%%%%%%%\n%%%%%%\tEvaluating for Phase: phase_successor_lgp\n%%%%%%%%%%%%%%%%%%%%%")
        out_dict = evaluate_successor_lgp(graphs_gt, graphs_pred)

        pprint.pprint(out_dict)

        # this goes to the leaderboard (average of all cities
        metrics_successor = out_dict["eval"]["avg"]

        output["result"] = [{"eval_split_succ": metrics_successor}]

        # To display the results in the result file (all cities)
        output["submission_result"] = out_dict

    elif phase_codename == "phase_full_lgp":
        print("%%%%%%%%%%%%%%%%%%%%%\n%%%%%%\tEvaluating for Phase: phase_full_lgp\n%%%%%%%%%%%%%%%%%%%%%")

        out_dict = evaluate_full_lgp(graphs_gt, graphs_pred)

        pprint.pprint(out_dict)

        # the average over all cities for the eval split is this dict entry:
        metrics_full = out_dict["eval"]["avg"]

        output["result"] = [{"eval_split_full": metrics_full}]

        # To display the results in the result file
        output["submission_result"] = output["result"][0]

    elif phase_codename == "phase_planning":
        print("%%%%%%%%%%%%%%%%%%%%%\n%%%%%%\tEvaluating for Phase: phase_planning\n%%%%%%%%%%%%%%%%%%%%%")

        out_dict = evaluate_planning(graphs_gt, graphs_pred)

        pprint.pprint(out_dict)

        # the average over all cities for the eval split is this dict entry:
        metrics_planning = out_dict["eval"]["avg"]

        output["result"] = [{"eval_split_planning": metrics_planning}]

        # To display the results in the result file
        output["submission_result"] = output["result"][0]

    return output


def visualize_graph(G, ax, aerial_image, node_color='white', edge_color='white'):
    if aerial_image is not None:
        ax.imshow(aerial_image)

    nx.draw_networkx(G,
                     ax=ax,
                     pos=nx.get_node_attributes(G, 'pos'),
                     edge_color=edge_color,
                     node_color=node_color,
                     with_labels=False,
                     width=4,
                     arrowsize=10,
                     node_size=20)

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

def load_succ_model(model_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v


    model_succ = DeepLabv3Plus(models.resnet101(pretrained=True),
                               num_in_channels=9,
                               num_classes=1).cuda()
    model_succ.load_state_dict(new_state_dict)
    model_succ.eval()

    print("Model {} loaded".format(model_path))

    return model_succ



def quantitative():
    from urbanlanegraph_evaluator.evaluator import GraphEvaluator

    # Evaluate the submission for each task

    # Task: Successor LGP, Eval Split
    results_dict = evaluate(annotation_file="annotations_successor_lgp_eval.pickle",
                            user_submission_file="annotations_successor_lgp_eval.pickle",
                            phase_codename="phase_successor_lgp")

    # Task: Full LGP, Eval Split
    results_dict = evaluate(annotation_file="annotations_full_lgp_eval.pickle",
                            user_submission_file="annotations_full_lgp_eval.pickle",
                            phase_codename="phase_full_lgp")

    # Task: Planning, Eval Split
    results_dict = evaluate(annotation_file="annotations_full_lgp_eval.pickle",
                            user_submission_file="annotations_full_lgp_eval.pickle",
                            phase_codename="phase_planning")

    pprint.pprint(results_dict)




def qualitative():



    # Load model
    model_full = load_full_model(model_path="/data/autograph/checkpoints/clean-hill-97/e-014.pth")
    model_succ = load_succ_model(model_path="/data/autograph/checkpoints/smart-rain-99/e-023.pth")


    city_name = "Austin"

    # Image folder
    test_images = sorted(glob("/data/lanegraph/urbanlanegraph-dataset-dev/{}/successor-lgp/test/*-rgb.png".format(city_name)))
    test_graphs = sorted(glob("/data/lanegraph/urbanlanegraph-dataset-dev/{}/successor-lgp/test/*.gpickle".format(city_name)))

    pos_encoding = generate_pos_encoding()
    pos_encoding_torch = torch.from_numpy(pos_encoding).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255


    for test_image, test_graph in zip(test_images, test_graphs):

        print("Loading sample: {}".format(test_image))


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

            in_tensor = torch.cat([rgb_torch, pos_encoding_torch, pred_drivable, pred_angles], dim=1)
            in_tensor = torch.cat([in_tensor, in_tensor], dim=0)

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


        # Visualize
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualize_graph(gt_graph, ax[0], aerial_image=img, node_color='white', edge_color='white')
        visualize_graph(succ_graph, ax[1], aerial_image=img, node_color='red', edge_color='red')
        visualize_graph(gt_graph, ax[2], aerial_image=img, node_color='white', edge_color='white')
        visualize_graph(succ_graph, ax[2], aerial_image=img, node_color='red', edge_color='red')
        plt.show()







if __name__ == "__main__":
    qualitative()

    #quantitative()