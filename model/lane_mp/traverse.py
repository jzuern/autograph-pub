import networkx as nx
import numpy as np
import cv2
import os

# Please only comment out, do not delete
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import networkx as nx
import argparse
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.utils.data
import torch_geometric.data
from PIL import Image

from lane_mp.lane_mpnn import LaneGNN
from lane_mp.data import GraphDataset, PreGraphDataset
from lane_mp.utils import ParamLib, unbatch_edge_index

#  Please only commment out, do not delete
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt


def plot_example(params, model, data):

    # Load example data
    edge_scores, node_scores = model(data.to(params.model.device))
    edge_scores = torch.nn.Sigmoid()(edge_scores).squeeze()
    node_scores = torch.nn.Sigmoid()(node_scores).squeeze()

    data.edge_scores = edge_scores
    data.node_scores = node_scores

    edge_scores_pred = edge_scores.detach().cpu().numpy()
    node_scores_pred = node_scores.detach().cpu().numpy()

    node_pos = data.x.cpu().numpy()
    node_pos[:, [1, 0]] = node_pos[:, [0, 1]]

    figure, axarr = plt.subplots(1, 4)

    edge_scores_gt_onehot = data.edge_gt_onehot.cpu().numpy()
    # Plot original pred graph
    cmap = plt.get_cmap('viridis')
    color_edge_pred = np.hstack([cmap(edge_scores_pred)[:, 0:3], edge_scores_pred[:, None]])
    color_edge_gt = np.hstack([cmap(edge_scores_gt_onehot)[:, 0:3], edge_scores_gt_onehot[:, None]])
    color_node_pred = np.hstack([cmap(node_scores_pred)[:, 0:3], node_scores_pred[:, None]])

    img_rgb = data.rgb.cpu().numpy()[0:256, :, :]
    node_pos = data.x.cpu().numpy()
    node_pos[:, [1, 0]] = node_pos[:, [0, 1]]

    axarr[0].cla()
    axarr[1].cla()
    axarr[2].cla()
    axarr[3].cla()
    axarr[0].imshow(img_rgb)
    axarr[1].imshow(img_rgb)
    axarr[2].imshow(img_rgb)
    axarr[3].imshow(img_rgb)
    axarr[0].title.set_text('prediction')
    axarr[3].title.set_text('one-hot GT')

    pred_nx_graph = torch_geometric.utils.to_networkx(data, node_attrs=["node_scores"], edge_attrs=["edge_scores"])
    gt_nx_graph = torch_geometric.utils.to_networkx(data, node_attrs=["node_scores"], edge_attrs=["edge_gt_onehot"])

    #nx.draw_networkx(pred_nx_graph, ax=axarr[0], pos=node_pos, edge_color=color_edge_pred,
    #                 node_color=color_node_pred, with_labels=False, node_size=5)

    nx.draw_networkx(pred_nx_graph, ax=axarr[3], pos=node_pos, edge_color=color_edge_gt,
                     with_labels=False, node_size=0)

    figure.canvas.draw()
    figure.canvas.flush_events()

    axarr[0].set_xlim(0, 255)
    axarr[0].set_ylim(255, 0)

    cmap = plt.get_cmap('viridis')

    # SELECT RELEVANT NODE POSITIONS MEETING THRESHOLD
    selected_nodes_dist = defaultdict()
    # plot edge scores
    for node_idx, node_position in enumerate(node_pos):
        if node_scores_pred[node_idx] > 0.3:
            # Plot node positiions
            #axarr[2].scatter(node_position[0], node_position[1], color=cmap(node_scores_pred[node_idx]), s=20)
            selected_nodes_dist[node_idx] = np.linalg.norm(node_position - np.array([128, 256]))

    # Get minimum distance node from selected_nodes_dist
    min_dist_node_idx = min(selected_nodes_dist, key=selected_nodes_dist.get)

    axarr[2].scatter(128, 255, color='r', s=10)  # Plot origin

    fut_nodes = defaultdict(list)
    past_nodes = defaultdict(list)

    nx_graph = torch_geometric.utils.to_networkx(data, node_attrs=["node_scores"], edge_attrs=["edge_scores"])

    # ADD ALL EDGES ABOVE THRESHOLD TO GRAPH
    nx_selected = nx.DiGraph()
    for edge_idx, edge in enumerate(data.edge_index.T):
        i, j = edge
        i, j = i.item(), j.item()
        if edge_scores_pred[edge_idx] > 0.6:
            nx_selected.add_edge(i, j, weight=1-edge_scores_pred[edge_idx])
            nx_selected.add_node(j, pos=node_pos[j])
            nx_selected.add_node(i, pos=node_pos[i])

            # Plot arrow based on node_pos for all edges
            axarr[1].arrow(node_pos[i, 0], node_pos[i, 1], node_pos[j, 0] - node_pos[i, 0],
                        node_pos[j, 1] - node_pos[i, 1], color=cmap(edge_scores_pred[edge_idx]), head_width=5, head_length=6)

    # INCLUDE ORIGIN NODE AND ORIGIN-CLOSEST-EDGE
    # Obtain max node idx from nx_selected
    max_node_idx = max(nx_selected.nodes())
    nx_selected.add_node(max_node_idx+1, pos=np.array([128, 255]))
    nx_selected.add_edge(max_node_idx+1, min_dist_node_idx, weight=0)

    #node_pos = np.vstack([node_pos, np.array([128, 255])])

    axarr[2].arrow(128, 255,
                   node_pos[min_dist_node_idx, 0] - 128,
                   node_pos[min_dist_node_idx, 1] - 255, head_width=5, head_length=6,
                   color="red")

    figure.canvas.draw()
    figure.canvas.flush_events()

    img_rgb = data.rgb.detach().cpu().numpy()
    axarr[1].set_xlim(0, 255)
    axarr[1].set_ylim(255, 0)


    edges_to_start = []
    # Get edge mid points in certain radius of (255, 128)

    # List will hold tuples like (start_idx, end_idx, weight)
    outgoing_edges = list(nx_selected.edges(min_dist_node_idx, data="weight"))

    for edge in outgoing_edges:

        #edge_mid = (nx_selected.nodes[edge[0]]['pos'] + nx_selected.nodes[edge[1]]['pos']) / 2
        #if np.linalg.norm(edge_mid - np.array([128, 256])) < 40:
        #axarr[0].arrow(node_pos[edge[0], 0], node_pos[edge[0], 1], node_pos[edge[1], 0] - node_pos[edge[0], 0],
        #            node_pos[edge[1], 1] - node_pos[edge[0], 1], head_width=5, head_length=6, color="red")
        #plt.show()
        # If edge direction is within -45 and +45 degrees add to selected graph
        edge_dx = nx_selected.nodes[edge[1]]['pos'][0] - nx_selected.nodes[edge[0]]['pos'][0]
        edge_dy = nx_selected.nodes[edge[1]]['pos'][1] - nx_selected.nodes[edge[0]]['pos'][1]
        edge_angle = np.arctan2(edge_dy, edge_dx)

        if -3*np.pi/4 < edge_angle < - np.pi / 4:
            axarr[2].arrow(node_pos[edge[0], 0], node_pos[edge[0], 1], node_pos[edge[1], 0] - node_pos[edge[0], 0],
                        node_pos[edge[1], 1] - node_pos[edge[0], 1], head_width=5, head_length=6, color="red")
            # Add edge to selected graph including edge weight
            edges_to_start.append((edge[0], edge[1], edge[2]))

            # plot arrow based on node_pos of selected edge
            #axarr[0].arrow(nx_selected.nodes[edge[0]]['pos'][0], nx_selected.nodes[edge[0]]['pos'][1], edge_dx, edge_dy, color="red")

    edges_to_start = sorted(edges_to_start, key=lambda x: x[2])


    for local_edge_idx, edge in enumerate(nx_selected.edges()):
        i,j = edge
        fut_nodes[i].append((j, nx_selected.edges[edge]['weight']))
        past_nodes[j].append((i, nx_selected.edges[edge]['weight']))

    # Sort each list entry in fut_nodes by weight
    for key in fut_nodes:
        fut_nodes[key] = sorted(fut_nodes[key], key=lambda x: x[1])
    # Sort each list entry in past_nodes by weight
    for key in past_nodes:
        past_nodes[key] = sorted(past_nodes[key], key=lambda x: x[1])

    def get_unvisited_nodes(list_of_fut_nodes):
        unvisited_nodes = []
        for node in list_of_fut_nodes:
            if node[0] not in visited:
                unvisited_nodes.append(node[0])
        return unvisited_nodes

    def get_unvisited_reachable_nodes(fut_nodes, node_idx, curr_path):
        unvisited_nodes = []
        for node in fut_nodes[node_idx]:
            if node[0] not in visited:
                if node_in_corridor(node[0], node_idx, curr_path[-2]) <  np.pi / 3:
                    unvisited_nodes.append(node[0])
        return unvisited_nodes

    def node_in_corridor(fut_idx, prev_idx, prevprev_idx):
        edge_dx_fut = nx_selected.nodes[fut_idx]['pos'][0] - nx_selected.nodes[prev_idx]['pos'][0]
        edge_dy_fut = nx_selected.nodes[fut_idx]['pos'][1] - nx_selected.nodes[prev_idx]['pos'][1]
        edge_angle_fut = np.arctan2(edge_dy_fut, edge_dx_fut)

        edge_dx_past = nx_selected.nodes[prev_idx]['pos'][0] - nx_selected.nodes[prevprev_idx]['pos'][0]
        edge_dy_past = nx_selected.nodes[prev_idx]['pos'][1] - nx_selected.nodes[prevprev_idx]['pos'][1]
        edge_angle_past = np.arctan2(edge_dy_past, edge_dx_past)

        edge_diff = np.abs(edge_angle_fut - edge_angle_past)
        return edge_diff


    visited = list()
    disregarded_edges = list()
    for edge in edges_to_start:
        curr_path = []
        curr_path.append(edge[0])
        curr_path.append(edge[1])
        visited.append(edge[1])
        visited.append(edge[0])
        node_idx = edge[1]
        prev_node_idx_constant = 0
        #while get_unvisited_nodes(fut_nodes[node_idx]) != []:
        while get_unvisited_reachable_nodes(fut_nodes, node_idx, curr_path) != []:
            prev_node_idx = node_idx

            for fut_node in fut_nodes[node_idx]:
                if fut_node[0] not in visited and (prev_node_idx, fut_node[0]) not in disregarded_edges:
                    angle_diff = node_in_corridor(fut_node[0], prev_node_idx, curr_path[-2])
                    if angle_diff < np.pi / 3:
                        node_idx = fut_node[0]
                        curr_path.append(node_idx)
                        visited.append(node_idx)
                        # Plot path from start to end4
                        axarr[2].arrow(node_pos[prev_node_idx, 0], node_pos[prev_node_idx, 1],
                                    node_pos[node_idx, 0] - node_pos[prev_node_idx, 0],
                                    node_pos[node_idx, 1] - node_pos[prev_node_idx, 1], head_width=5, head_length=6,
                                    color="red")

                        # Remove parallel edges from potential edges to traverse

                        break
                    else:
                        disregarded_edges.append((prev_node_idx, fut_node[0]))


    # MST of directed graph
    mst = nx.algorithms.minimum_spanning_tree(nx_selected.to_undirected())
    mst_edges = [(i, o, w) for i, o, w in nx_selected.edges(data=True) if ((i, o) in mst.edges() or (o, i) in mst.edges())]

    mst_graph = nx.Graph()
    for edge in mst_edges:
        mst_graph.add_edge(edge[0], edge[1], weight=edge[2])
        try:
            axarr[0].arrow(node_pos[edge[0]][0], node_pos[edge[0]][1], node_pos[edge[1]][0] - node_pos[edge[0]][0], node_pos[edge[1]][1] - node_pos[edge[0]][1], head_width=5, head_length=6, color="red")
        except Exception as e:
            print(e)

    visited_mst = list()
    disregarded_edges_mst = list()
    for edge in edges_to_start:
        curr_path = []
        curr_path.append(edge[0])
        curr_path.append(edge[1])
        visited_mst.append(edge[1])
        visited_mst.append(edge[0])
        node_idx = edge[1]
        prev_node_idx_constant = 0
        while get_unvisited_reachable_nodes(fut_nodes, node_idx, curr_path) != []:
            prev_node_idx = node_idx

            for fut_node in fut_nodes[node_idx]:
                if fut_node[0] not in visited_mst and (prev_node_idx, fut_node[0]) not in disregarded_edges_mst:
                    angle_diff = node_in_corridor(fut_node[0], prev_node_idx, curr_path[-2])
                    if angle_diff < np.pi / 3:
                        node_idx = fut_node[0]
                        curr_path.append(node_idx)
                        visited_mst.append(node_idx)
                        # Plot path from start to end4
                        axarr[2].arrow(node_pos[prev_node_idx, 0], node_pos[prev_node_idx, 1],
                                       node_pos[node_idx, 0] - node_pos[prev_node_idx, 0],
                                       node_pos[node_idx, 1] - node_pos[prev_node_idx, 1], head_width=5, head_length=6,
                                       color="red")

                        # Remove parallel edges from potential edges to traverse

                        break
                    else:
                        disregarded_edges_mst.append((prev_node_idx, fut_node[0]))

    """
    di_graph = nx.DiGraph()
    for local_edge_idx, edge in enumerate(mst.edges()):
        i,j = edge
        fut_nodes[edge[0]].append((edge[1], mst.edges[edge]['weight']))
        past_nodes[edge[1]].append((edge[0], mst.edges[edge]['weight']))
        # Obtain edge weight from mst and add to di_graph
        di_graph.add_edge(edge[0], edge[1], weight=mst.edges[edge]['weight'])

        # Plot arrow based on node_pos
        #axarr[0].arrow(nx_selected.nodes[i]['pos'][0], nx_selected.nodes[i]['pos'][1], nx_selected.nodes[j]['pos'][0] - nx_selected.nodes[i]['pos'][0],
        #            nx_selected.nodes[j]['pos'][1] - nx_selected.nodes[i]['pos'][1], head_width=5, head_length=6)

    curr_node = max_node_idx+1
    traversed = list()

    print(fut_nodes[curr_node])

    while fut_nodes[curr_node] != []:
        # Select next edge from tuple with highest weight
        curr_node = min(fut_nodes[curr_node], key=lambda x: x[1])
        traversed.append(curr_node)
    print(traversed)


    print(list(nx.dfs_edges(di_graph, source=max_node_idx+1, depth_limit=10)))
    """


    plt.xlim(0, 255)
    plt.ylim(255, 0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ----------- Parameter sourcing --------------

    parser = argparse.ArgumentParser(description="Train LaneMP architecture")

    # General parameters (namespace: main)
    parser.add_argument('--config', type=str, help='Provide a config YAML!', required=True)
    parser.add_argument('--dataset', type=str, help="dataset path")
    parser.add_argument('--version', type=str, help="define the dataset version that is used")

    # Namespace-specific arguments (namespace: training)
    parser.add_argument('--lr', type=str, help='model path')
    parser.add_argument('--epochs', type=str, help='model path')

    opt = parser.parse_args()

    params = ParamLib(opt.config)
    params.main.overwrite(opt)
    params.preprocessing.overwrite(opt)
    params.model.overwrite(opt)

    # -------  Model and data initialization ------

    model = LaneGNN(gnn_depth=params.model.gnn_depth,
                    edge_geo_dim=params.model.edge_geo_dim,
                    map_feat_dim=params.model.map_feat_dim,
                    edge_dim=params.model.edge_dim,
                    node_dim=params.model.node_dim,
                    msg_dim=params.model.msg_dim,
                    in_channels=params.model.in_channels,
                    )
    model.load_state_dict(torch.load(os.path.join(params.paths.checkpoints, params.model.checkpoint),
                                     map_location=torch.device('cuda')))
    model = model.to(params.model.device)
    model.eval()
    print("Model loaded")

    # define own collator that skips bad samples
    train_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, "preprocessed", "train",
                              params.paths.config_name)
    test_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, "preprocessed", "test",
                             params.paths.config_name)
    #trainoverfit_path = os.path.join("/data2/buechner/lanegraph/data/", params.paths.rel_dataset, "preprocessed",
    #                                 "trainoverfit", params.paths.config_name)
    dataset_test = PreGraphDataset(params, path=train_path, visualize=params.preprocessing.visualize)

    data = dataset_test[640] # 820

    #plot_example(params, model, dataset_test[640])

    plot_example(params, model, dataset_test[820])
    #plot_example(params, model, dataset_test[650])
    #plot_example(params, model, dataset_test[280])
    #plot_example(params, model, dataset_test[440])
    #plot_example(params, model, dataset_test[860])
    #plot_example(params, model, dataset_test[830])
    #plot_example(params, model, dataset_test[850])
    """

    examples = [994, 932, 730, 719, 656, 612, 578, 575, 566, 496, 473, 382, 376, 260]

    for scene_token in examples:
        plot_example(params, model, dataset_test[scene_token])
    """




















