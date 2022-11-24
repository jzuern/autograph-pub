import numpy as np
import os
from torch_geometric.data import Batch
import networkx as nx
import argparse
from collections import defaultdict
import torch
import torch.utils.data
import torch_geometric.data
from shapely.geometry import MultiLineString, Point

from lanegnn.lanegnn import LaneGNN
from lanegnn.data import GraphDataset, PreGraphDataset
from lanegnn.utils import ParamLib, unbatch_edge_index
from queue import PriorityQueue
import matplotlib.pyplot as plt


def preprocess_predictions(params, model, data, gt_available=True):

    # LOAD DATA
    if params.model.dataparallel:
        data = [item.to(params.model.device) for item in data]
    else:
        data = data.to(params.model.device)

    with torch.no_grad():
        edge_scores, node_scores, endpoint_scores = model(data)

    edge_scores = torch.nn.Sigmoid()(edge_scores).squeeze()
    node_scores = torch.nn.Sigmoid()(node_scores).squeeze()
    endpoint_scores = torch.nn.Sigmoid()(endpoint_scores).squeeze()

    edge_scores_pred = edge_scores.detach().cpu().numpy()
    node_scores_pred = node_scores.detach().cpu().numpy()
    endpoint_scores_pred = endpoint_scores.detach().cpu().numpy()

    # Convert list of Data to DataBatch for post-processing
    if params.model.dataparallel:
        data = Batch.from_data_list(data)

    node_pos = data.x.cpu().numpy()
    node_pos[:, [1, 0]] = node_pos[:, [0, 1]]

    if gt_available:
        edge_scores_gt_onehot = data.edge_dijkstra.cpu().numpy()
    else:
        edge_scores_gt_onehot = None

    img_rgb = data.rgb.cpu().numpy()[0:256, :, :]
    node_pos = data.x.cpu().numpy()
    node_pos[:, [1, 0]] = node_pos[:, [0, 1]]

    fut_nodes = defaultdict(list)
    past_nodes = defaultdict(list)

    # Get starting position
    startpoint_idx = node_pos.shape[0] - 1

    # ADD ALL EDGES ABOVE THRESHOLD TO GRAPH
    nx_selected = nx.DiGraph()
    for edge_idx, edge in enumerate(data.edge_index.T):
        i, j = edge
        i, j = i.item(), j.item()
        if edge_scores_pred[edge_idx] > 0.35:
            nx_selected.add_edge(i, j, weight=1 - edge_scores_pred[edge_idx])
            nx_selected.add_node(j, pos=node_pos[j], score=node_scores_pred[j])
            nx_selected.add_node(i, pos=node_pos[i], score=node_scores_pred[i])
        else:
            # add startpoint-connecting edges to graph in any case
            if i == startpoint_idx or j == startpoint_idx:
                nx_selected.add_edge(i, j, weight=1 - edge_scores_pred[edge_idx])
                nx_selected.add_node(j, pos=node_pos[j], score=node_scores_pred[j])
                nx_selected.add_node(i, pos=node_pos[i], score=node_scores_pred[i])

    # Write fut_nodes and past_nodes dictionaries to be used in UCS
    for local_edge_idx, edge in enumerate(nx_selected.edges()):
        i, j = edge
        fut_nodes[i].append((j, nx_selected.edges[edge]['weight']))
        past_nodes[j].append((i, nx_selected.edges[edge]['weight']))

    # Sort each list entry in fut_nodes & past_nodes by weight
    for key in fut_nodes:
        fut_nodes[key] = sorted(fut_nodes[key], key=lambda x: x[1])
    for key in past_nodes:
        past_nodes[key] = sorted(past_nodes[key], key=lambda x: x[1])


    return edge_scores_pred, node_scores_pred, endpoint_scores_pred, edge_scores_gt_onehot, img_rgb, \
           node_pos, fut_nodes, past_nodes, startpoint_idx


def get_endpoint_ranking(endpoint_scores):
    endpoint_ranking = endpoint_scores.argsort()[::-1]
    return endpoint_ranking


def uniform_cost_search(fut_nodes, start_idx, goal_idx, node_pos, rad_thresh, debug=False):
    visited = set()
    priority_queue = PriorityQueue()
    priority_queue.put((0, [start_idx]))

    while priority_queue:
        if priority_queue.empty():
            if debug:
                print('distance: infinity \nroute: \nnone')
            break

        # priority queue automatically sorts by first element = cost
        cost, route = priority_queue.get()
        curr_idx = route[len(route) - 1]

        if curr_idx not in visited:
            visited.add(curr_idx)
            # Check for goal
            if np.linalg.norm(node_pos[curr_idx]-node_pos[goal_idx]) < rad_thresh:
                route.append(cost)
                if debug:
                    display_route(fut_nodes, route)
                return route
            #if curr_idx == goal_idx:
            #    route.append(cost)
            #    display_route(fut_nodes, route)
            #    return route

        children_idcs = get_unvisited_reachable_nodes(fut_nodes, curr_idx, route, visited, node_pos)

        for _, child_idx in enumerate(children_idcs):
            #print(fut_nodes[child_idx])
            rel_index_child = [elt[0] for elt in fut_nodes[curr_idx]].index(child_idx)
            child_cost = cost + fut_nodes[curr_idx][rel_index_child][1]
            temp = route[:]
            temp.append(child_idx) # Add children node to route
            priority_queue.put((child_cost, temp))

    return priority_queue


def display_route(fut_nodes, route):
    length = len(route)
    distance = route[-1]
    print('Cost: %s' % distance)
    print('Best route: ')
    count = 0
    while count < (length - 2):
        print('%s -> %s' % (route[count], route[count + 1]))
        count += 1
    return

def filter_endpoints(unvisited_endpoints, node_pos, lanegraph_shapely, rad_thresh):
    # Go through all endpoints and check whether they are close to the lanegraph_shapely
    for idx, val in enumerate(unvisited_endpoints):
        if val:
            endpoint_point = Point([(node_pos[idx, 0], node_pos[idx, 1])])
            endpoint_dist = endpoint_point.distance(lanegraph_shapely)
            if endpoint_dist < rad_thresh:
                unvisited_endpoints[idx] = False
    return unvisited_endpoints


def predict_lanegraph(fut_nodes, start_idx, node_scores_pred, endpoint_scores_pred, node_pos, endpoint_thresh=0.5, rad_thresh=20, debug=False):
    # Get endpoint ranking and start with maximum first
    lanegraph = nx.DiGraph()
    multilines = []
    lanegraph_shapely = MultiLineString()
    node_ranking = get_endpoint_ranking(endpoint_scores_pred)
    goal_frontier = node_ranking[0]
    unvisited_endpoints = endpoint_scores_pred > endpoint_thresh

    while goal_frontier:
        ucs_output = uniform_cost_search(fut_nodes, start_idx, goal_frontier, node_pos, rad_thresh, debug=debug)
        if isinstance(ucs_output, PriorityQueue):
            if debug:
                print('No route found')
            unvisited_endpoints[goal_frontier] = False

        elif isinstance(ucs_output, list):
            route = ucs_output

            route_idx = 0
            while route_idx < (len(route) - 2):
                # Add edges to lanegraph object and shapely object
                # TODO
                # Check if edge is already contained in lanegraph
                lanegraph.add_node(route[route_idx], pos=node_pos[route[route_idx]], score=node_scores_pred[route[route_idx]])
                lanegraph.add_node(route[route_idx + 1], pos=node_pos[route[route_idx + 1]], score=node_scores_pred[route[route_idx + 1]])
                lanegraph.add_edge(route[route_idx], route[route_idx + 1])
                multilines.append(((node_pos[route[route_idx], 0], node_pos[route[route_idx], 1]), (node_pos[route[route_idx + 1], 0], node_pos[route[route_idx + 1], 1])))
                # Set edge costs of all edges contained in lanegraph to 0 to enforce future traversal of these edges
                child_list_idx = [elt[0] for elt in fut_nodes[route[route_idx]]].index(route[route_idx + 1])

                fut_nodes[route[route_idx]][child_list_idx] = (fut_nodes[route[route_idx]][child_list_idx][0], 0)
                route_idx += 1
            lanegraph_shapely = MultiLineString(multilines)

        # Filter endpoints based on already found paths & select next maximum-score endpoint and add to frontier
        unvisited_endpoints = filter_endpoints(unvisited_endpoints, node_pos, lanegraph_shapely, rad_thresh)
        node_ranking = get_endpoint_ranking(endpoint_scores_pred)

        # Look whether there are remaining valid endpoints
        if np.sum(unvisited_endpoints) > 0:
            for _, node_idx in enumerate(node_ranking):
                bool_value = unvisited_endpoints[node_idx]
                if unvisited_endpoints[node_idx]:
                    # Check for endpoints still to be traversed that are close enough to the set
                    # of thresholded nodes
                    fut_nodes[node_idx] = [(elt[0], 0) for elt in fut_nodes[node_idx]]
                    # Get all keys and values from fut_nodes
                    all_values_idcs = [item[0] for sublist in fut_nodes.values() for item in sublist]
                    node_set = set(fut_nodes.keys()).union(set(all_values_idcs))

                    cand_dists = defaultdict()
                    for cand_idx in node_set:
                        cand_dists[cand_idx] = np.linalg.norm(node_pos[cand_idx]-node_pos[node_idx])

                    if min(cand_dists) < 30:
                        closest_endpoint_cand = min(cand_dists, key=cand_dists.get)
                    else:
                        continue
                    unvisited_endpoints[node_idx] = False
                    goal_frontier = closest_endpoint_cand
                    break
        else:
            goal_frontier = None

    return lanegraph


def get_unvisited_reachable_nodes(fut_nodes, curr_idx, curr_path, visited, node_pos):
    unvisited_nodes = []
    for node in fut_nodes[curr_idx]:
        if node[0] not in visited:
            if len(curr_path) == 1:
                unvisited_nodes.append(node[0])
            else:
               angle_diff = node_in_corridor(node[0], curr_idx, curr_path[-2], node_pos)
               if angle_diff < np.pi / 3:
                   unvisited_nodes.append(node[0])
            #else:
            #    unvisited_nodes.append(node[0])
    return unvisited_nodes

def node_in_corridor(fut_idx, prev_idx, prevprev_idx, node_pos):
    edge_dx_fut = node_pos[fut_idx, 0] - node_pos[prev_idx, 0]
    edge_dy_fut = node_pos[fut_idx, 1] - node_pos[prev_idx, 1]
    edge_angle_fut = np.arctan2(edge_dy_fut, edge_dx_fut)

    edge_dx_past = node_pos[prev_idx, 0] - node_pos[prevprev_idx, 0]
    edge_dy_past = node_pos[prev_idx, 1] - node_pos[prevprev_idx, 1]
    edge_angle_past = np.arctan2(edge_dy_past, edge_dx_past)

    edge_diff = np.abs(edge_angle_fut - edge_angle_past)
    return edge_diff



def plot_example(params, model, data):

    # LOAD DATA
    edge_scores, node_scores, endpoint_scores = model(data.to(params.model.device))
    edge_scores = torch.nn.Sigmoid()(edge_scores).squeeze()
    node_scores = torch.nn.Sigmoid()(node_scores).squeeze()
    endpoint_scores = torch.nn.Sigmoid()(endpoint_scores).squeeze()

    data.edge_scores = edge_scores
    data.node_scores = node_scores
    data.endpoint_scores = endpoint_scores

    edge_scores_pred = edge_scores.detach().cpu().numpy()
    node_scores_pred = node_scores.detach().cpu().numpy()
    endpoint_scores_pred = endpoint_scores.detach().cpu().numpy()

    node_pos = data.x.cpu().numpy()
    node_pos[:, [1, 0]] = node_pos[:, [0, 1]]

    figure, axarr = plt.subplots(1, 4, figsize=(20,4))

    edge_scores_gt_onehot = data.edge_gt_onehot.cpu().numpy()
    cmap = plt.get_cmap('viridis')
    color_edge_pred = np.hstack([cmap(edge_scores_pred)[:, 0:3], edge_scores_pred[:, None]])
    color_edge_gt = np.hstack([cmap(edge_scores_gt_onehot)[:, 0:3], edge_scores_gt_onehot[:, None]])
    color_node_pred = np.hstack([cmap(node_scores_pred)[:, 0:3], node_scores_pred[:, None]])

    img_rgb = data.rgb.cpu().numpy()[0:256, :, :]
    context_regr_smooth = data.context_regr_smooth.cpu().numpy()

    # crop 256x256 context_regr
    context_regr_smooth = context_regr_smooth[128:384, 128:384, :]
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
    axarr[3].imshow(context_regr_smooth, alpha=0.5)
    axarr[0].title.set_text('UCS Output')
    axarr[1].title.set_text('Model predictions > 0.4')
    axarr[2].title.set_text('Endpoint predictions > 0.5')
    axarr[3].title.set_text('One-hot ground truth')

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
    axarr[1].set_xlim(0, 255)
    axarr[1].set_ylim(255, 0)
    axarr[2].set_xlim(0, 255)
    axarr[2].set_ylim(255, 0)
    axarr[3].set_xlim(0, 255)
    axarr[3].set_ylim(255, 0)

    cmap = plt.get_cmap('viridis')

    # SELECT RELEVANT NODE POSITIONS MEETING THRESHOLD
    # plot edge scores
    for node_idx, node_position in enumerate(node_pos):
        if endpoint_scores_pred[node_idx] > 0.5:
            axarr[2].scatter(node_position[0], node_position[1], color=cmap(endpoint_scores_pred[node_idx]), s=20)

    axarr[2].scatter(128, 255, color='r', s=10)  # Plot origin

    fut_nodes = defaultdict(list)
    past_nodes = defaultdict(list)

    nx_graph = torch_geometric.utils.to_networkx(data, node_attrs=["node_scores"], edge_attrs=["edge_scores"])

    # Get starting position
    startpoint_idx = node_pos.shape[0] - 1

    # ADD ALL EDGES ABOVE THRESHOLD TO GRAPH
    nx_selected = nx.DiGraph()
    for edge_idx, edge in enumerate(data.edge_index.T):
        i, j = edge
        i, j = i.item(), j.item()
        if edge_scores_pred[edge_idx] > 0.03:
            nx_selected.add_edge(i, j, weight=1-edge_scores_pred[edge_idx])
            nx_selected.add_node(j, pos=node_pos[j])
            nx_selected.add_node(i, pos=node_pos[i])
            # Plot arrow based on node_pos for all edges
            axarr[1].arrow(node_pos[i, 0], node_pos[i, 1], node_pos[j, 0] - node_pos[i, 0],
                        node_pos[j, 1] - node_pos[i, 1], color=cmap(edge_scores_pred[edge_idx]), head_width=5, head_length=6)
        else:
            if i == startpoint_idx or j == startpoint_idx:
                nx_selected.add_edge(i, j, weight=1 - edge_scores_pred[edge_idx])
                nx_selected.add_node(j, pos=node_pos[j])
                nx_selected.add_node(i, pos=node_pos[i])
                # Plot edges connected to startpoint
                axarr[1].arrow(node_pos[i, 0], node_pos[i, 1], node_pos[j, 0] - node_pos[i, 0],
                               node_pos[j, 1] - node_pos[i, 1], color=cmap(edge_scores_pred[edge_idx]), head_width=5,
                               head_length=6)

    figure.canvas.draw()
    figure.canvas.flush_events()

    for local_edge_idx, edge in enumerate(nx_selected.edges()):
        i,j = edge
        fut_nodes[i].append((j, nx_selected.edges[edge]['weight']))
        past_nodes[j].append((i, nx_selected.edges[edge]['weight']))

    # Sort each list entry in fut_nodes & past_nodes by weight
    for key in fut_nodes:
        fut_nodes[key] = sorted(fut_nodes[key], key=lambda x: x[1])
    for key in past_nodes:
        past_nodes[key] = sorted(past_nodes[key], key=lambda x: x[1])


    def get_unvisited_reachable_nodes(fut_nodes, curr_idx, curr_path, visited, node_pos):
        unvisited_nodes = []
        for node in fut_nodes[curr_idx]:
            if node[0] not in visited:
                if len(curr_path) == 1:
                    unvisited_nodes.append(node[0])
                else:
                    angle_diff = node_in_corridor(node[0], curr_idx, curr_path[-2], node_pos)
                    if angle_diff < np.pi / 3:
                        unvisited_nodes.append(node[0])
                #else:
                #    unvisited_nodes.append(node[0])
        return unvisited_nodes

    def get_endpoint_ranking(endpoint_scores):
        endpoint_ranking = endpoint_scores.argsort()[::-1]
        return endpoint_ranking


    def uniform_cost_search(fut_nodes, start_idx, goal_idx, node_pos):
        visited = set()
        priority_queue = PriorityQueue()
        priority_queue.put((0, [start_idx]))

        while priority_queue:
            if priority_queue.empty():
                print('distance: infinity \nroute: \nnone')
                break

            # priority queue automatically sorts by first element = cost
            cost, route = priority_queue.get()
            curr_idx = route[len(route) - 1]

            if curr_idx not in visited:
                visited.add(curr_idx)
                # Check for goal
                if np.linalg.norm(node_pos[curr_idx]-node_pos[goal_idx]) < 20:
                    route.append(cost)
                    display_route(fut_nodes, route)
                    return route
                #if curr_idx == goal_idx:
                #    route.append(cost)
                #    display_route(fut_nodes, route)
                #    return route

            children_idcs = get_unvisited_reachable_nodes(fut_nodes, curr_idx, route, visited, node_pos)

            for _, child_idx in enumerate(children_idcs):
                #print(fut_nodes[child_idx])
                rel_index_child = [elt[0] for elt in fut_nodes[curr_idx]].index(child_idx)
                child_cost = cost + fut_nodes[curr_idx][rel_index_child][1]
                temp = route[:]
                temp.append(child_idx) # Add children node to route
                priority_queue.put((child_cost, temp))

        return priority_queue


    def display_route(fut_nodes, route):
        length = len(route)
        distance = route[-1]
        print('Cost: %s' % distance)
        print('Best route: ')
        count = 0
        while count < (length - 2):
            print('%s -> %s' % (route[count], route[count + 1]))
            count += 1
        return

    def filter_endpoints(unvisited_endpoints, node_pos, lanegraph_shapely):
        # Go through all endpoints and check whether they are close to the lanegraph_shapely
        if not lanegraph_shapely.is_empty:
            for idx, val in enumerate(unvisited_endpoints):
                if val:
                    endpoint_point = Point([(node_pos[idx, 0], node_pos[idx, 1])])
                    endpoint_dist = endpoint_point.distance(lanegraph_shapely)
                    if endpoint_dist < 15:
                        unvisited_endpoints[idx] = False
        return unvisited_endpoints


    def predict_lanegraph_ex(fut_nodes, start_idx, endpoint_scores_pred, node_pos):
        # Get endpoint ranking and start with maximum first
        lanegraph = nx.DiGraph()
        multilines = []
        lanegraph_shapely = MultiLineString()
        node_ranking = get_endpoint_ranking(endpoint_scores_pred)

        for rank in node_ranking:
            print(endpoint_scores_pred[rank], node_pos[rank])

        goal_frontier = node_ranking[0]
        unvisited_endpoints = endpoint_scores_pred > 0.5

        while goal_frontier:
            ucs_output = uniform_cost_search(fut_nodes, start_idx, goal_frontier, node_pos)
            if isinstance(ucs_output, PriorityQueue):
                print('No route found')
                unvisited_endpoints[goal_frontier] = False

            elif isinstance(ucs_output, list):
                route = ucs_output

                route_idx = 0
                while route_idx < (len(route) - 2):
                    # Add edges to lanegraph object and shapely object
                    # TODO
                    # Check if edge is already contained in lanegraph
                    lanegraph.add_node(route[route_idx], pos=node_pos[route[route_idx]])
                    lanegraph.add_node(route[route_idx + 1], pos=node_pos[route[route_idx + 1]])
                    lanegraph.add_edge(route[route_idx], route[route_idx + 1])
                    multilines.append(((node_pos[route[route_idx], 0], node_pos[route[route_idx], 1]), (node_pos[route[route_idx + 1], 0], node_pos[route[route_idx + 1], 1])))
                    # Set edge costs of all edges contained in lanegraph to 0 to enforce future traversal of these edges
                    child_list_idx = [elt[0] for elt in fut_nodes[route[route_idx]]].index(route[route_idx + 1])

                    fut_nodes[route[route_idx]][child_list_idx] = (fut_nodes[route[route_idx]][child_list_idx][0], 0)
                    route_idx += 1
                lanegraph_shapely = MultiLineString(multilines)

            # Filter endpoints based on already found paths & select next maximum-score endpoint and add to frontier
            unvisited_endpoints = filter_endpoints(unvisited_endpoints, node_pos, lanegraph_shapely)
            node_ranking = get_endpoint_ranking(endpoint_scores_pred)

            # Look whether there are remaining valid endpoints
            if np.sum(unvisited_endpoints) > 0:
                for _, node_idx in enumerate(node_ranking):
                    bool_value = unvisited_endpoints[node_idx]
                    if unvisited_endpoints[node_idx]:
                        # Look for closest node contained in

                        fut_nodes[node_idx] = [(elt[0], 0) for elt in fut_nodes[node_idx]]
                        # Get all keys and values from fut_nodes
                        all_values_idcs = [item[0] for sublist in fut_nodes.values() for item in sublist]
                        node_set = set(fut_nodes.keys()).union(set(all_values_idcs))

                        cand_dists = defaultdict()
                        for cand_idx in node_set:
                            cand_dists[cand_idx] = np.linalg.norm(node_pos[cand_idx]-node_pos[node_idx])

                        if min(cand_dists) < 30:
                            closest_endpoint_cand = min(cand_dists, key=cand_dists.get)
                        else:
                            continue
                        unvisited_endpoints[node_idx] = False
                        goal_frontier = closest_endpoint_cand
                        break
            else:
                goal_frontier = None
        return lanegraph

    #edge_scores_pred, node_scores_pred, endpoint_scores_pred, edge_scores_gt_onehot, img_rgb, \
    #node_pos, fut_nodes, past_nodes, startpoint_idx = preprocess_predictions(params, model, data)

    lanegraph = predict_lanegraph_ex(fut_nodes, startpoint_idx, endpoint_scores_pred, node_pos)

    if lanegraph.edges():
        for edge in lanegraph.edges():
            i,j = edge
            # Plot arrow based on node_pos for all edges
            axarr[0].arrow(node_pos[i, 0], node_pos[i, 1], node_pos[j, 0] - node_pos[i, 0],
                           node_pos[j, 1] - node_pos[i, 1], color='r', head_width=5,
                           head_length=6)

    figure.canvas.draw()
    figure.canvas.flush_events()
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
    state_dict = torch.load(os.path.join(params.paths.home, params.model.checkpoint),
                                     map_location=torch.device('cuda')
                            )
    model.load_state_dict(state_dict)
    model = model.to(params.model.device)
    model.eval()
    print("Model loaded")

    # define own collator that skips bad samples
    train_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, "preprocessed", "train",
                              params.paths.config_name)
    test_path = os.path.join(params.paths.dataroot, params.paths.rel_dataset, "preprocessed", "test",
                             params.paths.config_name)
    print("DATA PATH (train): ", train_path)
    print("DATA PATH (test): ", test_path)
    dataset_test = PreGraphDataset(params, path=test_path, visualize=params.preprocessing.visualize)

    # idx = 640
    # es funktioniert 0,1 und 2 nicht
    idx = 10
    data = dataset_test[idx] # 820

    plot_example(params, model, dataset_test[idx])



    #examples = [312, 232, 130, 119, 256, 112, 178, 75, 66, 96, 273, 382, 324, 210]
    #examples = [0, 1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    """
    for scene_token in examples:
        try:
            plot_example(params, model, dataset_test[scene_token])
        except Exception as e:
            print(e)
    """




















