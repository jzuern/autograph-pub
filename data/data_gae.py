import numpy as np
from glob import glob
import torch_geometric.data.dataset
import networkx as nx
import json
import codecs
import matplotlib.pyplot as plt


def random_func(x):
    return np.random.choice([np.sin, np.cos])(x)


def get_successor_graphs(n_graphs):
    json_files = sorted(glob("/data/lanegraph/data_sep/all/010922-large/pao/test/*.json"))
    print(len(json_files))
    graphs = []

    for json_file in json_files:
        graph = json.loads(codecs.open(json_file, 'r', encoding='utf-8').read())

        # GT graph representation
        waypoints = np.array(graph["bboxes"])
        relation_labels = np.array(graph["relation_labels"])

        waypoints /= waypoints.max()
        waypoints -= waypoints.mean()

        # Get 1 graph start node and N graph end nodes
        G_gt_nx = nx.DiGraph()
        for e in relation_labels:
            if not G_gt_nx.has_node(e[0]):
                G_gt_nx.add_node(e[0], pos=waypoints[e[0]])
            if not G_gt_nx.has_node(e[1]):
                G_gt_nx.add_node(e[1], pos=waypoints[e[1]])
            G_gt_nx.add_edge(e[0], e[1])

        if max(list([G_gt_nx.out_degree(node) for node in G_gt_nx.nodes])) >= 2:
            graphs.append(G_gt_nx)

        if len(graphs) >= n_graphs:
            break

    return graphs



def get_func_graphs(n_graphs):

    # define list of random networkx graphs
    graphs = []
    for _ in range(n_graphs):
        # semicircle graph
        x_coords = np.linspace(-1, 1, 10)
        y_coords = random_func(x_coords)
        coordinates = np.array(list(zip(x_coords, y_coords)))
        # coordinates += np.random.uniform(-0.1, 0.1, size=coordinates.shape)
        graph = nx.Graph()
        for i, c in enumerate(coordinates):
            graph.add_node(i, pos=c)
            # graph.add_node(i, pos=np.random.rand(2))
        for i in range(len(coordinates) - 1):
            graph.add_edge(i, i + 1)

        # randomly rotate nodes around origin
        theta = np.random.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for i in range(len(coordinates)):
            graph.nodes[i]["pos"] = np.matmul(R, graph.nodes[i]["pos"])


        graphs.append(graph)

    return graphs


def get_random_circles(n_graphs, n_nodes=10):

    graphs = []

    for _ in range(n_graphs):
        center = np.random.normal(loc=0, scale=0.01, size=2)
        radius = np.random.normal(loc=0.5, scale=0.1)
        graph = nx.Graph()
        for i in range(0, n_nodes):
            angle = i * 0.6
            graph.add_node(i, pos=center + radius * np.array([np.cos(angle), np.sin(angle)]))
            if i > 0:
                graph.add_edge(i, i - 1)
        graphs.append(graph)

    return graphs


def get_t_intersections(n_graphs):
    graphs = []

    for _ in range(n_graphs):
        # T intersection graph
        n_interp = 5

        lane_0 = np.linspace((0.1, 0.5), (0.4, 0.5), n_interp)
        lane_1 = np.linspace((0.5, 0.9), (0.5, 0.1), n_interp)

        # add some noise
        lane_0 += np.random.normal(loc=0, scale=0.01, size=lane_0.shape)
        lane_1 += np.random.normal(loc=0, scale=0.01, size=lane_1.shape)

        graph = nx.Graph()
        for i, c in enumerate(lane_0):
            graph.add_node(i, pos=c)
        for i, c in enumerate(lane_1):
            graph.add_node(i+n_interp, pos=c)

        for i in range(len(lane_0) - 1):
            graph.add_edge(i, i + 1)
        for i in range(len(lane_1) - 1):
            graph.add_edge(i+n_interp, i + 1+n_interp)

        # connect two lanes
        graph.add_edge(n_interp-1, n_interp + n_interp//2)



        # randomly rename nodes
        node_mapping = np.random.permutation(list(graph.nodes))
        graph = nx.relabel_nodes(graph, dict(zip(graph.nodes, node_mapping)))


        # rotate graph around origin
        theta = np.random.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        for i in range(len(lane_0) + len(lane_1)):
            graph.nodes[i]["pos"] = np.matmul(R, graph.nodes[i]["pos"])

        graphs.append(graph)

    return graphs


class ToyDataset(torch_geometric.data.dataset.Dataset):
    def __init__(self):
        super(ToyDataset, self).__init__()
        n_graphs = 10000

        #self.graphs = get_random_circles(n_graphs)
        #self.graphs = get_func_graphs(n_graphs)
        #self.graphs = get_t_intersections(n_graphs)
        self.graphs = get_successor_graphs(1001)


    def shuffle_samples(self):
        np.random.shuffle(self.graphs)


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        graph = self.graphs[item]


        # get feature matrix
        features = np.array([graph.nodes[n]['pos'] for n in graph.nodes])

        # get adjacency matrix
        adj = nx.adjacency_matrix(graph)

        return adj, features

