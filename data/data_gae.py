import numpy as np
import torch
import os
from glob import glob
from PIL import Image
import cv2
import time
import torch_geometric.data.dataset
import torchvision.transforms as T
from shapely.geometry import LineString, MultiLineString, Point
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from svgpathtools import svg2paths
import scipy


def random_func(x):
    return np.random.choice([np.sin, np.cos])(x)



def get_func_graphs(n_graphs):

    # define list of random networkx graphs
    graphs = []
    for _ in range(n_graphs):
        # semicircle graph
        x_coords = np.linspace(-2, 2, 10)
        y_coords = random_func(x_coords)
        coordinates = np.array(list(zip(x_coords, y_coords)))
        # coordinates += np.random.uniform(-0.1, 0.1, size=coordinates.shape)
        graph = nx.Graph()
        for i, c in enumerate(coordinates):
            graph.add_node(i, pos=c)
            # graph.add_node(i, pos=np.random.rand(2))
        for i in range(len(coordinates) - 1):
            graph.add_edge(i, i + 1)

        graphs.append(graph)

    return graphs

def get_random_circles(n_graphs):
    graphs = []

    for _ in range(n_graphs):
        center = np.random.normal(loc=0, scale=0.2, size=2)
        radius = np.random.uniform(0.5, 0.55)
        graph = nx.Graph()
        for i in range(0, 200):
            angle = i * np.random.uniform(0.1, 0.12)
            graph.add_node(i, pos=center + radius * np.array([np.cos(angle), np.sin(angle)]))
            if i > 0:
                graph.add_edge(i, i - 1)
        graphs.append(graph)

    return graphs


class ToyDataset(torch_geometric.data.dataset.Dataset):
    def __init__(self):
        super(ToyDataset, self).__init__()
        n_graphs = 10
        circle_interp = 10

        self.graphs = get_random_circles(n_graphs)

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
        #adj = adj.todense()

        return adj, features

