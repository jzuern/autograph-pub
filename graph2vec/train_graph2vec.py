import networkx as nx
from karateclub import Graph2Vec
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from data.data_gae import get_random_circles, get_t_intersections, get_successor_graphs, get_func_graphs


n_per_class = 100

graphs = get_random_circles(n_per_class, n_nodes=1000) + \
         get_t_intersections(n_per_class) + \
         get_successor_graphs(n_per_class) + \
         get_func_graphs(n_per_class)

gt_classes = [0] * n_per_class + [1] * n_per_class + [2] * n_per_class + [3] * n_per_class

model = Graph2Vec(epochs=100)
model.fit(graphs)

latents = model.infer(graphs)


latents_2d = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(latents)

plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=gt_classes)
plt.show()
