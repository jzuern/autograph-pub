import networkx as nx
from karateclub import GL2Vec

from data.data_gae import get_random_circles

graphs = get_random_circles(100)

model = GL2Vec()
model.fit(graphs)
