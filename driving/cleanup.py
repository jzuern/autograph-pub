import pickle
from aggregation.utils import visualize_graph, laplacian_smoothing
import matplotlib.pyplot as plt

fname = "/data/autograph/evaluations/G_agg/austin_72_29021_46605/G_agg_naive_all.pickle"


with open(fname, "rb") as f:
    G_agg = pickle.load(f)



fig, axarr = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
[ax.set_aspect('equal') for ax in axarr]
axarr[0].set_title("Before")
axarr[1].set_title("After")

visualize_graph(G_agg, axarr[0])




# remove node if it has no successors:
G_agg_copy = G_agg.copy(as_view=False)
for node in G_agg_copy.nodes():
    if len(list(G_agg_copy.successors(node))) == 0:
        G_agg.remove_node(node)


print("Number of nodes: {}, Number of edges".format(len(G_agg.nodes()), len(G_agg.edges())))

G_agg_copy = G_agg.copy(as_view=False)
for node in G_agg_copy.nodes():
    if len(list(G_agg_copy.predecessors(node))) == 0:
        G_agg.remove_node(node)

print("Number of nodes: {}, Number of edges".format(len(G_agg.nodes()), len(G_agg.edges())))

# connect edges that are close according to their step attribute


# sample edges more densly: TODO!
G_agg_copy = G_agg.copy(as_view=False)
for edge in G_agg_copy.edges():
    # place midpoint between two nodes
    node_1_pos = G_agg_copy.nodes[edge[0]]["pos"]
    node_2_pos = G_agg_copy.nodes[edge[1]]["pos"]
    midpoint = (node_1_pos + node_2_pos) / 2

    midpoint_node = (int(midpoint[0]), int(midpoint[1]))

    G_agg.add_node(midpoint_node, pos=midpoint)

    # remove edge between two nodes
    G_agg.remove_edge(edge[0], edge[1])


    # add edge between midpoint and both nodes
    G_agg.add_edge(edge[0], midpoint_node)
    G_agg.add_edge(midpoint_node, edge[1])


# smooth graph
G_agg = laplacian_smoothing(G_agg, gamma=0.1, iterations=3)


visualize_graph(G_agg, axarr[1])
plt.show()


#
# with open("{}/G_agg_naive_all.pickle".format(dumpdir), "wb") as f:
#     pickle.dump(self.G_agg_naive, f)
