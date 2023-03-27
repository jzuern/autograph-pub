from __future__ import division
from __future__ import print_function

import argparse
import time
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import networkx as nx

from data.data_gae import ToyDataset


from model import GCNModelVAE, GCNModelVAE_large
from optimizer import loss_function
from utils import mask_test_edges, preprocess_graph, get_roc_score, sigmoid




class Trainer(object):
    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader

        self.feat_dim = 2

        self.model = GCNModelVAE_large(self.feat_dim, args.hidden1, args.hidden2, args.hidden3, args.dropout).cuda()

        print("Loading model from {}".format(args.model_path))
        self.model.load_state_dict(torch.load(args.model_path))


    def get_scores(self, adj_orig, edges_pos, edges_neg, adj_rec):

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(self, adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy


    def test_decode(self):
        self.model.eval()

    def test(self):

        self.model.eval()

        for data in self.dataloader:

            adj, features = data
            self.features = features
            features = torch.FloatTensor(features)

            # get random nonzero elemnts of adjancecy matrix and set them to zero
            nonzeros = np.nonzero(adj)
            nonzeros = np.stack([nonzeros[0], nonzeros[1]]).T
            test_percent = 0.2
            test_size = int(len(nonzeros) * test_percent)
            test_idx = np.random.choice(len(nonzeros), test_size, replace=False)
            nonzeros_test = nonzeros[test_idx]
            adj[nonzeros_test[:, 0], nonzeros_test[:, 1]] = 0
            adj[nonzeros_test[:, 1], nonzeros_test[:, 0]] = 0

            # Store original adjacency matrix (without diagonal entries) for later
            adj_orig = adj
            adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
            adj_orig.eliminate_zeros()

            self.adj = np.array(adj.todense())  # for debugging

            # Some preprocessing
            adj_norm = preprocess_graph(adj).cuda()
            features = features.cuda()

            self.adj_recovered, mu, logvar = self.model(features, adj_norm)

            self.visualize()



    def visualize(self):
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
        self.ax[0].set_aspect('equal')
        self.ax[1].set_aspect('equal')

        # visualize adjacency matrix
        adj_pred = torch.sigmoid(self.adj_recovered).cpu().detach().numpy()
        adj_target = np.array(self.adj, dtype=np.float32)

        adj_pred_viz = cv2.applyColorMap(np.uint8(255 * adj_pred), cv2.COLORMAP_JET)
        adj_target_viz = cv2.applyColorMap(np.uint8(255 * adj_target), cv2.COLORMAP_JET)

        # resize
        adj_target_viz = cv2.resize(adj_target_viz, (300, 300), interpolation=cv2.INTER_AREA)
        adj_pred_viz = cv2.resize(adj_pred_viz, (300, 300), interpolation=cv2.INTER_AREA)

        # concatenate
        adj_viz = np.concatenate((adj_target_viz, adj_pred_viz), axis=1)
        cv2.imshow("adj_viz", adj_viz)
        cv2.waitKey(10)

        cmap = plt.get_cmap('viridis')
        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[0].set_title("Target")
        self.ax[1].set_title("Prediction")

        G_pred = nx.Graph()
        G_pred.add_nodes_from(range(len(adj_pred)))
        G_target = nx.Graph()
        G_target.add_nodes_from(range(len(adj_target)))
        for i in range(len(adj_target)):
            for j in range(i, len(adj_target)):
                if i != j:
                    G_pred.add_edge(i, j, weight=adj_pred[i, j])
                    G_target.add_edge(i, j, weight=adj_target[i, j])

        edge_scores_pred = nx.get_edge_attributes(G_pred, 'weight')
        edge_scores_pred = np.array([edge_scores_pred[e] for e in G_pred.edges()])
        edge_scores_pred[edge_scores_pred < 0.5] = 0

        color_edge_pred = np.hstack([cmap(edge_scores_pred)[:, 0:3], edge_scores_pred[:, None]**5])


        nx.draw(G_pred, pos=self.features,
                with_labels=False,
                ax=self.ax[1],
                node_size=10,
                edge_color=color_edge_pred,
                width=2,
                node_color='b')

        edge_scores_target = nx.get_edge_attributes(G_target, 'weight')
        edge_scores_target = np.array([edge_scores_target[e] for e in G_target.edges()])
        edge_scores_target[edge_scores_target < 0.5] = 0

        color_edge_target = np.hstack([cmap(edge_scores_target)[:, 0:3], edge_scores_target[:, None]])


        nx.draw(G_target, pos=self.features,
                with_labels=False,
                ax=self.ax[0],
                node_size=10,
                edge_color=color_edge_target,
                width=2,
                node_color='b')


        plt.pause(0.01)
        plt.show()

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pth")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=64, help='Number of units in hidden layer 2.')
    parser.add_argument('--hidden3', type=int, default=128, help='Number of units in hidden layer 3.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--model_path', type=str, default='model-0007.ckpt', help='Path to save the model.')

    args = parser.parse_args()

    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = ToyDataset()
    #dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    trainer = Trainer(args, dataset)

    trainer.test()
    #trainer.test_decode()
