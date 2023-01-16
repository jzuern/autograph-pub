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
from torchmetrics import AUROC, AUC
from torchmetrics.classification import Precision, Recall, BinaryAccuracy, Accuracy
from data.data_gae import ToyDataset
import wandb
from tqdm import tqdm

from model import GCNModelVAE, GCNModelVAE_large
from optimizer import loss_function
from utils import preprocess_graph



class Trainer(object):
    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader

        self.feat_dim = 2
        #self.model = GCNModelVAE(self.feat_dim, args.hidden1, args.hidden2, args.dropout).cuda()
        self.model = GCNModelVAE_large(self.feat_dim,
                                       args.hidden1,
                                       args.hidden2,
                                       args.hidden3,
                                       args.dropout).cuda()
        self.train_losses = []
        self.train_accs = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.epoch = 0
        self.global_step = 0
        self.fig, self.ax = plt.subplots(1, 4, figsize=(20, 5))


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
        labels_all = adj_label.view(-1).float().cpu()
        preds_all = (adj_rec > 0.5).view(-1).float().cpu()

        return BinaryAccuracy()(preds_all, labels_all)


    def train(self):

        self.model.train()

        train_progress = tqdm(self.dataloader, desc="Training", total=len(self.dataloader))

        for data in train_progress:

            self.optimizer.zero_grad()

            adj, features = data
            self.features = features
            features = torch.FloatTensor(features)

            # Store original adjacency matrix (without diagonal entries) for later
            adj_orig = adj
            adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
            adj_orig.eliminate_zeros()

            self.adj = np.array(adj_orig.todense())  # for debugging

            # Some preprocessing
            adj_norm = preprocess_graph(adj)
            adj_label = adj + sp.eye(adj.shape[0])
            adj_label = torch.FloatTensor(adj_label.toarray())

            pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            pos_weight = torch.tensor(pos_weight)

            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

            features = features.cuda()
            adj_norm = adj_norm.cuda()
            adj_label = adj_label.cuda()

            self.adj_recovered, mu, logvar = self.model(features, adj_norm)

            # adj_norm: normalized adjacency matrix
            # adj_label: binary adjacency matrix with diagonal elements
            # adj_recovered: reconstructed adjacency matrix
            # mu: mean of the latent variable
            # logvar: log variance of the latent variable

            loss = loss_function(preds=self.adj_recovered,
                                 labels=adj_label,
                                 mu=mu,
                                 logvar=logvar,
                                 n_nodes=10,
                                 norm=norm,
                                 pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            self.optimizer.step()

            #hidden_emb = mu.data.cpu().numpy()

            self.train_losses.append(cur_loss)


            text = "Epoch: {}, Loss: {:.4f}".format(self.epoch, cur_loss)
            train_progress.set_description(text)

            # get accuracy
            acc = self.get_acc(self.adj_recovered, adj_label).item()
            self.train_accs.append(acc)

            if self.global_step % 20 == 0:
                if not args.disable_wandb:
                    wandb.log({"train_loss": cur_loss,
                               "train_acc": acc, })

            if self.global_step % 1000 == 0:
                self.visualize()

            self.global_step += 1

        self.epoch += 1

        # roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
        # print('Test ROC score: ' + str(roc_score))
        # print('Test AP score: ' + str(ap_score))

    def visualize(self):
        print("Visualizing...")

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
        cv2.waitKey(1)

        if self.epoch % 2 == 0:
            cmap = plt.get_cmap('viridis')
            self.ax[0].clear()
            self.ax[1].clear()
            self.ax[2].clear()
            self.ax[3].clear()
            self.ax[0].set_title("Target")
            self.ax[1].set_title("Prediction")
            self.ax[2].set_title("Loss")
            self.ax[2].set_xlabel("Step")
            self.ax[2].set_yscale("log")
            self.ax[3].set_title("accuracy")
            self.ax[0].set_aspect('equal')
            self.ax[1].set_aspect('equal')

            G_pred = nx.Graph()
            G_pred.add_nodes_from(range(len(adj_pred)))
            G_target = nx.Graph()
            G_target.add_nodes_from(range(len(adj_target)))
            for i in range(len(adj_target)):
                for j in range(i, len(adj_target)):
                    if i != j:
                        G_pred.add_edge(i, j, weight=adj_pred[i, j])
                        G_target.add_edge(i, j, weight=adj_target[i, j])

            edge_scores = nx.get_edge_attributes(G_pred, 'weight')
            edge_scores = np.array([edge_scores[e] for e in G_pred.edges()])
            color_edge_pred = np.hstack([cmap(edge_scores)[:, 0:3], edge_scores[:, None]**5])

            nx.draw(G_pred, pos=self.features,
                    with_labels=False,
                    ax=self.ax[1],
                    node_size=10,
                    edge_color=color_edge_pred,
                    width=2,
                    node_color='b')

            edge_scores = nx.get_edge_attributes(G_target, 'weight')
            edge_scores = np.array([edge_scores[e] for e in G_target.edges()])
            color_edge_target = np.hstack([cmap(edge_scores)[:, 0:3], edge_scores[:, None]])

            nx.draw(G_target, pos=self.features,
                    with_labels=False,
                    ax=self.ax[0],
                    node_size=10,
                    edge_color=color_edge_target,
                    width=2,
                    node_color='b')

            n_avg = 100
            self.ax[2].plot(np.array(np.convolve(self.train_losses, np.ones(n_avg), 'valid') / n_avg))
            self.ax[3].plot(np.array(np.convolve(self.train_accs, np.ones(n_avg), 'valid') / n_avg))

            plt.pause(0.1)

            if not args.disable_wandb:
                wandb.log({"chart": wandb.Image(self.fig)})

    def save_model(self):
        self.save_path = 'checkpoints/gae-{epoch:04d}.ckpt'.format(epoch=self.epoch)
        print("Saving model as " + self.save_path)
        #torch.save(self.model.state_dict(), self.save_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--disable-wandb', '-d', action='store_true', help='Disable wandb logging')
    parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train.')
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=64, help='Number of units in hidden layer 2.')
    parser.add_argument('--hidden3', type=int, default=128, help='Number of units in hidden layer 3.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')

    args = parser.parse_args()
    print(args)


    if not args.disable_wandb:
        wandb.init(
            entity='jannik-zuern',
            project='autograph-gvae',
            notes='gvae',
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = ToyDataset()
    #dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    trainer = Trainer(args, dataset)

    for epoch in range(args.epochs):
        trainer.train()
        trainer.save_model()

        dataset.shuffle_samples()
