import networkx as nx
import numpy as np

"""
Import adjacency and feature matrices from Zachary's Karate Club data set. 
To be used for testing/building Graph ConvNet
"""

def get_adj_mtx():
    G = nx.karate_club_graph()
    print("Building Adjancency Matrix from {}".format(G))

    A = np.zeros((len(G), len(G)), dtype=int)

    for node in G:

        for neighbor in G.neighbors(node):
            A[node, neighbor] = 1

    return A


def get_degree_mtx(A):
    n = A.shape[0]
    # count degrees
    d = np.sum(A, axis=1)
    # Put into diag matrix
    D = np.diag(d)

    # get D^-1/2

    D = np.linalg.inv(D)
    D = np.power(D, .5)


    return D

def get_a_hat(A):
    n = A.shape[0]

    return A + np.eye(n)





def get_features():
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G)

    X = np.zeros((len(G),2), dtype=np.float32)

    for node, point in pos.items():
        X[node, :] = point

    return X


def get_labels(n_samples, class_probs):
    n_classes = len(class_probs)
    classes = range(n_classes)
    y = np.zeros(shape=(n_samples))

    for i in range(n_samples):

        label = np.random.choice(classes, p=class_probs)
        y[i] = label

    return y


def get_weighted_graph(n_nodes):
    W = np.zeros(shape=(n_nodes, n_nodes))
    for i in range(n_nodes):
        lam = np.random.randint(2, 150, size=1)
        W[i, :] = np.random.poisson(lam=lam, size=n_nodes)

    mtx_sum = np.sum(W, axis=None)
    W_norm = np.divide(W, mtx_sum)

    return W_norm


if __name__ == "__main__":

    A = get_adj_mtx()
    print(A)


    X = get_features()
    print(X)

    D = get_degree_mtx(A)
    print(D)

    A_hat = get_a_hat(A)
    print(A_hat)

    y = get_labels(n_samples=X.shape[0], class_probs=[.2, .5, .2, .1])
    print(y)