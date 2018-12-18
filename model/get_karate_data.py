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


def get_features():
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G)

    X = np.zeros((len(G),2), dtype=np.float32)

    for node, point in pos.items():
        X[node, :] = point

    return X


if __name__ == "__main__":

    A = get_adj_mtx()
    print(A)

    X = get_features()
    print(X)