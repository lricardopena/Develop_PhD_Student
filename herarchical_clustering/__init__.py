import numpy as np
import scipy as sc
import scipy.spatial
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import cophenet, linkage, dendrogram
from scipy.spatial.distance import pdist
import random


if __name__ == '__main__':
    X = []
    with open("S1.txt") as f:
        for line in f:
            X.append(np.array(line.split("    ")[1:]).astype(dtype=float))

    X = np.array(X)
    random.shuffle(X)

    Z = linkage(X)
    c, coph_dists = cophenet(Z, pdist(X))

    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()



