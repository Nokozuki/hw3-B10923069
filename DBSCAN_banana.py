import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import entropy
from collections import Counter
from sklearn.metrics import adjusted_rand_score
import time

data = pd.read_csv('/data/notebook_files/資料探勘/banana  (with class label).csv')
X = data[['x', 'y']].values
y_true = data['class'].values


X_scaled = StandardScaler().fit_transform(X)


eps_values = [0.3, 0.5, 0.7]
min_samples_values = [5, 10, 15]
markers = ['.','+']
fig, axs = plt.subplots(len(eps_values), len(min_samples_values), figsize=(15, 10), sharex=True, sharey=True)
start_time = time.time()
for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db.labels_
    

        unique_labels = set(labels)
        for k,label in enumerate(unique_labels):
            if label == -1:
                
                axs[i, j].scatter(X_scaled[labels == label, 0], X_scaled[labels == label, 1], 
                                  c='black', s=5, marker=".", label='Noise')
            else:
                
                 axs[i, j].scatter(X_scaled[labels == label, 0], X_scaled[labels == label, 1], 
                                  s=10, marker=markers[k], label=f'Cluster {label}')

        axs[i, j].set_title(f'eps={eps}, min_samples={min_samples}')
        axs[i, j].set_xlabel('x')
        axs[i, j].set_ylabel('y')
        axs[i, j].legend()
end_time = time.time()
plt.tight_layout()
plt.savefig("dbscan.png", dpi=300)
plt.show()

def calculate_metrics(labels, true_labels):
    
    cluster_labels = {}
    for label, true_label in zip(labels, true_labels):
        if label not in cluster_labels:
            cluster_labels[label] = []
        cluster_labels[label].append(true_label)

    total_entropy = 0
    for cluster, distribution in cluster_labels.items():
        if cluster != -1:
            label_count = Counter(distribution)
            probabilities = [count / len(distribution) for count in label_count.values()]
            total_entropy += entropy(probabilities) * len(distribution) / len(labels)

   
    sse = 0
    for cluster in set(labels):
        if cluster != -1:
            cluster_points = X[labels == cluster]
            cluster_center = np.mean(cluster_points, axis=0)
            sse += np.sum((cluster_points - cluster_center) ** 2)

    
    ari = adjusted_rand_score(true_labels, labels)

    return total_entropy, sse, ari


db = DBSCAN(eps=0.3, min_samples=5).fit(X_scaled)
labels = db.labels_
cluster_mertics = calculate_metrics(labels, y_true)
print("DBSCAN used {:.2f} seconds".format(end_time - start_time))
print("Entropy:", cluster_mertics[0])      
print("SSE:", cluster_mertics[1])
print("ARI:", cluster_mertics[2])