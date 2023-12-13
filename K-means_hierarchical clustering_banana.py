import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.stats import entropy
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/data/notebook_files/資料探勘/banana  (with class label).csv')

X = data[['x', 'y']]
y = data['class']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

Kmeans_start_time = time.time()

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(X)

Kmeans_end_time = time.time()
Kmeans_elapsed_time = Kmeans_end_time - Kmeans_start_time

#SSE
Kmeans_sse = kmeans.inertia_

#entropy
cluster_entropy = []
for i in np.unique(clusters):
    cluster_labels = data[clusters == i]['class']
    cluster_entropy.append(entropy(cluster_labels.value_counts(normalize=True), base=2))
Kmeans_mean_entropy = np.mean(cluster_entropy)

#Accuracy
def adjust_labels(true_labels, predicted_labels):
    from scipy.stats import mode

    labels = np.zeros_like(predicted_labels)
    for i in range(2):
        mask = (predicted_labels == i)
        labels[mask] = mode(true_labels[mask])[0]
    return labels

adjusted_clusters = adjust_labels(y_encoded, clusters)
Kmeans_accuracy = accuracy_score(y_encoded, adjusted_clusters)

# 繪製分群結果
plt.scatter(data.loc[clusters == 0, 'x'], data.loc[clusters == 0, 'y'], c='blue', marker='o')
plt.scatter(data.loc[clusters == 1, 'x'], data.loc[clusters == 1, 'y'], c='red', marker='+')
plt.title('K-means Clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(f"K-means 用時: {Kmeans_elapsed_time} seconds")
print(f"K-means SSE: {Kmeans_sse}")
print(f"K-means 平均entropy: {Kmeans_mean_entropy}")
print(f"K-means Accuracy: {Kmeans_accuracy}")

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist

hierarchical_start_time = time.time()

Z = linkage(X, method='ward')

max_d = 7.5  #固定閾值(本次實驗使用7.5作為固定閾值)
hierarchical_clusters = fcluster(Z, max_d, criterion='distance')

hierarchical_end_time = time.time()
hierarchical_elapsed_time = hierarchical_end_time - hierarchical_start_time

#SSE
hierarchical_sse = sum(np.min(cdist(X, [np.mean(X[hierarchical_clusters == i], axis=0) for i in np.unique(hierarchical_clusters)], 'euclidean')**2, axis=1))

#entropy
hierarchical_entropy = []
for i in np.unique(hierarchical_clusters):
    cluster_labels = data[hierarchical_clusters == i]['class']
    hierarchical_entropy.append(entropy(cluster_labels.value_counts(normalize=True), base=2))
hierarchical_mean_entropy = np.mean(hierarchical_entropy)

#Accuracy
hierarchical_adjusted_clusters = adjust_labels(y_encoded, hierarchical_clusters)
hierarchical_accuracy = accuracy_score(y_encoded, hierarchical_adjusted_clusters)

plt.scatter(data.loc[hierarchical_clusters == 1, 'x'], data.loc[hierarchical_clusters == 1, 'y'], c='blue', marker='o')
plt.scatter(data.loc[hierarchical_clusters == 2, 'x'], data.loc[hierarchical_clusters == 2, 'y'], c='red', marker='+')
plt.title('Hierarchical Clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(f"Hierarchical Clustering 用時: {hierarchical_elapsed_time} seconds")
print(f"Hierarchical Clustering SSE: {hierarchical_sse}")
print(f"Hierarchical Clustering 平均entropy: {hierarchical_mean_entropy}")
print(f"Hierarchical Clustering Accuracy: {hierarchical_accuracy}")
