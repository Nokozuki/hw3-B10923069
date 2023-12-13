
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import entropy
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/data/notebook_files/資料探勘/sizes3 (with class label).csv')  # 請替換為實際的文件路徑

X = data[['x', 'y']]
y = data['class']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

Kmeans_start_time = time.time()

kmeans = KMeans(n_clusters=4, random_state=0)
Kmeans_clusters = kmeans.fit_predict(X)

Kmeans_end_time = time.time()
Kmeans_elapsed_time = Kmeans_end_time - Kmeans_start_time

#SSE
Kmeans_sse = kmeans.inertia_

#entropy
Kmeans_cluster_entropy = []
for i in np.unique(Kmeans_clusters):
    cluster_labels = data[Kmeans_clusters == i]['class']
    Kmeans_cluster_entropy.append(entropy(cluster_labels.value_counts(normalize=True), base=2))
Kmeans_mean_entropy = np.mean(Kmeans_cluster_entropy)

# 計算準確度
def Kmeans_adjust_labels(true_labels, predicted_labels):
    from scipy.stats import mode

    labels = np.zeros_like(predicted_labels)
    for i in range(4):
        mask = (predicted_labels == i)
        labels[mask] = mode(true_labels[mask])[0]
    return labels

Kmeans_adjusted_clusters = Kmeans_adjust_labels(y_encoded, Kmeans_clusters)
Kmeans_accuracy = accuracy_score(y_encoded, Kmeans_adjusted_clusters)

# 繪製分群結果
markers = ['1', '2', '3', '4']
for i in range(4):
    plt.scatter(data.loc[Kmeans_clusters == i, 'x'], data.loc[Kmeans_clusters == i, 'y'], marker=markers[i])
plt.title('K-means Clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 輸出結果
print(f"K-means 用時: {Kmeans_elapsed_time} seconds")
print(f"K-means SSE: {Kmeans_sse}")
print(f"K-means 平均Entropy: {Kmeans_mean_entropy}")
print(f"K-means Accuracy: {Kmeans_accuracy}")

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist

Hierarchical_start_time = time.time()

Z = linkage(X, method='ward')

Hierarchical_clusters = fcluster(Z, t=4, criterion='maxclust')

Hierarchical_end_time = time.time()
Hierarchical_elapsed_time = Hierarchical_end_time - Hierarchical_start_time

#SSE
Hierarchical_sse = sum(np.min(cdist(X, [np.mean(X[Hierarchical_clusters == i], axis=0) for i in np.unique(Hierarchical_clusters)], 'euclidean')**2, axis=1))

#entropy
Hierarchical_cluster_entropy = []
for i in np.unique(Hierarchical_clusters):
    cluster_labels = data[Hierarchical_clusters == i]['class']
    Hierarchical_cluster_entropy.append(entropy(cluster_labels.value_counts(normalize=True), base=2))
Hierarchical_mean_entropy = np.mean(Hierarchical_cluster_entropy)

#Accuracy
Hierarchical_adjusted_clusters = Kmeans_adjust_labels(y_encoded, Hierarchical_clusters)
Hierarchical_accuracy = accuracy_score(y_encoded, Hierarchical_adjusted_clusters)

markers = ['1', '2', '3', '4']
for i in range(1, 5):
    plt.scatter(data.loc[Hierarchical_clusters == i, 'x'], data.loc[Hierarchical_clusters == i, 'y'], marker=markers[i-1])
plt.title('Hierarchical Clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

print(f"Hierarchical 用時: {Hierarchical_elapsed_time} seconds")
print(f"Hierarchical SSE: {Hierarchical_sse}")
print(f"Hierarchical 平均Entropy: {Hierarchical_mean_entropy}")
print(f"Hierarchical Accuracy: {Hierarchical_accuracy}")