import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def elbow_method(data, max_k):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    plt.show()

def main():
    # Sample data
    volume_capacity_ratio = np.array([0.21, 0.67, 0.96, 1.07, 0.20, 0.66, 0.91, 1.07, 0.18, 0.60, 0.90, 1.05, 0.17, 0.62, 0.90, 1.04])
    delay_time = np.array([21.53, 29.48, 51.35, 79.39, 21.47, 28.82, 45.03, 78.16, 26.40, 33.40, 49.35, 81.26, 32.78, 41.45, 57.47, 85.04])
    queue_length = np.array([1.61, 6.69, 14.34, 21.97, 1.56, 6.41, 12.42, 21.65, 1.67, 6.82, 13.73, 23.08, 1.98, 8.69, 16.21, 24.75])

    # Combining data into one array
    data = np.column_stack((volume_capacity_ratio, delay_time, queue_length))

    # Determining the optimal number of clusters using the elbow method
    max_k = min(10, len(data))
    elbow_method(data, max_k)

    # Asking user to input the optimal number of clusters
    k = int(input("Enter the optimal number of clusters based on the elbow method: "))

    # Performing K-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_

    # Printing cluster centers and cluster labels
    print("Cluster Centers:")
    print(kmeans.cluster_centers_)
    print("\nCluster Labels:")
    print(labels)

if __name__ == "__main__":
    main()
