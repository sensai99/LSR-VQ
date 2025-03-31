import logging
import math
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def balanced_kmeans_clustering(embeddings: np.ndarray, k: int, tolerance: float = 0.1, max_iterations: int = 10) -> np.array:
    """
    Perform k-means clustering with post-processing to ensure roughly equal-sized clusters.
    
    Args:
        embeddings: numpy array of shape (N, d) containing N d-dimensional embeddings
        k: number of desired clusters
        tolerance: allowed deviation from ideal cluster size (as a fraction)
        max_iterations: maximum number of iterations for balancing clusters
    
    Returns:
        numpy array of cluster labels
    """
    # Add start time tracking
    start_time = time.time()
    
    # Reduce dimensionality to 128 using PCA
    if embeddings.shape[1] > 64:
        print(f"Reducing dimensionality from {embeddings.shape[1]} to 64 dimensions using PCA...")
        pca = PCA(n_components=64)
        embeddings = pca.fit_transform(embeddings)
        print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.3f}")

    # Initial k-means clustering with verbose logging and parallel processing
    kmeans = KMeans(
        n_clusters=k, 
        random_state=42,
        verbose=1  # Enable logging
    )
    labels = kmeans.fit_predict(embeddings)
    
    N = len(embeddings)
    target_size = N // k
    min_size = int(target_size * (1 - tolerance))
    max_size = int(target_size * (1 + tolerance))
    
    # Get cluster sizes and centers
    unique_labels, counts = np.unique(labels, return_counts=True)
    centers = kmeans.cluster_centers_
    
    # Add initial clustering info
    #print(f"\nInitial clustering:")
    print(f"Target size: {target_size} (min: {min_size}, max: {max_size})")
    #print(f"Initial cluster sizes: {dict(zip(unique_labels, counts))}")
    
    iteration = 0
    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}:")
        
        # Get current cluster info
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Calculate how many clusters we need for redistribution
        total_points = len(embeddings)
        target_size = total_points // k
        oversized_count = np.sum(counts > target_size)
        points_to_redistribute = np.sum(np.maximum(counts - target_size, 0))
        min_clusters_needed = k  # We always need at least k clusters
        
        # Check if we have enough clusters for redistribution
        if len(unique_labels) < min_clusters_needed:
            # Find the largest cluster to split
            largest_idx = np.argmax(counts)
            largest_cluster = unique_labels[largest_idx]
            
            # Split large cluster
            mask = labels == largest_cluster
            cluster_points = embeddings[mask]
            
            sub_kmeans = KMeans(n_clusters=2, random_state=42, verbose=1)
            sub_labels = sub_kmeans.fit_predict(cluster_points)
            
            # Reassign points
            new_labels = labels.copy()
            new_labels[mask] = np.where(sub_labels == 0, 
                                      largest_cluster, 
                                      len(unique_labels))
            
            # Update tracking variables
            labels = new_labels
            centers = np.vstack([centers, sub_kmeans.cluster_centers_[1]])
            print(f"Too few clusters ({len(unique_labels)}), splitting largest cluster {largest_cluster}")
            continue
        
        # Only merge clusters if we have more than enough clusters for redistribution
        if len(unique_labels) > k and min(counts) < min_size:
            smallest_idx = np.argmin(counts)
            smallest_cluster = unique_labels[smallest_idx]
            smallest_size = counts[smallest_idx]
            
            # Only merge if we'll still have enough clusters after merging
            if len(unique_labels) - 1 >= min_clusters_needed:
                # Find nearest cluster to merge with
                distances = np.linalg.norm(centers - centers[smallest_idx], axis=1)
                distances[smallest_idx] = np.inf
                nearest_idx = np.argmin(distances)
                nearest_cluster = unique_labels[nearest_idx]
                
                # Merge clusters
                labels[labels == smallest_cluster] = nearest_cluster
                mask = unique_labels != smallest_cluster
                unique_labels = unique_labels[mask]
                counts = counts[mask]
                centers = centers[mask]
                print(f"Merging undersized cluster {smallest_cluster} (size: {smallest_size}) with cluster {nearest_cluster}")
            else:
                print(f"Skipping merge to maintain minimum number of clusters")
                break
        else:
            break
            
        iteration += 1
    
    # Add redistribution phase
    print("\nStarting redistribution phase...")
    unique_labels, counts = np.unique(labels, return_counts=True)
    while True:
        # Find oversized and undersized clusters
        oversized = unique_labels[counts > max_size]
        
        # If no oversized clusters remain, we're done
        if len(oversized) == 0:
            break
            
        # Find clusters that can accept more points
        available_clusters = unique_labels[counts < max_size]
        
        # If no clusters can accept points, we can't balance further
        if len(available_clusters) == 0:
            print("Warning: Cannot redistribute further - no clusters available to accept points")
            break
            
        # For each oversized cluster, find points closest to available clusters
        for over_cluster in oversized:
            over_mask = labels == over_cluster
            over_points = embeddings[over_mask]
            
            # Calculate how many points to move
            excess = counts[unique_labels == over_cluster][0] - max_size
            points_to_move = min(excess, sum(max_size - counts[np.isin(unique_labels, available_clusters)]))
            
            if points_to_move == 0:
                continue
                
            # Calculate distances to available cluster centers
            available_centers = centers[np.isin(unique_labels, available_clusters)]
            distances = np.zeros((len(over_points), len(available_centers)))
            
            for i, point in enumerate(over_points):
                distances[i] = np.linalg.norm(point - available_centers, axis=1)
            
            # Find the points closest to available clusters
            closest_points_idx = np.argsort(distances.min(axis=1))[:points_to_move]
            points_to_reassign = np.where(over_mask)[0][closest_points_idx]
            
            # Reassign points to nearest available cluster
            for point_idx in points_to_reassign:
                point = embeddings[point_idx]
                distances_to_available = np.linalg.norm(point - available_centers, axis=1)
                target_cluster = available_clusters[np.argmin(distances_to_available)]
                labels[point_idx] = target_cluster
            
            # Update counts
            unique_labels, counts = np.unique(labels, return_counts=True)
            
        #print(f"After redistribution: {dict(zip(unique_labels, counts))}")
    
    # Add end time tracking and printing before return
    end_time = time.time()
    print(f"\nTotal clustering time: {end_time - start_time:.2f} seconds")
    return labels


if __name__ == "__main__":
    # # Create synthetic data with actual cluster structure but more noise
    # n_samples_per_cluster = 1000
    # noise_level = 1.0
    # centers = [
    #     [-4, -4],
    #     [4, -4],
    #     [-4, 4],
    #     [4, 4],
    #     [0, 0],    # Center cluster
    #     [-4, 0],   # Left cluster
    #     [4, 0],    # Right cluster
    #     [0, -4],   # Bottom cluster
    # ]
    
    # # Generate noisier clustered 2D data
    # embeddings = np.vstack([
    #     np.random.randn(n_samples_per_cluster, 2) * noise_level + center 
    #     for center in centers
    # ])
    
    # # Run clustering with more iterations
    # labels = balanced_kmeans_clustering(embeddings, k=8, max_iterations=10)
    
    # # Print cluster sizes
    # unique_labels, counts = np.unique(labels, return_counts=True)
    # for label, count in zip(unique_labels, counts):
    #     print(f"Cluster {label}: {count} points")
    
    # # Since data is already 2D, we don't need t-SNE
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
    #                      c=labels, cmap='tab10', alpha=0.6)
    # plt.colorbar(scatter, label='Cluster')
    # plt.title('Balanced K-means Clustering Visualization')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.savefig("balanced_kmeans_clustering.png")

    # Add high-dimensional embedding example
    print("\nTesting with high-dimensional embeddings:")
    n_samples = 100_000
    dim = 1024
    high_dim_embeddings = np.random.randn(n_samples, dim)
    n_clusters = math.floor(n_samples / 64)
    
    # Normalize the embeddings (common practice for embedding vectors)
    high_dim_embeddings = high_dim_embeddings / np.linalg.norm(high_dim_embeddings, axis=1)[:, np.newaxis]
    
    # Run clustering
    print(f"Clustering {n_samples} {dim}-dimensional vectors into {n_clusters} clusters...")
    high_dim_labels = balanced_kmeans_clustering(high_dim_embeddings, k=n_clusters)
    
    # Print first 10 cluster sizes
    unique_labels, counts = np.unique(high_dim_labels, return_counts=True)
    print("\nFinal cluster sizes for high-dimensional data:")
    for label, count in zip(unique_labels[:20], counts[:20]):
        print(f"Cluster {label}: {count} points")
    
    # # PCA visualization
    # pca = PCA(n_components=2)
    # high_dim_embeddings_2d = pca.fit_transform(high_dim_embeddings)
    # plt.figure(figsize=(10, 8))
    # scatter = plt.scatter(high_dim_embeddings_2d[:, 0], high_dim_embeddings_2d[:, 1], c=high_dim_labels, cmap='tab10', alpha=0.6)
    # plt.colorbar(scatter, label='Cluster')
    # plt.title('Balanced K-means Clustering Visualization')
    # plt.savefig("balanced_kmeans_clustering_pca.png")
