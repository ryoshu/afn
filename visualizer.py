import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from afn import ApproximateFurthestNeighbors

class AFNVisualizer:
    def __init__(self, afn: ApproximateFurthestNeighbors, idx_to_word: Dict[int, str] = None):
        """Initialize visualizer for AFN with optional word labels."""
        self.afn = afn
        self.points = afn.points
        self.dims = self.points.shape[1]
        self.idx_to_word = idx_to_word
        
        # Create color palette for pivots
        self.colors = sns.color_palette("husl", n_colors=len(afn.pivots))
    
    def _reduce_dimensions(self, points: np.ndarray, target_dims: int = 2) -> np.ndarray:
        """Reduce dimensionality of points for visualization."""
        if points.shape[1] <= target_dims:
            return points
        pca = PCA(n_components=target_dims)
        return pca.fit_transform(points)
    
    def plot_pivot_structure_2d(self, 
                              show_neighbors: bool = True,
                              show_connections: bool = True,
                              show_labels: bool = True,
                              figsize: Tuple[int, int] = (12, 8)):
        """Visualize pivot structure in 2D with word labels."""
        points_2d = self._reduce_dimensions(self.points, target_dims=2)
        
        plt.figure(figsize=figsize)
        
        # Plot all points in light gray
        plt.scatter(points_2d[:, 0], points_2d[:, 1], 
                   c='lightgray', alpha=0.3, label='Non-pivot points')
        
        # Plot pivot neighborhoods
        for i, pivot in enumerate(self.afn.pivots):
            pivot_point = points_2d[pivot.index]
            
            # Plot pivot point
            plt.scatter(pivot_point[0], pivot_point[1], 
                       c=[self.colors[i]], s=100, label=f'Pivot {i}')
            
            # Add word label for pivot
            if show_labels and self.idx_to_word is not None:
                plt.annotate(self.idx_to_word[pivot.index], 
                           (pivot_point[0], pivot_point[1]),
                           xytext=(5, 5), textcoords='offset points')
            
            if show_neighbors and pivot.neighbors:
                # Plot neighbor points
                neighbor_points = points_2d[pivot.neighbors]
                plt.scatter(neighbor_points[:, 0], neighbor_points[:, 1],
                          c=[self.colors[i]], alpha=0.5, s=50)
                
                # Add word labels for neighbors
                if show_labels and self.idx_to_word is not None:
                    for idx, neighbor in zip(pivot.neighbors, neighbor_points):
                        plt.annotate(self.idx_to_word[idx],
                                   (neighbor[0], neighbor[1]),
                                   xytext=(5, 5), textcoords='offset points',
                                   alpha=0.7)
                
                if show_connections:
                    for neighbor in neighbor_points:
                        plt.plot([pivot_point[0], neighbor[0]],
                               [pivot_point[1], neighbor[1]],
                               c=self.colors[i], alpha=0.2)
        
        plt.title('Word Embedding Pivot Structure (2D)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
