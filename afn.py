import numpy as np
from typing import List, Tuple
from enum import Enum
from dataclasses import dataclass
import random
from collections import defaultdict

class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"

@dataclass
class Pivot:
    index: int
    point: np.ndarray
    neighbors: List[int]
    
class ApproximateFurthestNeighbors:
    def __init__(
        self, 
        points: np.ndarray, 
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        num_pivots: int = 10,
        points_per_pivot: int = 50
    ):
        """
        Initialize the AFN algorithm with sophisticated sampling.
        
        Args:
            points: numpy array of shape (n_points, n_dimensions)
            metric: distance metric to use (euclidean or cosine)
            num_pivots: number of pivot points for sampling strategy
            points_per_pivot: number of points to associate with each pivot
        """
        self.points = points
        self.metric = metric
        self.num_pivots = min(num_pivots, len(points))
        self.points_per_pivot = points_per_pivot
        
        # Normalize points for cosine similarity if needed
        if self.metric == DistanceMetric.COSINE:
            self.normalized_points = self._normalize_points(points)
        
        # Initialize pivots and their neighborhoods
        self.pivots = self._initialize_pivots()
        
    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """Normalize points to unit length for cosine similarity."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Prevent division by zero
        return points / norms
    
    def _initialize_pivots(self) -> List[Pivot]:
        """Initialize pivot points using maxmin sampling."""
        pivots = []
        
        # Choose first pivot randomly
        first_pivot_idx = random.randrange(len(self.points))
        pivots.append(Pivot(
            index=first_pivot_idx,
            point=self.points[first_pivot_idx],
            neighbors=[]
        ))
        
        # Choose remaining pivots using maxmin strategy
        remaining_indices = set(range(len(self.points))) - {first_pivot_idx}
        
        while len(pivots) < self.num_pivots and remaining_indices:
            # Find point maximizing minimum distance to existing pivots
            max_min_dist = float('-inf')
            next_pivot_idx = None
            
            for idx in remaining_indices:
                min_dist = float('inf')
                for pivot in pivots:
                    dist = self._distance(self.points[idx], pivot.point)
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    next_pivot_idx = idx
            
            if next_pivot_idx is not None:
                pivots.append(Pivot(
                    index=next_pivot_idx,
                    point=self.points[next_pivot_idx],
                    neighbors=[]
                ))
                remaining_indices.remove(next_pivot_idx)
        
        # Assign points to nearest pivots
        point_assignments = defaultdict(list)
        for idx in range(len(self.points)):
            if idx in {p.index for p in pivots}:
                continue
                
            # Find nearest pivot
            nearest_pivot_idx = min(
                range(len(pivots)),
                key=lambda i: self._distance(self.points[idx], pivots[i].point)
            )
            point_assignments[nearest_pivot_idx].append(idx)
        
        # Sample points_per_pivot points for each pivot
        for pivot_idx, assigned_points in point_assignments.items():
            if assigned_points:
                sample_size = min(self.points_per_pivot, len(assigned_points))
                pivots[pivot_idx].neighbors = random.sample(assigned_points, sample_size)
        
        return pivots
    
    def _euclidean_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x - y) ** 2))
    
    def _cosine_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate cosine distance (1 - cosine similarity) between two points."""
        if self.metric == DistanceMetric.COSINE:
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)
        return 1 - np.dot(x, y)
    
    def _distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance using the selected metric."""
        if self.metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_distance(x, y)
        else:
            return self._cosine_distance(x, y)
    
    def find_furthest_neighbors(self, query: np.ndarray, k: int = 1) -> Tuple[List[int], List[float]]:
        """Find k approximate furthest neighbors for the query point."""
        if k > len(self.points):
            raise ValueError("k cannot be larger than the number of points")
        
        # Normalize query for cosine similarity if needed
        if self.metric == DistanceMetric.COSINE:
            query = query / np.linalg.norm(query)
        
        # Find the nearest and furthest pivots to the query
        pivot_distances = [(self._distance(query, p.point), p) for p in self.pivots]
        pivot_distances.sort(key=lambda x: x[0])
        
        # Prioritize checking points around the furthest pivots
        candidate_points = set()
        for _, pivot in reversed(pivot_distances[:3]):  # Check top 3 furthest pivots
            candidate_points.update(pivot.neighbors)
            candidate_points.add(pivot.index)
        
        # Calculate distances to all candidate points
        distances = []
        for idx in candidate_points:
            dist = self._distance(query, self.points[idx])
            distances.append((dist, idx))
        
        # Sort by distance in descending order and get top k
        distances.sort(reverse=True)
        
        # Separate distances and indices
        furthest_distances = [d for d, _ in distances[:k]]
        furthest_indices = [i for _, i in distances[:k]]
        
        return furthest_indices, furthest_distances
    
    def find_furthest_pairs(self, num_pairs: int) -> List[Tuple[int, int, float]]:
        """Find approximate furthest pairs of points in the dataset."""
        pairs = []
        seen_pairs = set()
        
        # Use pivots to guide the search
        for pivot in self.pivots:
            # Find furthest neighbors for the pivot point
            far_indices, far_distances = self.find_furthest_neighbors(self.points[pivot.index], k=3)
            
            for far_idx, far_dist in zip(far_indices, far_distances):
                pair = tuple(sorted([pivot.index, far_idx]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    pairs.append((pair[0], pair[1], far_dist))
                    
                # Also check points in the neighborhood of the furthest point
                for neighbor_idx in self.pivots[0].neighbors:  # Use first pivot's neighbors
                    pair = tuple(sorted([far_idx, neighbor_idx]))
                    if pair not in seen_pairs:
                        dist = self._distance(self.points[far_idx], self.points[neighbor_idx])
                        seen_pairs.add(pair)
                        pairs.append((pair[0], pair[1], dist))
        
        # Sort pairs by distance and return top num_pairs
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:num_pairs]
