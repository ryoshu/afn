from typing import List, Dict, Tuple
import numpy as np
from word_embedder import TextEmbedder
from afn import ApproximateFurthestNeighbors, DistanceMetric
import random

class CreativeSpaceFinder:
    def __init__(self, embedder: TextEmbedder = None):
        """Initialize the creative space finder."""
        self.embedder = embedder or TextEmbedder()
        self.afn = None
        self.idx_to_text = None
        self.embeddings = None
    
    # [Previous methods remain the same until generate_creative_prompt]
    
    def generate_creative_prompt(self, concept: str) -> str:
        """
        Generate a creative prompt based on conceptual opposites and tensions.
        Now with better error handling and dynamic content based on available results.
        """
        # Get opposites and tensions with error handling
        try:
            opposites = self.find_conceptual_opposites(concept, num_opposites=3)
        except:
            opposites = []
        
        try:
            tensions = self.find_creative_tension(concept, tension_radius=0.3)  # Reduced radius for more results
        except:
            tensions = []
        
        # Build prompt dynamically based on available results
        prompt_parts = [f"Let's explore the concept of '{concept}' creatively:\n"]
        
        # Add opposites section if available
        if opposites:
            prompt_parts.append("1. Conceptual Opposites:")
            for opp, dist in opposites[:2]:
                prompt_parts.append(f"   - What if '{concept}' had qualities of '{opp}'?")
        
        # Add tensions section if available
        if tensions:
            prompt_parts.append("\n2. Creative Tensions:")
            for t1, t2, _ in tensions[:2]:
                prompt_parts.append(f"   - How might '{t1}' and '{t2}' interact with '{concept}'?")
        
        # Add exploration questions
        prompt_parts.append("\n3. Creative Exploration:")
        prompt_parts.append(f"   - What unexpected aspects of '{concept}' emerge when viewed from these angles?")
        prompt_parts.append(f"   - How could you combine these different perspectives of '{concept}'?")
        prompt_parts.append(f"   - What exists in the spaces between these ideas?")
        
        if opposites:
            prompt_parts.append(f"   - How might '{opposites[0][0]}' help redefine '{concept}'?")
        
        # Add synthesis prompt
        prompt_parts.append("\n4. Synthesis:")
        prompt_parts.append("   Consider these questions:")
        prompt_parts.append(f"   - What new forms of '{concept}' become possible?")
        prompt_parts.append("   - What assumptions are you challenging?")
        prompt_parts.append("   - What novel combinations could emerge?")
        
        return "\n".join(prompt_parts)

    def find_creative_tension(self, concept: str, tension_radius: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Find pairs of concepts that create interesting tension with the given concept.
        Now with improved sampling and fallback options.
        """
        # Get embedding for input concept
        concept_emb = self.embedder.embed_texts([concept])[0][0]
        
        # Find concepts within a certain radius - now with adaptive radius
        distances = np.linalg.norm(self.embeddings - concept_emb, axis=1)
        
        # Start with initial radius, but adapt if not enough results
        while tension_radius <= 1.0:
            candidate_indices = np.where((distances > tension_radius) & 
                                       (distances < tension_radius * 2))[0]
            
            if len(candidate_indices) >= 4:  # Need at least 4 candidates to form 2 pairs
                break
            
            tension_radius += 0.1
        
        # If still not enough candidates, return empty list
        if len(candidate_indices) < 4:
            return []
        
        # Find pairs of these concepts that are also moderately distant from each other
        tensions = []
        for i in range(len(candidate_indices)):
            for j in range(i + 1, len(candidate_indices)):
                idx1, idx2 = candidate_indices[i], candidate_indices[j]
                dist = np.linalg.norm(self.embeddings[idx1] - self.embeddings[idx2])
                if tension_radius < dist < tension_radius * 2:
                    tensions.append((self.idx_to_text[idx1], 
                                  self.idx_to_text[idx2], 
                                  float(dist)))
        
        # Ensure we have some results
        if not tensions:
            # Fallback: just return some pairs of candidates
            for i in range(min(2, len(candidate_indices))):
                for j in range(i + 1, min(3, len(candidate_indices))):
                    idx1, idx2 = candidate_indices[i], candidate_indices[j]
                    dist = np.linalg.norm(self.embeddings[idx1] - self.embeddings[idx2])
                    tensions.append((self.idx_to_text[idx1],
                                  self.idx_to_text[idx2],
                                  float(dist)))
        
        return sorted(tensions, key=lambda x: x[2], reverse=True)[:5]

    def initialize_creative_space(self, seed_concepts: List[str]):
        """Initialize the creative space with seed concepts and their variations."""
        try:
            # Generate embeddings for seed concepts
            embeddings, idx_to_text = self.embedder.embed_texts(list(set(seed_concepts)))
            self.embeddings = embeddings
            self.idx_to_text = idx_to_text
            
            # Initialize AFN with appropriate parameters for the dataset size
            self.afn = ApproximateFurthestNeighbors(
                embeddings,
                metric=DistanceMetric.COSINE,
                num_pivots=min(6, len(embeddings)),
                points_per_pivot=min(5, len(embeddings) // 2)
            )
            
            # Verify initialization
            if self.afn is None or self.embeddings is None or self.idx_to_text is None:
                raise ValueError("Initialization failed")
                
        except Exception as e:
            print(f"Error initializing creative space: {str(e)}")
            raise

    def find_conceptual_opposites(self, concept: str, num_opposites: int = 5) -> List[Tuple[str, float]]:
        """
        Find semantic opposites of a concept to spark new ideas.
        Rather than traditional antonyms, this finds concepts that are 
        semantically distant in the embedding space.
        
        Args:
            concept: The concept to find opposites for
            num_opposites: Number of opposites to return
            
        Returns:
            List of tuples containing (opposite_concept, distance)
        """
        try:
            # Get embedding for the concept
            query_embedding = self.embedder.embed_texts([concept])[0][0]
            
            # Check if AFN is initialized
            if self.afn is None:
                raise ValueError("AFN not initialized. Call initialize_creative_space first.")
            
            # Find furthest neighbors
            indices, distances = self.afn.find_furthest_neighbors(
                query_embedding, 
                k=min(num_opposites, len(self.embeddings))
            )
            
            # Convert to list of tuples with text and distance
            opposites = []
            for idx, dist in zip(indices, distances):
                if idx in self.idx_to_text:
                    opposites.append((self.idx_to_text[idx], float(dist)))
                
            return opposites
            
        except Exception as e:
            print(f"Error finding conceptual opposites: {str(e)}")
            return []

    def find_creative_midpoint(self, concept1: str, concept2: str, num_midpoints: int = 5) -> List[Tuple[str, float]]:
        """
        Find concepts that lie in the semantic space between two concepts.
        This can help identify novel combinations or hybrid ideas.
        
        Args:
            concept1: First concept
            concept2: Second concept
            num_midpoints: Number of midpoint concepts to return
            
        Returns:
            List of tuples containing (midpoint_concept, distance_from_midpoint)
        """
        try:
            # Get embeddings for both concepts
            emb1 = self.embedder.embed_texts([concept1])[0][0]
            emb2 = self.embedder.embed_texts([concept2])[0][0]
            
            # Calculate midpoint in embedding space
            midpoint = (emb1 + emb2) / 2
            
            # Normalize midpoint to match the embedding space
            if self.afn.metric == DistanceMetric.COSINE:
                midpoint = midpoint / np.linalg.norm(midpoint)
                
            # Find nearest concepts to midpoint
            distances = []
            for idx, embedding in enumerate(self.embeddings):
                if self.afn.metric == DistanceMetric.COSINE:
                    dist = 1 - np.dot(embedding, midpoint)
                else:
                    dist = np.linalg.norm(embedding - midpoint)
                distances.append((idx, dist))
            
            # Sort by distance and get top results
            distances.sort(key=lambda x: x[1])
            
            # Filter out the original concepts and format results
            midpoints = []
            seen_concepts = {concept1.lower(), concept2.lower()}
            
            for idx, dist in distances:
                concept = self.idx_to_text[idx]
                if concept.lower() not in seen_concepts:
                    midpoints.append((concept, float(dist)))
                    seen_concepts.add(concept.lower())
                    
                if len(midpoints) >= num_midpoints:
                    break
                    
            return midpoints
            
        except Exception as e:
            print(f"Error finding creative midpoints: {str(e)}")
            return []

    def find_midpoint_space(self, concept1: str, concept2: str, 
                           radius: float = 0.3, max_concepts: int = 10) -> List[Tuple[str, float, float]]:
        """
        Find concepts that exist in the general space around the midpoint,
        with their relative distances to both original concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            radius: Radius around midpoint to search
            max_concepts: Maximum number of concepts to return
            
        Returns:
            List of tuples containing (concept, distance_to_concept1, distance_to_concept2)
        """
        try:
            # Get embeddings for both concepts
            emb1 = self.embedder.embed_texts([concept1])[0][0]
            emb2 = self.embedder.embed_texts([concept2])[0][0]
            
            # Calculate midpoint
            midpoint = (emb1 + emb2) / 2
            if self.afn.metric == DistanceMetric.COSINE:
                midpoint = midpoint / np.linalg.norm(midpoint)
            
            # Find concepts in the space around the midpoint
            space_concepts = []
            seen_concepts = {concept1.lower(), concept2.lower()}
            
            for idx, embedding in enumerate(self.embeddings):
                concept = self.idx_to_text[idx]
                if concept.lower() in seen_concepts:
                    continue
                    
                # Calculate distances
                if self.afn.metric == DistanceMetric.COSINE:
                    dist_to_midpoint = 1 - np.dot(embedding, midpoint)
                    dist1 = 1 - np.dot(embedding, emb1)
                    dist2 = 1 - np.dot(embedding, emb2)
                else:
                    dist_to_midpoint = np.linalg.norm(embedding - midpoint)
                    dist1 = np.linalg.norm(embedding - emb1)
                    dist2 = np.linalg.norm(embedding - emb2)
                
                # Check if within radius of midpoint and distances are balanced
                if dist_to_midpoint <= radius and abs(dist1 - dist2) < radius:
                    space_concepts.append((concept, float(dist1), float(dist2)))
                    
            # Sort by balance of distances and closeness to midpoint
            space_concepts.sort(key=lambda x: abs(x[1] - x[2]) + (x[1] + x[2])/4)
            
            return space_concepts[:max_concepts]
            
        except Exception as e:
            print(f"Error exploring midpoint space: {str(e)}")
            return []

    def analyze_creative_space(self, concept1: str, concept2: str) -> dict:
        """
        Provide a comprehensive analysis of the creative space between two concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            
        Returns:
            Dictionary containing various analyses of the creative space
        """
        try:
            # Get direct midpoints
            midpoints = self.find_creative_midpoint(concept1, concept2)
            
            # Get broader space concepts
            space_concepts = self.find_midpoint_space(concept1, concept2)
            
            # Find concepts that contrast with both original concepts
            contrasts = []
            if len(self.embeddings) > 2:
                c1_opposites = set(x[0] for x in self.find_conceptual_opposites(concept1, 3))
                c2_opposites = set(x[0] for x in self.find_conceptual_opposites(concept2, 3))
                contrasts = list(c1_opposites.intersection(c2_opposites))
            
            return {
                'midpoints': midpoints,
                'space_concepts': space_concepts,
                'contrasts': contrasts,
                'original_concepts': (concept1, concept2)
            }
            
        except Exception as e:
            print(f"Error analyzing creative space: {str(e)}")
            return {
                'midpoints': [],
                'space_concepts': [],
                'contrasts': [],
                'original_concepts': (concept1, concept2)
            }