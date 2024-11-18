from word_embedder import WordEmbedder
from afn import ApproximateFurthestNeighbors, DistanceMetric
from visualizer import AFNVisualizer

def test_word_embeddings():
    # Sample word list
    words = [
        "happy", "sad", "joy", "sorrow", "laugh", "cry",
        "good", "bad", "excellent", "terrible",
        "hot", "cold", "warm", "cool",
        "big", "small", "huge", "tiny",
        "fast", "slow", "quick", "sluggish",
        "bright", "dark", "light", "dim",
        "love", "hate", "like", "dislike"
    ]
    
    # Initialize embedder and generate embeddings
    print("Generating word embeddings...")
    embedder = WordEmbedder()
    embeddings, idx_to_word = embedder.embed_words(words)
    
    # Initialize AFN
    print("Initializing AFN...")
    afn = ApproximateFurthestNeighbors(
        embeddings,
        metric=DistanceMetric.COSINE,  # Cosine similarity is often better for word embeddings
        num_pivots=6,
        points_per_pivot=5
    )
    
    # Create visualizer
    visualizer = AFNVisualizer(afn, idx_to_word)
    
    # Generate visualizations
    print("Generating 2D visualization...")
    visualizer.plot_pivot_structure_2d(show_labels=True)
    
    # Find some example furthest pairs
    print("\nFinding furthest word pairs...")
    pairs = afn.find_furthest_pairs(num_pairs=5)
    print("\nFurthest word pairs:")
    for idx1, idx2, dist in pairs:
        print(f"'{idx_to_word[idx1]}' <-> '{idx_to_word[idx2]}': distance = {dist:.3f}")
    
    # Find furthest words from a query
    query_word = "happy"
    print(f"\nFinding words furthest from '{query_word}'...")
    query_embedding = embedder.embed_words([query_word])[0][0]
    indices, distances = afn.find_furthest_neighbors(query_embedding, k=5)
    print(f"\nWords furthest from '{query_word}':")
    for idx, dist in zip(indices, distances):
        print(f"'{idx_to_word[idx]}': distance = {dist:.3f}")

if __name__ == "__main__":
    test_word_embeddings()