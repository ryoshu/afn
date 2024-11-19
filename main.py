from word_embedder import TextEmbedder
from afn import ApproximateFurthestNeighbors, DistanceMetric
from visualizer import AFNVisualizer

def test_embeddings(data_type: str = "sentences"):
    """
    Test the AFN system with different types of text data.
    
    Args:
        data_type: Type of data to use ("words", "sentences", or "mixed")
    """
    # Initialize embedder and get example data
    print(f"Initializing embedder and generating {data_type} examples...")
    embedder = TextEmbedder()
    texts = embedder.generate_example_data(data_type)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings, idx_to_text = embedder.embed_texts(texts)
    
    # Initialize AFN
    print("Initializing AFN...")
    afn = ApproximateFurthestNeighbors(
        embeddings,
        metric=DistanceMetric.COSINE,
        num_pivots=2,
        points_per_pivot=15
    )
    
    # Create visualizer
    visualizer = AFNVisualizer(afn, idx_to_text)
    
    # Generate visualization
    print("Generating 2D visualization...")
    visualizer.plot_pivot_structure_2d(show_labels=True)
    
    # Find furthest pairs
    print("\nFinding furthest pairs...")
    pairs = afn.find_furthest_pairs(num_pairs=5)
    print("\nFurthest text pairs:")
    for idx1, idx2, dist in pairs:
        print(f"\nPair with distance = {dist:.3f}")
        print(f"Text 1: \"{idx_to_text[idx1]}\"")
        print(f"Text 2: \"{idx_to_text[idx2]}\"")
    
    # Example query
    if data_type == "words":
        query_text = "happy"
    else:
        query_text = "I absolutely love this beautiful day!"
    
    print(f"\nFinding texts furthest from: \"{query_text}\"")
    query_embedding = embedder.embed_texts([query_text])[0][0]
    indices, distances = afn.find_furthest_neighbors(query_embedding, k=5)
    
    print("\nFurthest texts:")
    for idx, dist in zip(indices, distances):
        print(f"\nDistance = {dist:.3f}")
        print(f"Text: \"{idx_to_text[idx]}\"")

def main():
    # Test with different types of data
    print("Testing with words...")
    test_embeddings("words")
    
    print("\nTesting with sentences...")
    test_embeddings("sentences")
    
    print("\nTesting with mixed data...")
    test_embeddings("mixed")

if __name__ == "__main__":
    main()