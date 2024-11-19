# Approximate Furthest Neighbors with Text Embeddings

An implementation of Approximate Furthest Neighbors algorithm optimized for text embeddings, with visualization tools and example datasets.

## Features

- Efficient AFN implementation supporting both Euclidean and Cosine distance metrics
- Text embedding support for both words and sentences using Transformer models
- Sophisticated pivot-based sampling strategy
- Interactive visualizations of embedding spaces
- Example datasets including words, sentences, and mixed text
- Built-in analysis tools for finding furthest pairs and neighbors

## Installation

```bash
pip install transformers torch sentence-transformers matplotlib seaborn scikit-learn
```

## Quick Start

```python
from word_embedder import TextEmbedder
from afn import ApproximateFurthestNeighbors, DistanceMetric
from visualizer import AFNVisualizer

# Generate embeddings
embedder = TextEmbedder()
texts = ["first text", "second text", "third text"]
embeddings, idx_to_text = embedder.embed_texts(texts)

# Initialize AFN
afn = ApproximateFurthestNeighbors(
    embeddings,
    metric=DistanceMetric.COSINE,
    num_pivots=3,
    points_per_pivot=2
)

# Visualize
visualizer = AFNVisualizer(afn, idx_to_text)
visualizer.plot_pivot_structure_2d()
```

## Structure

- `word_embedder.py`: Text embedding generation using transformer models
- `afn.py`: Core Approximate Furthest Neighbors implementation
- `visualizer.py`: Visualization tools for embedding spaces
- `example_usage.py`: Example usage and testing scripts

## Example Usage

```python
from example_usage import test_embeddings

# Test with different data types
test_embeddings("words")  # Single words
test_embeddings("sentences")  # Full sentences
test_embeddings("mixed")  # Combination of both
```

## License

MIT

## Citation

If you use this code in your research, please cite:

```bibtex
@software{afn_text_embeddings,
  title = {Approximate Furthest Neighbors with Text Embeddings},
  year = {2024},
  url = {https://github.com/ryoshu/afn}
}
```