import numpy as np
from typing import List, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModel

class WordEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the word embedding model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
    def embed_words(self, words: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Generate embeddings for a list of words.
        
        Args:
            words: List of words to embed
            
        Returns:
            Tuple of (embeddings array, mapping of indices to original words)
        """
        # Create word to index mapping
        word_to_idx = {word: idx for idx, word in enumerate(words)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        embeddings = []
        with torch.no_grad():
            for word in words:
                # Tokenize and get model outputs
                inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True)
                outputs = self.model(**inputs)
                
                # Use mean pooling to get single vector per word
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                embeddings.append(embedding.numpy()[0])
        
        return np.array(embeddings), idx_to_word
