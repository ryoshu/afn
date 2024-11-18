import numpy as np
from typing import List, Tuple, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModel

class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the text embedding model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Generate embeddings for a list of texts (words or sentences).
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            
        Returns:
            Tuple of (embeddings array, mapping of indices to original texts)
        """
        # Create text to index mapping
        text_to_idx = {text: idx for idx, text in enumerate(texts)}
        idx_to_text = {idx: text for text, idx in text_to_idx.items()}
        
        embeddings = []
        with torch.no_grad():
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize and get model outputs
                inputs = self.tokenizer(batch_texts, 
                                      return_tensors="pt", 
                                      padding=True, 
                                      truncation=True,
                                      max_length=512)
                
                outputs = self.model(**inputs)
                
                # Use mean pooling to get single vector per text
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).numpy()
                
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings), idx_to_text
    
    def generate_example_data(self, data_type: str = "sentences") -> List[str]:
        """
        Generate example data for testing.
        
        Args:
            data_type: Type of data to generate ("words", "sentences", or "mixed")
            
        Returns:
            List of example texts
        """
        if data_type == "words":
            return [
                "happy", "sad", "joy", "sorrow", "laugh", "cry",
                "good", "bad", "excellent", "terrible",
                "hot", "cold", "warm", "cool",
                "big", "small", "huge", "tiny",
                "fast", "slow", "quick", "sluggish",
                "bright", "dark", "light", "dim",
                "love", "hate", "like", "dislike"
            ]
        
        elif data_type == "sentences":
            return [
                # Positive sentences
                "I absolutely love this beautiful day!",
                "The concert was an amazing experience.",
                "She achieved her lifelong dream of becoming a doctor.",
                "The new restaurant exceeded all our expectations.",
                "The children's laughter filled the playground with joy.",
                
                # Negative sentences
                "The movie was a complete waste of time.",
                "I'm really disappointed with the service.",
                "The traffic made me late for the important meeting.",
                "The weather ruined our outdoor plans.",
                "The exam results were worse than expected.",
                
                # Neutral/Factual sentences
                "The sun rises in the east and sets in the west.",
                "Water boils at 100 degrees Celsius at sea level.",
                "The Earth revolves around the Sun.",
                "Humans need oxygen to survive.",
                "Paris is the capital of France.",
                
                # Questions
                "What time does the movie start?",
                "How do you solve this math problem?",
                "Where did you park the car?",
                "When is the next train arriving?",
                "Why does this keep happening?",
                
                # Commands/Requests
                "Please send me the report by tomorrow.",
                "Could you help me with this task?",
                "Don't forget to lock the door.",
                "Make sure to follow the instructions carefully.",
                "Let me know when you're available.",
                
                # Complex emotions
                "The bittersweet feeling of saying goodbye to old friends.",
                "Mixed emotions filled the room during the graduation ceremony.",
                "A sense of nostalgia washed over me as I looked at old photos.",
                "The anticipation of meeting someone new brought both excitement and anxiety.",
                "The victory felt hollow without sharing it with loved ones."
            ]
        
        else:  # mixed
            words = self.generate_example_data("words")
            sentences = self.generate_example_data("sentences")
            # Take a subset of both
            return words[:15] + sentences[:15]