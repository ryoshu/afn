from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
from word_embedder import TextEmbedder
from afn import ApproximateFurthestNeighbors, DistanceMetric
import uvicorn

app = FastAPI(
    title="AFN Text API",
    description="API for finding furthest neighbors in text embedding space",
    version="1.0.0"
)

# Initialize global components
embedder = TextEmbedder()
current_afn = None
current_idx_to_text = None

class TextInput(BaseModel):
    texts: List[str]
    num_pivots: Optional[int] = 6
    points_per_pivot: Optional[int] = 5

class QueryInput(BaseModel):
    text: str
    k: Optional[int] = 5

class FurthestPairsInput(BaseModel):
    num_pairs: Optional[int] = 5

class TextPair(BaseModel):
    text1: str
    text2: str
    distance: float

class FurthestNeighbor(BaseModel):
    text: str
    distance: float

@app.post("/initialize", response_model=Dict[str, str])
async def initialize_afn(input_data: TextInput):
    """Initialize the AFN with a set of texts."""
    global current_afn, current_idx_to_text
    
    try:
        # Generate embeddings
        embeddings, idx_to_text = embedder.embed_texts(input_data.texts)
        
        # Initialize AFN
        afn = ApproximateFurthestNeighbors(
            embeddings,
            metric=DistanceMetric.COSINE,
            num_pivots=input_data.num_pivots,
            points_per_pivot=input_data.points_per_pivot
        )
        
        current_afn = afn
        current_idx_to_text = idx_to_text
        
        return {"status": "success", "message": f"Initialized AFN with {len(input_data.texts)} texts"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=List[FurthestNeighbor])
async def find_furthest(query: QueryInput):
    """Find the furthest texts from a query text."""
    if current_afn is None:
        raise HTTPException(status_code=400, detail="AFN not initialized. Call /initialize first.")
    
    try:
        # Generate embedding for query
        query_embedding = embedder.embed_texts([query.text])[0][0]
        
        # Find furthest neighbors
        indices, distances = current_afn.find_furthest_neighbors(query_embedding, k=query.k)
        
        # Format results
        results = []
        for idx, dist in zip(indices, distances):
            results.append(FurthestNeighbor(
                text=current_idx_to_text[idx],
                distance=float(dist)
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/furthest_pairs", response_model=List[TextPair])
async def find_furthest_pairs(input_data: FurthestPairsInput):
    """Find the furthest pairs of texts in the dataset."""
    if current_afn is None:
        raise HTTPException(status_code=400, detail="AFN not initialized. Call /initialize first.")
    
    try:
        # Find furthest pairs
        pairs = current_afn.find_furthest_pairs(num_pairs=input_data.num_pairs)
        
        # Format results
        results = []
        for idx1, idx2, dist in pairs:
            results.append(TextPair(
                text1=current_idx_to_text[idx1],
                text2=current_idx_to_text[idx2],
                distance=float(dist)
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get the current status of the AFN system."""
    return {
        "initialized": current_afn is not None,
        "num_texts": len(current_idx_to_text) if current_idx_to_text else 0,
        "num_pivots": current_afn.num_pivots if current_afn else 0,
        "points_per_pivot": current_afn.points_per_pivot if current_afn else 0
    }

def run_server():
    """Run the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()