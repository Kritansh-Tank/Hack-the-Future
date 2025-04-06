"""
Embeddings Utility

This module provides utilities for generating and working with embeddings from Ollama models.
"""

import logging
import numpy as np
import requests
import json
import time
import pickle
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

import config

logger = logging.getLogger(__name__)

class EmbeddingUtility:
    """Utility for generating and working with embeddings from Ollama models."""
    
    def __init__(self, model=None, base_url=None, timeout=None, cache_dir=None):
        """Initialize the embedding utility.
        
        Args:
            model (str, optional): The Ollama embedding model to use
            base_url (str, optional): Base URL for the Ollama API
            timeout (int, optional): Timeout in seconds for API calls
            cache_dir (str, optional): Directory to cache embeddings
        """
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.model = model or config.OLLAMA_EMBEDDING_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT
        self.cache_dir = cache_dir or (config.BASE_DIR / "tools" / "embeddings_cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if Ollama is available
        self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama API is available and the embedding model is present."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Ollama API is available at {self.base_url}")
            
            # Check if the model is available
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            if not model_names:
                logger.warning("No models found in Ollama")
            elif self.model not in model_names:
                logger.warning(f"Embedding model {self.model} not found in available models: {model_names}")
                logger.warning(f"Attempting to pull the model {self.model}...")
                self._pull_model()
            else:
                logger.info(f"Using Ollama embedding model: {self.model}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API is not available: {str(e)}")
            logger.warning("Make sure Ollama is running on your system.")
            
    def _pull_model(self):
        """Pull the embedding model from Ollama if it doesn't exist."""
        try:
            logger.info(f"Pulling embedding model {self.model} from Ollama...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=300  # Longer timeout for model pulling
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled embedding model {self.model}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull embedding model {self.model}: {str(e)}")
    
    def get_embedding(self, text, use_cache=True):
        """Get embedding for a text.
        
        Args:
            text (str): Text to get embedding for
            use_cache (bool, optional): Whether to use cached embeddings
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = str(hash(text))
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        embedding = pickle.load(f)
                        logger.debug(f"Loaded embedding from cache: {cache_key}")
                        return embedding
                except Exception as e:
                    logger.warning(f"Error loading cached embedding: {str(e)}")
        
        # Generate embedding if not cached or cache loading failed
        embedding = self._generate_embedding(text)
        
        # Cache the embedding if enabled
        if use_cache and embedding is not None:
            cache_key = str(hash(text))
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
                    logger.debug(f"Cached embedding: {cache_key}")
            except Exception as e:
                logger.warning(f"Error caching embedding: {str(e)}")
        
        return embedding
    
    def _generate_embedding(self, text):
        """Generate embedding for text using Ollama.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            start_time = time.time()
            logger.debug(f"Generating embedding for text: {text[:100]}...")
            
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            if not embedding:
                logger.error("No embedding returned from Ollama")
                return None
            
            embedding_vector = np.array(embedding, dtype=np.float32)
            
            end_time = time.time()
            logger.debug(f"Generated embedding of shape {embedding_vector.shape} in {end_time - start_time:.2f} seconds")
            
            return embedding_vector
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def get_batch_embeddings(self, texts, use_cache=True):
        """Get embeddings for a batch of texts.
        
        Args:
            texts (list): List of texts to get embeddings for
            use_cache (bool, optional): Whether to use cached embeddings
            
        Returns:
            list: List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            embedding = self.get_embedding(text, use_cache=use_cache)
            embeddings.append(embedding)
        
        return embeddings
    
    def compute_similarity(self, text1, text2):
        """Compute cosine similarity between two texts.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Cosine similarity (-1 to 1, higher is more similar)
        """
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        if embedding1 is None or embedding2 is None:
            logger.error("Failed to compute embeddings for similarity")
            return -1
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    
    def semantic_search(self, query, documents, top_k=5):
        """Perform semantic search over a set of documents.
        
        Args:
            query (str): Query text
            documents (list): List of documents (strings or dicts with 'text' key)
            top_k (int, optional): Number of top results to return
            
        Returns:
            list: List of (index, score, document) tuples in descending order of similarity
        """
        query_embedding = self.get_embedding(query)
        
        if query_embedding is None:
            logger.error("Failed to compute query embedding for semantic search")
            return []
        
        results = []
        
        for i, doc in enumerate(documents):
            # Extract text from document if it's a dict
            doc_text = doc['text'] if isinstance(doc, dict) else doc
            
            # Get document embedding
            doc_embedding = self.get_embedding(doc_text)
            
            if doc_embedding is None:
                logger.warning(f"Failed to compute embedding for document {i}")
                continue
            
            # Compute similarity
            similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            
            # Add to results
            results.append((i, float(similarity), doc))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return results[:top_k]

# Example usage
if __name__ == "__main__":
    # Configure basic logging to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    embedding_util = EmbeddingUtility()
    
    # Test embedding
    text = "Software engineer with experience in Python, JavaScript, and cloud computing."
    embedding = embedding_util.get_embedding(text)
    print(f"\nEmbedding shape: {embedding.shape}")
    
    # Test similarity
    text1 = "Data scientist with machine learning and Python experience"
    text2 = "Python developer with machine learning expertise"
    similarity = embedding_util.compute_similarity(text1, text2)
    print(f"\nSimilarity between texts: {similarity:.4f}")
    
    # Test semantic search
    query = "Cloud computing expert"
    documents = [
        "Software engineer with cloud computing experience",
        "Data scientist specializing in machine learning",
        "Cloud solutions architect with AWS and Azure experience",
        "Frontend developer with React.js skills",
        "DevOps engineer experienced in cloud infrastructure"
    ]
    
    results = embedding_util.semantic_search(query, documents, top_k=3)
    print("\nSemantic search results:")
    for i, score, doc in results:
        print(f"  {score:.4f}: {doc}") 