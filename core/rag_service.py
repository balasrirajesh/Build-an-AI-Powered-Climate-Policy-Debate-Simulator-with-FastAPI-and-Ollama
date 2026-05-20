import json
import os
import threading

import chromadb
from chromadb.utils import embedding_functions

class RAGService:
    def __init__(self, data_dir="data/policies"):
        self.data_dir = data_dir
        self.client = None
        self.embedding_fn = None
        self.collections = {}
        self._initialized = False
        self._lock = threading.Lock()

    def ensure_initialized(self):
        """Initialize ChromaDB and collections lazily in a thread-safe manner."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".json"):
                    country_code = filename.split("_")[0].lower()
                    collection_name = f"{country_code}_policy"
                    
                    # Try to retrieve existing collection first
                    try:
                        collection = self.client.get_collection(
                            name=collection_name,
                            embedding_function=self.embedding_fn
                        )
                        if collection.count() > 0:
                            self.collections[country_code] = collection
                            continue
                    except Exception:
                        pass
                    
                    # Delete existing collection to refresh data if needed
                    try:
                        self.client.delete_collection(name=collection_name)
                    except Exception:
                        pass
                    
                    collection = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_fn
                    )
                    
                    with open(os.path.join(self.data_dir, filename), "r") as f:
                        data = json.load(f)
                        
                    documents = []
                    ids = []
                    
                    for i, pos in enumerate(data.get("key_positions", [])):
                        documents.append(pos)
                        ids.append(f"pos_{i}")
                        
                    for i, red in enumerate(data.get("red_lines", [])):
                        documents.append(red)
                        ids.append(f"red_{i}")
                        
                    if documents:
                        collection.add(
                            documents=documents,
                            ids=ids
                        )
                    
                    self.collections[country_code] = collection
            
            self._initialized = True

    def query_policy(self, country_code: str, query: str, n_results: int = 3):
        """Query the relevant policy points for a country."""
        self.ensure_initialized()
        country_code = country_code.lower()
        if country_code not in self.collections:
            return []
            
        collection = self.collections[country_code]
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return results["documents"][0] if results["documents"] else []

# Global instance
rag_service = RAGService()

