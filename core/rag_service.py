import json
import os

import chromadb
from chromadb.utils import embedding_functions

class RAGService:
    def __init__(self, data_dir="data/policies"):
        self.data_dir = data_dir
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collections = {}
        self._initialize_collections()

    def _initialize_collections(self):
        """Initialize collections for each country policy."""
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                country_code = filename.split("_")[0].lower()
                collection_name = f"{country_code}_policy"
                
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

    def query_policy(self, country_code: str, query: str, n_results: int = 3):
        """Query the relevant policy points for a country."""
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
