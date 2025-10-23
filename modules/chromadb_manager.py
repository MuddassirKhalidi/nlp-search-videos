import chromadb
import os
from typing import List, Dict, Any, Optional

class ChromaDBManager:
    """Manages ChromaDB operations for video embeddings"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "video_embeddings"):
        """
        Initialize ChromaDB manager
        
        Args:
            db_path: Path to store ChromaDB data
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            print(f"Found existing collection '{self.collection_name}' with {collection.count()} embeddings")
            return collection
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "CLIP embeddings for video frames"}
            )
            print(f"Created new collection '{self.collection_name}'")
            return collection
    
    def save_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """
        Save embeddings to ChromaDB
        
        Args:
            embeddings_data: List of dictionaries containing 'id', 'embedding', and 'metadata'
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not embeddings_data:
            print("No embeddings to save - skipping ChromaDB update")
            return False
        
        try:
            ids = [item['id'] for item in embeddings_data]
            embeddings = [item['embedding'].tolist() for item in embeddings_data]
            metadatas = [item['metadata'] for item in embeddings_data]
            
            # Check current count
            existing_count = self.collection.count()
            print(f"Collection currently has {existing_count} embeddings")
            
            # Add new embeddings (will skip duplicates based on ID)
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            new_count = self.collection.count()
            print(f"Added {len(embeddings_data)} embeddings. Collection now has {new_count} total embeddings")
            return True
            
        except Exception as e:
            print(f"Error saving embeddings: {e}")
            return False
    
    def search_similar_frames(self, query_embedding, n_results: int = 5) -> Optional[Dict]:
        """
        Search for similar frames using ChromaDB
        
        Args:
            query_embedding: The embedding vector to search for
            n_results: Number of similar results to return
            
        Returns:
            Dict with search results or None if error
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_embeddings": count,
                "db_path": self.db_path
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def search_by_metadata(self, where_clause: Dict[str, Any], n_results: int = 10) -> Optional[Dict]:
        """
        Search embeddings by metadata filters
        
        Args:
            where_clause: Dictionary with metadata filters (e.g., {"scene_idx": 0})
            n_results: Number of results to return
            
        Returns:
            Dict with search results or None if error
        """
        try:
            results = self.collection.query(
                where=where_clause,
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error searching by metadata: {e}")
            return None
    
    def delete_embeddings(self, ids: List[str]) -> bool:
        """
        Delete specific embeddings by IDs
        
        Args:
            ids: List of IDs to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=ids)
            print(f"Deleted {len(ids)} embeddings")
            return True
        except Exception as e:
            print(f"Error deleting embeddings: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all embeddings from the collection"""
        try:
            # Get all IDs first
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                print(f"Cleared all {len(all_data['ids'])} embeddings from collection")
            else:
                print("Collection is already empty")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def get_embeddings_by_video(self, video_name: str) -> Optional[Dict]:
        """
        Get all embeddings for a specific video
        
        Args:
            video_name: Name of the video file
            
        Returns:
            Dict with embeddings data or None if error
        """
        try:
            results = self.collection.query(
                where={"video_name": video_name},
                n_results=1000  # Large number to get all results
            )
            return results
        except Exception as e:
            print(f"Error getting embeddings by video: {e}")
            return None

# Example usage and testing functions
def test_chromadb_manager():
    """Test the ChromaDB manager functionality"""
    manager = ChromaDBManager()
    
    # Get collection info
    info = manager.get_collection_info()
    print("Collection Info:", info)
    
    # Example search by metadata
    pepper_results = manager.search_by_metadata({"video_name": "cutting_pepper.mp4"})
    if pepper_results and pepper_results['ids']:
        print(f"Found {len(pepper_results['ids'])} embeddings for cutting_pepper.mp4")
    else:
        print("No embeddings found for cutting_pepper.mp4")

if __name__ == "__main__":
    test_chromadb_manager()
