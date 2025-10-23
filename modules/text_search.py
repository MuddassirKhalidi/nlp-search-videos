import torch
import numpy as np
import os
import cv2
from transformers import CLIPProcessor, CLIPModel
from modules.chromadb_manager import ChromaDBManager

def search_frames_by_text(text_query: str, n_results: int = 5):
    """
    Search video frames using text query
    
    Args:
        text_query: Natural language description (e.g., "person cutting vegetables")
        n_results: Number of results to return
        
    Returns:
        dict: Search results with frame IDs, distances, and metadata
    """
    # Initialize CLIP model
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Initialize ChromaDB
    db_manager = ChromaDBManager()
    
    # Convert text to embedding
    inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    text_embedding = text_features.numpy().flatten()
    
    # Search ChromaDB
    results = db_manager.search_similar_frames(text_embedding, n_results)
    
    return results

def save_matched_frames(text_query: str, n_results: int = 10):
    """
    Search video frames using text query and save them to matched_imgs directory
    
    Args:
        text_query: Natural language description (e.g., "person cutting vegetables")
        n_results: Number of results to return
        
    Returns:
        dict: Search results with frame IDs, distances, and metadata
    """
    # Search for frames
    results = search_frames_by_text(text_query, n_results)
    
    if not results or not results['ids'] or not results['ids'][0]:
        print("No results found to save")
        return results
    
    # Create directory name from query (replace spaces with underscores)
    query_dir = text_query.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_dir = os.path.join("matched_imgs", query_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {len(results['ids'][0])} frames to: {output_dir}")
    
    # Save each matched frame
    for i, (frame_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
        metadata = results['metadatas'][0][i]
        
        try:
            # Extract frame from video
            video_path = metadata['video_path']
            frame_sample = metadata['frame_sample']
            
            cap = cv2.VideoCapture(video_path)
            cap.set(1, frame_sample)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Create filename with similarity score
                similarity = 1 - distance
                filename = f"{i+1:02d}_{frame_id}_similarity_{similarity:.3f}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Save frame as image
                cv2.imwrite(filepath, frame)
                print(f"  Saved: {filename}")
            else:
                print(f"  Failed to extract frame {frame_id}")
                
        except Exception as e:
            print(f"  Error saving frame {frame_id}: {e}")
    
    print(f"✅ Saved frames to: {output_dir}")
    return results

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "two men talking"
    
    print(f"🔍 Searching for: '{query}'")
    results = save_matched_frames(query, n_results=5)
    
    if results and results['ids']:
        print(f"\n📊 Found {len(results['ids'][0])} results:")
        for i, (frame_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
            metadata = results['metadatas'][0][i]
            print(f"{i+1}. {frame_id}")
            print(f"   Video: {metadata['video_name']}")
            print(f"   Scene: {metadata['scene_idx']}, Frame: {metadata['frame_idx']}")
            print(f"   Similarity: {1-distance:.4f}")
    else:
        print("No results found")