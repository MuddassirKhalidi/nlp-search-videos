#!/usr/bin/env python3
"""
Main Video Processing Script

This script provides a clean interface for processing videos and saving
CLIP embeddings to ChromaDB. It can be used as a standalone script or
imported as a module.
"""

import os
import sys
from pathlib import Path
from modules.chromadb_manager import ChromaDBManager
from modules.clip_embeddings import process_video_embeddings
from modules.text_search import save_matched_frames

def process_video_to_chromadb(video_path: str, db_path: str = "./chroma_db", collection_name: str = "video_embeddings") -> dict:
    """
    Process a video file and save its CLIP embeddings to ChromaDB
    
    Args:
        video_path: Path to the video file
        db_path: Path to ChromaDB storage directory
        collection_name: Name of the ChromaDB collection
        
    Returns:
        dict: Results containing success status, embedding count, and any errors
    """
    result = {
        "success": False,
        "video_path": video_path,
        "embeddings_count": 0,
        "error": None,
        "collection_info": {}
    }
    
    try:
        # Validate video file
        if not os.path.exists(video_path):
            result["error"] = f"Video file not found: {video_path}"
            return result
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Full path: {video_path}")
        
        # Initialize ChromaDB manager
        db_manager = ChromaDBManager(db_path=db_path, collection_name=collection_name)
        
        # Process video embeddings
        print("Generating CLIP embeddings...")
        embeddings_data = process_video_embeddings(video_path)
        
        if not embeddings_data:
            result["error"] = "No embeddings generated from video"
            return result
        
        print(f"Generated {len(embeddings_data)} embeddings")
        
        # Save to ChromaDB
        print("Saving embeddings to ChromaDB...")
        success = db_manager.save_embeddings(embeddings_data)
        
        if not success:
            result["error"] = "Failed to save embeddings to ChromaDB"
            return result
        
        # Get collection info
        collection_info = db_manager.get_collection_info()
        
        result.update({
            "success": True,
            "embeddings_count": len(embeddings_data),
            "collection_info": collection_info
        })
        
        print(f"✅ Successfully processed video and saved {len(embeddings_data)} embeddings")
        print(f"📊 Collection now has {collection_info.get('total_embeddings', 0)} total embeddings")
        
        return result
        
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        print(f"❌ Error processing video: {e}")
        return result

def process_multiple_videos(video_paths: list, db_path: str = "./chroma_db", collection_name: str = "video_embeddings") -> list:
    """
    Process multiple video files and save their embeddings to ChromaDB
    
    Args:
        video_paths: List of video file paths
        db_path: Path to ChromaDB storage directory
        collection_name: Name of the ChromaDB collection
        
    Returns:
        list: List of results for each video processed
    """
    results = []
    
    print(f"Processing {len(video_paths)} videos...")
    print("=" * 50)
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n[{i}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")
        print("-" * 30)
        
        result = process_video_to_chromadb(video_path, db_path, collection_name)
        results.append(result)
        
        if result["success"]:
            print(f"✅ Success: {result['embeddings_count']} embeddings saved")
        else:
            print(f"❌ Failed: {result['error']}")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    total_embeddings = sum(r["embeddings_count"] for r in results)
    
    print("\n" + "=" * 50)
    print(f"📊 SUMMARY:")
    print(f"   Videos processed: {successful}/{len(video_paths)}")
    print(f"   Total embeddings: {total_embeddings}")
    
    if successful > 0:
        final_info = results[-1]["collection_info"]
        print(f"   Collection total: {final_info.get('total_embeddings', 0)} embeddings")
    
    return results

def get_videos_from_directory(directory_path: str, extensions: list = None) -> list:
    """
    Get all video files from a directory
    
    Args:
        directory_path: Path to directory containing videos
        extensions: List of video file extensions to include
        
    Returns:
        list: List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    video_paths = []
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory not found: {directory_path}")
        return video_paths
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            video_paths.append(str(file_path))
    
    return sorted(video_paths)

def search_videos_by_text(text_query: str, n_results: int = 10, save_images: bool = True):
    """
    Search video frames using text query
    
    Args:
        text_query: Natural language description (e.g., "person cutting vegetables")
        n_results: Number of results to return
        save_images: Whether to save matched frames to matched_imgs directory
        
    Returns:
        dict: Search results
    """
    try:
        print(f"🔍 Searching for: '{text_query}'")
        
        if save_images:
            # Use the save_matched_frames function which saves images
            results = save_matched_frames(text_query, n_results)
        else:
            # Just search without saving images
            from modules.text_search import search_frames_by_text
            results = search_frames_by_text(text_query, n_results)
            
            if results and results['ids']:
                print(f"\n📊 Found {len(results['ids'][0])} results:")
                for i, (frame_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    metadata = results['metadatas'][0][i]
                    print(f"{i+1}. {frame_id}")
                    print(f"   📹 Video: {metadata['video_name']}")
                    print(f"   🎬 Scene: {metadata['scene_idx']}, Frame: {metadata['frame_idx']}")
                    print(f"   📊 Similarity: {1-distance:.4f}")
            else:
                print("No results found")
        
        return results
        
    except Exception as e:
        print(f"❌ Error searching videos: {e}")
        return None

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <video_path>                    # Process single video")
        print("  python main.py <video_path1> <video_path2> ...  # Process multiple videos")
        print("  python main.py --directory <directory_path>     # Process all videos in directory")
        print("  python main.py --search 'text query'            # Search videos by text")
        print("  python main.py --search-no-save 'text query'    # Search without saving images")
        print("\nExamples:")
        print("  python main.py videos/sample.mp4")
        print("  python main.py videos/sample.mp4 videos/cat.mp4")
        print("  python main.py --directory videos/")
        print("  python main.py --search 'person cutting vegetables'")
        print("  python main.py --search 'kitchen scene'")
        return
    
    if sys.argv[1] == "--directory":
        if len(sys.argv) < 3:
            print("Error: Please provide directory path")
            return
        
        directory_path = sys.argv[2]
        video_paths = get_videos_from_directory(directory_path)
        
        if not video_paths:
            print(f"No video files found in directory: {directory_path}")
            return
        
        print(f"Found {len(video_paths)} video files in {directory_path}")
        process_multiple_videos(video_paths)
    
    elif sys.argv[1] == "--search":
        if len(sys.argv) < 3:
            print("Error: Please provide search query")
            print("Example: python main.py --search 'person cutting vegetables'")
            return
        
        query = " ".join(sys.argv[2:])
        search_videos_by_text(query, save_images=True)
    
    elif sys.argv[1] == "--search-no-save":
        if len(sys.argv) < 3:
            print("Error: Please provide search query")
            print("Example: python main.py --search-no-save 'person cutting vegetables'")
            return
        
        query = " ".join(sys.argv[2:])
        search_videos_by_text(query, save_images=False)
    
    else:
        # Process individual video files
        video_paths = sys.argv[1:]
        process_multiple_videos(video_paths)

if __name__ == "__main__":
    main()
