#!/usr/bin/env python3
"""
ChromaDB Query Utility

This script provides utilities for querying and managing the ChromaDB
collection of video embeddings.
"""

import sys
import argparse
from chromadb_manager import ChromaDBManager

def search_similar_frames(manager: ChromaDBManager, query_frame_id: str, n_results: int = 5):
    """Search for frames similar to a specific frame"""
    try:
        # Get the query frame's embedding
        frame_data = manager.collection.get(ids=[query_frame_id])
        if not frame_data['ids']:
            print(f"Frame ID '{query_frame_id}' not found")
            return
        
        query_embedding = frame_data['embeddings'][0]
        
        # Search for similar frames
        results = manager.search_similar_frames(query_embedding, n_results)
        
        if results and results['ids']:
            print(f"Similar frames to '{query_frame_id}':")
            print("-" * 50)
            for i, (frame_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                metadata = results['metadatas'][0][i]
                print(f"{i+1}. {frame_id}")
                print(f"   Video: {metadata['video_name']}")
                print(f"   Scene: {metadata['scene_idx']}, Frame: {metadata['frame_idx']}")
                print(f"   Distance: {distance:.4f}")
                print()
        else:
            print("No similar frames found")
            
    except Exception as e:
        print(f"Error searching similar frames: {e}")

def list_videos(manager: ChromaDBManager):
    """List all videos in the collection"""
    try:
        all_data = manager.collection.get()
        
        if not all_data['ids']:
            print("No videos found in collection")
            return
        
        # Group by video name
        videos = {}
        for i, metadata in enumerate(all_data['metadatas']):
            video_name = metadata['video_name']
            if video_name not in videos:
                videos[video_name] = []
            videos[video_name].append({
                'id': all_data['ids'][i],
                'scene': metadata['scene_idx'],
                'frame': metadata['frame_idx']
            })
        
        print("Videos in collection:")
        print("=" * 50)
        for video_name, frames in videos.items():
            print(f"📹 {video_name}")
            print(f"   Frames: {len(frames)}")
            print(f"   Scenes: {len(set(f['scene'] for f in frames))}")
            print()

    except Exception as e:
        print(f"Error listing videos: {e}")

def list_scenes(manager: ChromaDBManager, video_name: str = None):
    """List scenes for a specific video or all videos"""
    try:
        if video_name:
            # Get frames for specific video
            results = manager.search_by_metadata({"video_name": video_name})
            if not results or not results['ids']:
                print(f"No frames found for video: {video_name}")
                return
            
            frames = []
            for i, metadata in enumerate(results['metadatas']):
                frames.append({
                    'id': results['ids'][i],
                    'scene': metadata['scene_idx'],
                    'frame': metadata['frame_idx']
                })
            
            print(f"Scenes in '{video_name}':")
            print("-" * 30)
            
        else:
            # Get all frames
            all_data = manager.collection.get()
            if not all_data['ids']:
                print("No frames found in collection")
                return
            
            frames = []
            for i, metadata in enumerate(all_data['metadatas']):
                frames.append({
                    'id': all_data['ids'][i],
                    'video': metadata['video_name'],
                    'scene': metadata['scene_idx'],
                    'frame': metadata['frame_idx']
                })
            
            print("All scenes in collection:")
            print("-" * 30)
        
        # Group by scene
        scenes = {}
        for frame in frames:
            scene_key = frame['scene']
            if video_name:
                scene_key = f"Scene {scene_key}"
            else:
                scene_key = f"{frame['video']} - Scene {scene_key}"
            
            if scene_key not in scenes:
                scenes[scene_key] = []
            scenes[scene_key].append(frame)
        
        for scene_name, scene_frames in scenes.items():
            print(f"🎬 {scene_name}")
            print(f"   Frames: {len(scene_frames)}")
            print(f"   Frame IDs: {', '.join([f['id'] for f in scene_frames[:3]])}")
            if len(scene_frames) > 3:
                print(f"   ... and {len(scene_frames) - 3} more")
            print()

    except Exception as e:
        print(f"Error listing scenes: {e}")

def main():
    parser = argparse.ArgumentParser(description="ChromaDB Query Utility")
    parser.add_argument("--db-path", default="./chroma_db", help="Path to ChromaDB directory")
    parser.add_argument("--collection", default="video_embeddings", help="Collection name")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List videos command
    subparsers.add_parser("list-videos", help="List all videos in collection")
    
    # List scenes command
    scenes_parser = subparsers.add_parser("list-scenes", help="List scenes")
    scenes_parser.add_argument("--video", help="Video name to list scenes for")
    
    # Search similar command
    search_parser = subparsers.add_parser("search-similar", help="Search for similar frames")
    search_parser.add_argument("frame_id", help="Frame ID to search for")
    search_parser.add_argument("--results", type=int, default=5, help="Number of results to return")
    
    # Collection info command
    subparsers.add_parser("info", help="Show collection information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize ChromaDB manager
    manager = ChromaDBManager(db_path=args.db_path, collection_name=args.collection)
    
    if args.command == "list-videos":
        list_videos(manager)
    
    elif args.command == "list-scenes":
        list_scenes(manager, args.video)
    
    elif args.command == "search-similar":
        search_similar_frames(manager, args.frame_id, args.results)
    
    elif args.command == "info":
        info = manager.get_collection_info()
        print("Collection Information:")
        print("=" * 30)
        for key, value in info.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
