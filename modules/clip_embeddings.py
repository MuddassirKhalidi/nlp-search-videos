from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from modules.scene_utils import get_scene_frame_samples
import cv2
import numpy as np
import os
import torch
from modules.chromadb_manager import ChromaDBManager

video_path = '/mnt/c/Users/Administrator/Desktop/memryx-testing/videos/cutting_pepper.mp4' # path to video on machine



# Initialize CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embeddings(image):
    """Get CLIP embeddings for an image"""
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        # Normalize embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.numpy().flatten()

def process_video_embeddings(video_path):
    """Process video and return embeddings with metadata"""
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return []
    
    try:
        # Try different thresholds if no scenes detected
        thresholds = [15.0, 10.0, 5.0, 2.0]
        scenes_frame_samples = []
        
        for threshold in thresholds:
            scenes_frame_samples = get_scene_frame_samples(video_path, 3, threshold=threshold)
            if scenes_frame_samples:
                print(f"Successfully detected scenes with threshold {threshold}")
                break
        
        if not scenes_frame_samples:
            print("No scenes detected with any threshold - treating entire video as one scene")
            # Fallback: sample frames evenly across the entire video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f} seconds")
            
            # Sample 3 frames evenly across the video
            if total_frames > 0:
                frame_samples = [int(total_frames * i / 3) for i in range(3)]
                scenes_frame_samples = [frame_samples]  # Treat as one scene
                print(f"Using fallback sampling: frames {frame_samples}")
            else:
                print("Could not determine video properties")
                return []
        
        embeddings_data = []
        
        for scene_idx in range(len(scenes_frame_samples)):
            scene_samples = scenes_frame_samples[scene_idx]
            print(f"Processing scene {scene_idx} with {len(scene_samples)} frame samples")
            
            for frame_idx, frame_sample in enumerate(scene_samples):
                cap.set(1, frame_sample)
                ret, frame = cap.read()
                if not ret:
                    print(f'Failed to read frame {frame_sample} in scene {scene_idx}')
                    continue
                    
                pil_image = Image.fromarray(frame)
                embedding = get_clip_embeddings(pil_image)
                
                # Create unique ID for this frame
                frame_id = f"scene_{scene_idx}_frame_{frame_idx}_sample_{frame_sample}"
                
                embeddings_data.append({
                    'id': frame_id,
                    'embedding': embedding,
                    'metadata': {
                        'video_path': video_path,
                        'scene_idx': scene_idx,
                        'frame_idx': frame_idx,
                        'frame_sample': frame_sample,
                        'video_name': os.path.basename(video_path)
                    }
                })
        
        return embeddings_data
        
    finally:
        cap.release()

if __name__ == "__main__":
    process_video_embeddings(video_path)