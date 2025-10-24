#!/usr/bin/env python3
import torch
import tensorflow as tf
from transformers import CLIPModel

def test_imports():
    print("Testing imports...")
    try:
        import onnxscript
        print("✓ onnxscript")
    except ImportError as e:
        print(f"✗ onnxscript: {e}")
    
    try:
        import onnx
        print("✓ onnx")
    except ImportError as e:
        print(f"✗ onnx: {e}")
    
    try:
        import tf2onnx
        print("✓ tf2onnx")
    except ImportError as e:
        print(f"✗ tf2onnx: {e}")
    
    try:
        import memryx
        print("✓ memryx")
    except ImportError as e:
        print(f"✗ memryx: {e}")

def test_model_loading():
    print("\nTesting model loading...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        print("✓ CLIP model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None

if __name__ == "__main__":
    test_imports()
    test_model_loading()
