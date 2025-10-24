#!/usr/bin/env python3
"""
MemryX Accelerator Compilation Script

This script compiles CLIP models for memryX accelerators by:
1. Converting PyTorch CLIP model to TensorFlow/Keras format
2. Using memryX Neural Compiler (mx_nc) to create .dfp files
3. Providing utilities to inspect and test the compiled model
"""

import os
import sys
import subprocess
import torch
import tensorflow as tf
from transformers import CLIPModel
from pathlib import Path

def check_memryx_tools():
    """Check if memryX tools are available"""
    try:
        # Check if mx_nc is available
        result = subprocess.run(['mx_nc', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ memryX Neural Compiler (mx_nc) is available")
            return True
        else:
            print("✗ memryX Neural Compiler (mx_nc) not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ memryX Neural Compiler (mx_nc) not found")
        return False

def pytorch_to_tensorflow_conversion():
    """Convert PyTorch CLIP model to TensorFlow format"""
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Converting PyTorch model to TensorFlow...")
    # Create dummy input for tracing
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX first (as intermediate step)
    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            model.vision_model, 
            dummy_input, 
            "clip_vision.onnx", 
            opset_version=13,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("✓ ONNX export successful")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return None
    
    # Convert ONNX to TensorFlow
    print("Converting ONNX to TensorFlow...")
    try:
        # This would require onnx-tf or similar conversion tool
        # For now, we'll create a simple TensorFlow model structure
        print("Note: Direct ONNX to TensorFlow conversion requires additional tools")
        print("Consider using tf2onnx or onnx-tf for conversion")
        return "clip_vision.onnx"
    except Exception as e:
        print(f"✗ TensorFlow conversion failed: {e}")
        return None

def compile_to_dfp(model_path, num_chips=4):
    """Compile model to .dfp format using memryX Neural Compiler"""
    if not check_memryx_tools():
        print("Please install memryX SDK and Neural Compiler first")
        print("Visit: https://developer.memryx.com/")
        return False
    
    print(f"Compiling {model_path} to .dfp format for {num_chips} chips...")
    
    try:
        # Run mx_nc command
        cmd = ['mx_nc', '-v', '-m', model_path, '-c', str(num_chips)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            dfp_file = model_path.replace('.onnx', '.dfp').replace('.h5', '.dfp')
            print(f"✓ Compilation successful! Generated: {dfp_file}")
            return dfp_file
        else:
            print(f"✗ Compilation failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Compilation error: {e}")
        return False

def inspect_dfp(dfp_path):
    """Inspect the generated .dfp file"""
    try:
        result = subprocess.run(['dfp_inspect', dfp_path], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"DFP File Inspection Results:")
            print(result.stdout)
        else:
            print(f"Inspection failed: {result.stderr}")
    except FileNotFoundError:
        print("dfp_inspect tool not found")

def main():
    """Main compilation workflow"""
    print("=== MemryX Accelerator Compilation ===")
    
    # Step 1: Check memryX tools
    if not check_memryx_tools():
        print("\nTo install memryX tools:")
        print("1. Visit https://developer.memryx.com/")
        print("2. Download and install memryX SDK")
        print("3. Install memryX Neural Compiler (mx_nc)")
        return
    
    # Step 2: Convert model
    model_path = pytorch_to_tensorflow_conversion()
    if not model_path:
        print("Model conversion failed")
        return
    
    # Step 3: Compile to DFP
    dfp_path = compile_to_dfp(model_path, num_chips=4)
    if dfp_path:
        # Step 4: Inspect DFP
        inspect_dfp(dfp_path)
        print(f"\n✓ Compilation complete! Use {dfp_path} with memryX accelerators")
    else:
        print("Compilation failed")

if __name__ == "__main__":
    main()