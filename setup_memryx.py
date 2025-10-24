#!/usr/bin/env python3
"""
MemryX Setup and Installation Script

This script helps set up the environment for memryX accelerator compilation.
It installs required dependencies and provides guidance for memryX SDK installation.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_python_dependencies():
    """Install required Python packages"""
    print("Installing Python dependencies...")
    
    packages = [
        "onnxscript",
        "onnx", 
        "tf2onnx",
        "onnxruntime",
        "onnxsim",
        "protobuf",
        "tensorflow>=2.10.0"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

def check_memryx_sdk():
    """Check if memryX SDK is installed"""
    print("\nChecking for memryX SDK...")
    
    # Check for mx_nc command
    try:
        result = subprocess.run(['mx_nc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ memryX Neural Compiler (mx_nc) found")
            print(f"Version: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Check for memryx Python package
    try:
        import memryx
        print("✓ memryX Python SDK found")
        return True
    except ImportError:
        pass
    
    print("✗ memryX SDK not found")
    return False

def provide_installation_guidance():
    """Provide guidance for memryX SDK installation"""
    print("\n" + "="*60)
    print("MEMRYX SDK INSTALLATION REQUIRED")
    print("="*60)
    print("""
To compile models for memryX accelerators, you need to install the memryX SDK:

1. Visit: https://developer.memryx.com/
2. Create an account and download the SDK
3. Follow the installation instructions for your platform
4. Ensure the following tools are available:
   - mx_nc (memryX Neural Compiler)
   - dfp_inspect (DFP file inspector)
   - memryx Python package

After installation, verify with:
   mx_nc --version
   dfp_inspect --help
""")

def create_test_script():
    """Create a test script to verify the setup"""
    test_script = """#!/usr/bin/env python3
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
    print("\\nTesting model loading...")
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
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    print("✓ Created test_setup.py")

def main():
    """Main setup function"""
    print("=== MemryX Environment Setup ===")
    
    # Install Python dependencies
    install_python_dependencies()
    
    # Check for memryX SDK
    if not check_memryx_sdk():
        provide_installation_guidance()
    
    # Create test script
    create_test_script()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("""
Next steps:
1. Install memryX SDK if not already installed
2. Run: python test_setup.py
3. Run: python compilation.py

For more information, visit: https://developer.memryx.com/
""")

if __name__ == "__main__":
    main()
