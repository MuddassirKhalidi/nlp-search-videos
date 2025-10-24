# MemryX Accelerator Compilation Guide

This guide explains how to compile CLIP models for memryX accelerators to create `.dfp` (Data Flow Program) files.

## Current Process vs. Required Process

### What Was Happening (Incorrect):
- Your `compilation.py` was trying to export PyTorch CLIP model to ONNX format
- Missing `onnxscript` dependency caused the error
- ONNX format is not directly compatible with memryX accelerators

### What Should Happen (Correct):
1. **Install memryX SDK** - Download from https://developer.memryx.com/
2. **Convert model** - PyTorch → TensorFlow/Keras or ONNX
3. **Compile with mx_nc** - Use memryX Neural Compiler to create `.dfp` files
4. **Load and run** - Use memryX Accelerator API to execute on hardware

## Setup Instructions

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install memryX SDK
- Visit https://developer.memryx.com/
- Download and install memryX SDK for your platform
- Ensure `mx_nc` and `dfp_inspect` tools are available

### 3. Run Setup Script
```bash
python setup_memryx.py
```

### 4. Test Installation
```bash
python test_setup.py
```

## Compilation Process

### Step 1: Model Conversion
The script converts PyTorch CLIP model to ONNX format:
```python
torch.onnx.export(model.vision_model, dummy_input, "clip_vision.onnx", opset_version=13)
```

### Step 2: Compile to DFP
Use memryX Neural Compiler:
```bash
mx_nc -v -m clip_vision.onnx -c 4
```
This creates `clip_vision.dfp` for 4-chip configuration.

### Step 3: Inspect DFP
```bash
dfp_inspect clip_vision.dfp
```

## Usage Example

```python
from memryx import SyncAccl

# Load the compiled DFP file
accl = SyncAccl("clip_vision.dfp")

# Prepare input data
import numpy as np
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = accl.run(input_data)
```

## Troubleshooting

### Common Issues:

1. **Missing onnxscript**: Install with `pip install onnxscript`
2. **memryX tools not found**: Install memryX SDK from developer portal
3. **Model too large**: memryX MX3 chips support up to 10.5M parameters
4. **Unsupported operations**: Some operations may need to run on CPU

### File Size Optimization:
```bash
mx_nc -v -m model.onnx -c 4 --no_sim_dfp
```
Use `--no_sim_dfp` flag to exclude simulator config and reduce file size.

## Files Created

- `compilation.py` - Main compilation script
- `setup_memryx.py` - Environment setup script  
- `test_setup.py` - Installation verification script
- `requirements.txt` - Updated with memryX dependencies

## Next Steps

1. Install memryX SDK
2. Run `python setup_memryx.py`
3. Run `python compilation.py`
4. Use generated `.dfp` file with memryX accelerators

For more information, visit: https://developer.memryx.com/

