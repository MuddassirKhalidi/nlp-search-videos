# Video Processing with CLIP Embeddings

A Python application that processes videos, extracts CLIP embeddings, stores them in ChromaDB, and enables text-based video frame search.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Sample videos are available in the `/videos` directory for testing.

The `main.py` script provides several commands for video processing and search:

### Process Videos

**Process a single video:**
```bash
python main.py videos/sample.mp4
```

**Process multiple videos:**
```bash
python main.py videos/sample.mp4 videos/sample1.mp4
```

**Process all videos in a directory:**
```bash
python main.py --directory videos/
```

### Search Videos

**Search by text query (saves matched frames as images):**
```bash
python main.py --search "person cutting vegetables"
python main.py --search "kitchen scene"
```

**Search by text query (without saving images):**
```bash
python main.py --search-no-save "person cutting vegetables"
```

## What it does

1. **Video Processing**: Extracts frames from videos and generates CLIP embeddings
2. **Storage**: Saves embeddings to ChromaDB for efficient retrieval
3. **Search**: Enables natural language search through video frames
4. **Results**: Saves matching frames to `matched_imgs/` directory with similarity scores

## Dependencies

- `scenedetect` - Video scene detection
- `opencv-python` - Video processing
- `numpy` - Numerical operations
- `transformers` - CLIP model
- `Pillow` - Image processing
- `torch` - PyTorch for ML models
- `chromadb` - Vector database for embeddings
