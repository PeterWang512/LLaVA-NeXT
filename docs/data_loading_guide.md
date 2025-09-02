# Data Loading Guide for LLaVA-NeXT Training

This guide explains how data is loaded and processed during training time in LLaVA-NeXT. Understanding this process is crucial for preparing custom datasets and optimizing training performance.

## Overview

The LLaVA-NeXT training data loading pipeline consists of several key components:

1. **LazySupervisedDataset**: The main dataset class that loads and processes training data
2. **Data Preprocessing**: Functions that format conversations and process multimodal content
3. **Data Collation**: Batching and padding logic for efficient training
4. **LLaVATrainer**: Custom trainer that integrates with the data loading pipeline

## Data Loading Flow

```
JSON/YAML Files → LazySupervisedDataset → Preprocessing → DataCollator → LLaVATrainer → Model
```

## 1. Dataset Input Formats

### JSON Format
The primary data format is JSON with the following structure:

```json
[
  {
    "id": "sample_001",
    "image": "path/to/image.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat do you see in this image?"
      },
      {
        "from": "gpt", 
        "value": "I can see a beautiful landscape with mountains and trees."
      }
    ]
  }
]
```

### Video Format
For video data:

```json
[
  {
    "id": "video_001",
    "video": "path/to/video.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nDescribe what happens in this video."
      },
      {
        "from": "gpt",
        "value": "The video shows a person walking through a park."
      }
    ]
  }
]
```

### YAML Configuration Format
For complex dataset mixing, use YAML configuration:

```yaml
datasets:
  - json_path: dataset1.json
    sampling_strategy: first:1000
  - json_path: dataset2.json  
    sampling_strategy: random:500
  - json_path: dataset3.json
    sampling_strategy: end:200
```

**Sampling Strategies:**
- `first:N` - Take first N samples
- `end:N` - Take last N samples  
- `random:N` - Randomly sample N samples
- `first:N%` - Take first N% of samples
- `all` - Use all samples (default)

## 2. LazySupervisedDataset Class

Located in `llava/train/train.py`, this class handles:

### Data Loading
```python
class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        # Load data from JSON files or YAML config
        # Handle multiple files with pattern: /path/to/{a,b,c}.json
        # Apply sampling strategies if using YAML
```

### Key Features:
- **Lazy Loading**: Data is processed on-demand during training, not pre-loaded
- **Multiple File Support**: Load from multiple JSON files using `{file1,file2}.json` pattern
- **Retry Logic**: Automatic retry mechanism for failed data loading (network issues, corruption)
- **Mixed Data Types**: Support for images, videos, and text-only data

### Data Processing Methods:

#### Image Processing (`process_image` method)
```python
def process_image(self, image_file, overwrite_image_aspect_ratio=None):
    # Load image from image_folder
    # Apply aspect ratio processing based on configuration:
    # - "square": Standard square processing
    # - "pad": Pad to square with background color
    # - "highres": High-resolution grid processing  
    # - "anyres": Adaptive resolution processing
    # - "crop_split": Crop and split processing
```

**Image Aspect Ratio Options:**
- `square`: Resize to square dimensions
- `pad`: Pad image to square while maintaining aspect ratio
- `highres`: Process high-resolution images with grid-based approach
- `anyres`: Adaptive resolution processing for better quality
- `crop_split`: Crop and split large images

#### Video Processing
```python
# Video processing supports:
# - Frame sampling at specified FPS
# - Uniform frame distribution across video duration
# - Time instruction injection (optional)
# - Support for both video files and frame directories
```

### Error Handling and Retries
The dataset implements robust error handling:

```python
def __getitem__(self, i):
    # Try current sample (3 attempts)
    for attempt_idx in range(num_base_retries):
        try:
            return self._get_item(i)
        except Exception as e:
            print(f"Failed to fetch sample {i}. Exception:", e)
            time.sleep(1)
    
    # Try adjacent samples if current fails
    # Final retry before raising exception
```

## 3. Data Preprocessing

### Conversation Preprocessing
Different preprocessing functions handle various model architectures:

- `preprocess_v1()`: Original LLaVA format
- `preprocess_llama_2()`: LLaMA-2 chat format
- `preprocess_llama3()`: LLaMA-3 chat format  
- `preprocess_qwen()`: Qwen model format
- `preprocess_gemma()`: Gemma model format

#### Example: LLaMA-3 Preprocessing
```python
def preprocess_llama3(sources, tokenizer, has_image=False, system_message="..."):
    # Apply chat template for LLaMA-3 format
    # Handle image tokens and special tokens
    # Create input_ids and labels for training
    # Mask human utterances in labels (set to IGNORE_INDEX)
```

### Multimodal Token Handling
```python
def preprocess_multimodal(sources, data_args):
    # Replace DEFAULT_IMAGE_TOKEN with appropriate tokens
    # Add start/end tokens if configured
    # Handle multiple images in conversations
    # Clean noisy data patterns
```

**Key Tokens:**
- `DEFAULT_IMAGE_TOKEN`: `<image>` placeholder for images
- `DEFAULT_IM_START_TOKEN`: Start token for image regions
- `DEFAULT_IM_END_TOKEN`: End token for image regions
- `IMAGE_TOKEN_INDEX`: Special index for image tokens in vocabulary

## 4. Data Collation

The `DataCollatorForSupervisedDataset` class handles batching:

```python
class DataCollatorForSupervisedDataset:
    def __call__(self, instances):
        # Extract input_ids and labels from batch
        # Truncate sequences to model_max_length
        # Pad sequences to same length in batch
        # Handle attention masks
        # Process images/videos if present
        # Return batched tensors
```

### Batch Structure
```python
batch = {
    "input_ids": torch.Tensor,      # [batch_size, seq_len]
    "labels": torch.Tensor,         # [batch_size, seq_len] 
    "attention_mask": torch.Tensor, # [batch_size, seq_len]
    "images": List[torch.Tensor],   # List of image tensors
    "image_sizes": List[Tuple],     # Original image dimensions
    "modalities": List[str],        # ["image", "video", "text"]
}
```

## 5. Configuration Parameters

### DataArguments
Key parameters for data loading:

```python
@dataclass
class DataArguments:
    data_path: str                    # Path to training data
    lazy_preprocess: bool = False     # Enable lazy preprocessing
    is_multimodal: bool = False       # Enable multimodal processing
    image_folder: str = None          # Root folder for images
    image_aspect_ratio: str = "square" # Image processing mode
    image_grid_pinpoints: str = None  # Grid points for high-res processing
    video_folder: str = None          # Root folder for videos
    video_fps: int = 1               # Frames per second for video sampling
    frames_upbound: int = 0          # Maximum number of frames
    add_time_instruction: bool = False # Add timing info to video prompts
```

### Image Processing Configuration
```python
# Grid points for high-resolution processing
image_grid_pinpoints = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]

# Crop and split resolution settings  
image_crop_resolution = 448
image_split_resolution = 224
```

## 6. Training Integration

### LLaVATrainer Data Loading
```python
def get_train_dataloader(self):
    # Create DataLoader with configured parameters
    # Apply data collation
    # Set up distributed sampling if needed
    # Configure worker processes and prefetching
```

### Data Module Creation
```python
def make_supervised_data_module(tokenizer, data_args):
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, 
        data_path=data_args.data_path, 
        data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, 
        eval_dataset=None, 
        data_collator=data_collator
    )
```

## 7. Performance Considerations

### Memory Optimization
- **Lazy Loading**: Images/videos loaded on-demand, not pre-cached
- **Dynamic Batching**: Variable length sequences padded per batch
- **Worker Processes**: Configurable number of data loading workers

### Distributed Training
- **Data Sharding**: Automatic data distribution across multiple GPUs
- **Reproducible Sampling**: Consistent data ordering across processes
- **Load Balancing**: Even distribution of computational load

## 8. Common Issues and Solutions

### Data Loading Errors
```python
# Network/disk issues handled by retry mechanism
# Corrupt files automatically skipped to next sample
# Missing image/video files logged and handled gracefully
```

### Memory Issues
```python
# Reduce batch_size in training arguments
# Lower image resolution or reduce frames_upbound for videos
# Increase dataloader_num_workers for better I/O parallelism
```

### Format Issues
```python
# Ensure JSON structure matches expected format
# Verify image/video paths are relative to specified folders
# Check conversation format matches model requirements
```

## 9. Example Usage

### Basic Training Data Preparation
```bash
# Organize data structure
data/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
├── videos/  
│   ├── video1.mp4
│   └── video2.mp4
└── train.json

# Training command with data arguments
python llava/train/train.py \
    --data_path data/train.json \
    --image_folder data/images \
    --video_folder data/videos \
    --image_aspect_ratio pad \
    --video_fps 2 \
    --is_multimodal True
```

### Multi-Dataset Training with YAML
```yaml
# config.yaml
datasets:
  - json_path: dataset1.json
    sampling_strategy: first:5000
  - json_path: dataset2.json  
    sampling_strategy: random:3000
  - json_path: dataset3.json
    sampling_strategy: all
```

```bash
python llava/train/train.py \
    --data_path data/config.yaml \
    --image_folder data/images \
    --is_multimodal True
```

This guide provides a comprehensive understanding of how data flows through the LLaVA-NeXT training pipeline, from raw input files to processed batches ready for model training.