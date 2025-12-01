# VideoAgent 🎥🤖

A simplified video analysis system for question answering with YAML configuration management.

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

### Step 1: Setup Dependencies
```bash
# Clone the repository
git clone <repository-url>
cd VideoAgent_Main

# Install dependencies
pip install openai opencv-python numpy tqdm pyyaml
```

### Step 2: Configure API Keys
```yaml
# Edit configs/default.yaml or create your own config
aiml_api_key: "your-api-key-here"
```

### Step 3: Run Experiments

#### Using Shell Scripts (Recommended)
```bash
# List available configurations
python main.py --list-configs

# Run with default configuration
./scripts/run_default.sh

# Run high accuracy analysis
./scripts/run_high_accuracy.sh

# Run fast performance test
./scripts/run_fast_performance.sh

# Test with single video (debugging)
./scripts/run_single_video.sh
```

#### Using Python API
```python
from utils.config import load_config
from video_agent import VideoAgent

# Load configuration
config = load_config("high_accuracy")

# Initialize VideoAgent
agent = VideoAgent(config=config)

# Run experiment
output_dir = agent.run_experiment()
print(f"Results saved to: {output_dir}")
```

#### Using Command Line
```bash
# Use specific configuration
python main.py --config high_accuracy --max-videos 10

# Override specific parameters
python main.py --config default --scheduler-model gpt-4o --llm-logging

# Mix configuration and overrides
python main.py --config fast_performance --max-rounds 3 --caption-method detailed
```

## ✨ Features

- **📋 YAML Configuration Management**: Simple YAML config files in `configs/` directory
- **🔄 Iterative Refinement**: Multi-round analysis with confidence-based sampling
- **🖼️ Multiple Caption Methods**: Choose between detailed, multi-level, or group captioning
- **🎯 Smart Frame Sampling**: Intelligent frame selection with configurable methods
- **⚡ Performance Optimization**: Built-in caching and streamlined processing
- **🔧 Modular Design**: Clean separation of concerns with standardized interfaces
- **📊 Comprehensive Logging**: Detailed experiment tracking with config saving
- **📈 Progress Tracking**: Real-time progress bars with accuracy monitoring using tqdm
- **🚀 Multiprocessing Support**: Parallel video processing for faster experiments

## 🚀 Multiprocessing Support

VideoAgent now supports parallel processing to significantly speed up experiments when processing multiple videos.

### Configuration

```yaml
# In your config file (e.g., configs/default.yaml)
max_processes: 4      # Number of parallel processes (default: 1)
multi_process: true   # Enable multiprocessing (default: false)
```

### Command Line Usage

```bash
# Use 4 processes for parallel processing
python main.py --config default --max-processes 4

# Disable multiprocessing (single process mode)
python main.py --config default --no-multiprocess

# Auto-detect CPU count and use multiprocessing
python main.py --config default --max-processes 0
```

### Performance Benefits

- **Speed**: Process multiple videos simultaneously
- **Efficiency**: Better CPU utilization for I/O-bound tasks
- **Scalability**: Automatic process count detection based on CPU cores
- **Safety**: Automatic fallback to single process mode if multiprocessing fails

### Important Notes

- Progress updates may be less frequent in multiprocessing mode
- Each process has its own memory space and logging
- Recommended process count: 2-4 for most systems
- For very large datasets, consider using fewer processes to avoid memory issues

### Testing Multiprocessing

```bash
# Test multiprocessing functionality
python test_multiprocess.py

# Compare single vs multi-process performance
python main.py --config default --max-videos 10 --max-processes 1
python main.py --config default --max-videos 10 --max-processes 4
```

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd VideoAgent_Main

# 2. Install dependencies
pip install openai opencv-python numpy tqdm pyyaml

# 3. Setup data directories
mkdir -p dataset/videos cache output

# 4. Make scripts executable
chmod +x scripts/*.sh

# 5. Verify installation
python main.py --list-configs
```

## 🔧 Configuration System

### Available Configurations

```bash
python main.py --list-configs
```

**Built-in Configurations:**
- `default`: Balanced settings for general use
- `high_accuracy`: Maximum accuracy with thorough analysis  
- `fast_performance`: Speed-optimized for quick processing

### Configuration Structure

All configurations are YAML files in the `configs/` directory:

```yaml
scheduler_model: "gpt-4o-mini-2024-07-18"
viewer_model: "gpt-4o-mini-2024-07-18"
max_rounds: 1
caption_method: "multi_level"
video_processing_method: "standard"
experiment_name: "default_experiment"
```

### Creating Custom Configurations

1. **Copy an existing configuration**
   ```bash
   cp configs/default.yaml configs/my_config.yaml
   ```

2. **Edit the parameters**
   ```yaml
   scheduler_model: "gpt-4o"
   max_rounds: 5
   experiment_name: "my_experiment"
   ```

3. **Use your configuration**
   ```bash
   python main.py --config my_config
   ```

## 🏗️ Architecture

### Project Structure

```
VideoAgent_Main/
├── configs/                    # YAML configuration files
│   ├── default.yaml            # Default balanced configuration
│   ├── high_accuracy.yaml      # Maximum accuracy settings
│   └── fast_performance.yaml   # Speed-optimized settings
├── scripts/                    # Ready-to-use shell scripts
│   ├── run_default.sh          # Basic experiment runner
│   ├── run_high_accuracy.sh    # High accuracy analysis
│   ├── run_fast_performance.sh # Fast processing
│   ├── run_model_comparison.sh # Compare different models
│   └── run_single_video.sh     # Single video debugging
├── core/                       # Core data structures
│   └── video_memory.py         # Video state management
├── processors/                 # Processing modules  
│   ├── caption_processor.py    # Caption generation (detailed/multi_level/group)
│   └── question_processor.py   # Q&A and confidence evaluation
├── utils/                      # Utility modules
│   ├── config.py               # Configuration management
│   ├── AIML_API.py             # LLM API interface
│   ├── utils_clip.py           # CLIP embeddings and retrieval
│   └── general.py        # General utilities
├── video_agent.py              # Main orchestrator class
└── main.py                     # CLI interface with config management
```

## 📜 Shell Scripts

### Available Scripts

| Script | Purpose | Configuration |
|--------|---------|--------------|
| `run_default.sh` | Basic experiment | Default settings |
| `run_high_accuracy.sh` | Maximum accuracy | High accuracy config |
| `run_fast_performance.sh` | Speed optimized | Fast performance config |
| `run_model_comparison.sh` | Compare models | Multiple model combos |
| `run_single_video.sh` | Debug single video | Debug-friendly settings |

### Usage Examples

```bash
# Run default experiment
./scripts/run_default.sh

# High accuracy analysis with logging
./scripts/run_high_accuracy.sh

# Fast processing for large datasets
./scripts/run_fast_performance.sh

# Debug single video with detailed logs
./scripts/run_single_video.sh
```

## 🎛️ Caption Methods

### Available Methods

1. **`detailed`**: Focused visual descriptions
   - Single-level caption generation
   - Emphasis on visual elements
   - Good for basic analysis

2. **`multi_level`**: Comprehensive analysis (Default)
   - Visual descriptions + event understanding
   - Hierarchical information structure
   - Best for complex question answering

3. **`group`**: Efficient batch processing
   - Streamlined caption generation
   - Optimized for speed
   - Good for large-scale experiments

## 📊 Output Structure

```
output/
└── experiment_name__viewer_model__scheduler_model/
    ├── experiment_config.json   # Configuration used for this experiment
    ├── logging.log             # Experiment logs
    ├── llm.log                 # LLM interaction logs (optional)
    └── videos/                 # Per-video analysis
        └── video_id/
            ├── frames/         # Sampled frames
            ├── memory.txt      # Video memory state
            ├── result.json     # Video-specific results
            └── logging.log     # Video processing logs
```

## 🔧 Troubleshooting

### Common Issues

1. **Configuration Issues**
   ```bash
   # List available configs
   python main.py --list-configs
   
   # Test specific config
   python main.py --config default --help
   ```

2. **Missing Dependencies**
   ```bash
   # Install all dependencies
   pip install openai opencv-python numpy tqdm
   ```

3. **Shell Script Permission Issues**
   ```bash
   chmod +x scripts/*.sh
   ```

4. **API Key Issues**
   ```yaml
   # Edit configs/default.yaml and set your API key
   aiml_api_key: "your-api-key-here"
   ```

5. **Configuration File Issues**
   ```bash
   # Validate YAML syntax
   python -m yaml.tool configs/default.yaml
   ```

## 🤝 Contributing

1. **Fork the repository**
2. **Setup dependencies**: `pip install openai opencv-python numpy tqdm`
3. **Test with configurations**: `python main.py --list-configs`
4. **Create a feature branch**: `git checkout -b feature-name`
5. **Commit changes**: `git commit -am 'Add new feature'`
6. **Push to branch**: `git push origin feature-name`
7. **Submit a Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the GPT models and API
- AIML API for model access infrastructure  
- OpenCV community for video processing tools

