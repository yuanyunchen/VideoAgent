# VideoAgent

A video analysis system for question answering with LLM-powered iterative refinement.

## Quick Start

### Step 1: Setup Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd VideoAgent

# Install as package (recommended)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API key
AIML_API_KEY=your-api-key-here
```

### Step 3: Run Experiments

```bash
# Using the CLI
python -m video_agent.cli --config default --max-videos 10

# Using shell scripts
./scripts/run_default.sh
```

## Features

- **YAML Configuration**: Simple YAML config files in `configs/` directory
- **Iterative Refinement**: Multi-round analysis with confidence-based sampling
- **Multiple Caption Methods**: Detailed, multi-level, or group captioning
- **Smart Frame Sampling**: Intelligent frame selection based on analysis
- **Performance Optimization**: Built-in caching and response reuse
- **Multiprocessing Support**: Parallel video processing for faster experiments

## Installation

```bash
# Method 1: Install as package
pip install -e .

# Method 2: Install dependencies only
pip install -r requirements.txt

# Setup data directories
mkdir -p data/videos cache results

# Make scripts executable
chmod +x scripts/*.sh
```

## Usage

### Command Line

```bash
# List available configurations
python -m video_agent.cli --list-configs

# Run with default configuration
python -m video_agent.cli --config default

# Override specific parameters
python -m video_agent.cli --config default --max-videos 10 --scheduler-model gpt-4o

# Enable LLM logging for debugging
python -m video_agent.cli --config default --llm-logging

# Use multiprocessing
python -m video_agent.cli --config default --max-processes 4
```

### Python API

```python
from video_agent import VideoAgent
from video_agent.utils.config import load_config

# Load configuration
config = load_config("default")

# Initialize VideoAgent
agent = VideoAgent(config=config)

# Run experiment
output_dir = agent.run_experiment()
print(f"Results saved to: {output_dir}")
```

### Shell Scripts

```bash
# Default experiment
./scripts/run_default.sh

# Test experiment with custom models
./scripts/test.sh
```

## Project Structure

```
VideoAgent/
├── .env.example              # Environment variable template
├── pyproject.toml            # Python packaging configuration
├── requirements.txt          # Dependencies
├── configs/                  # YAML configuration files
│   └── default.yaml
├── scripts/                  # Shell scripts
│   ├── run_default.sh
│   └── test.sh
├── video_agent/              # Main Python package
│   ├── __init__.py
│   ├── cli.py                # Command-line interface
│   ├── agent.py              # Main VideoAgent class
│   ├── core/                 # Core data structures
│   │   └── video_memory.py
│   ├── processors/           # Processing modules
│   │   ├── caption_processor.py
│   │   └── question_processor.py
│   └── utils/                # Utilities
│       ├── api.py            # LLM API interface
│       ├── config.py         # Configuration management
│       ├── cache.py          # Caching
│       ├── logging_utils.py  # Logging
│       ├── video.py          # Video processing
│       └── parsing.py        # Text parsing
├── data/                     # Input data
│   ├── annotations/          # JSON annotations
│   ├── video_lists/          # Video ID lists
│   └── videos/               # Video files
├── cache/                    # Response cache (gitignored)
└── results/                  # Experiment results (gitignored)
```

## Configuration

### Available Configurations

```bash
python -m video_agent.cli --list-configs
```

### Configuration Options

Key configuration parameters in `configs/default.yaml`:

```yaml
# Models
scheduler_model: "gpt-4o-mini-2024-07-18"  # Model for Q&A
viewer_model: "gpt-4o-mini-2024-07-18"     # Model for captions

# Processing
max_rounds: 5                # Maximum analysis rounds
max_test_videos: -1          # Number of videos (-1 = all)
max_processes: 10            # Parallel processes

# Paths
output_dir: "results"
video_dir: "data/videos"
annotation_file: "data/annotations/subset_anno.json"
```

### Environment Variables

Set these in your `.env` file:

```bash
AIML_API_KEY=your_api_key_here
AIML_BASE_URL=https://api.aimlapi.com/v1  # Optional
```

## Output Structure

```
results/
└── experiment_name__model_info/
    ├── result.json              # Aggregated results
    ├── accuracy.txt             # Performance metrics
    ├── experiment_config.yaml   # Saved configuration
    ├── logging.log              # Experiment logs
    └── videos/
        └── [video_id]/
            ├── frames/          # Sampled frames
            ├── memory.txt       # Video memory state
            ├── question.txt     # Formatted question
            ├── result.json      # Video results
            └── logging.log      # Video logs
```

## Multiprocessing

VideoAgent supports parallel processing for faster experiments:

```bash
# Use 4 parallel processes
python -m video_agent.cli --config default --max-processes 4

# Disable multiprocessing
python -m video_agent.cli --config default --no-multiprocess
```

Performance notes:
- Recommended process count: 2-4 for most systems
- Progress updates may be less frequent in multiprocessing mode
- Each process has its own memory space and logging

## Troubleshooting

### API Key Issues

```bash
# Check if API key is set
echo $AIML_API_KEY

# Set API key
export AIML_API_KEY=your_key_here
```

### Permission Issues

```bash
chmod +x scripts/*.sh
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.
