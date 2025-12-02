# VideoAgent

A memory-augmented multimodal agent for long-form video understanding. VideoAgent implements an iterative analysis pipeline that combines vision-language models with intelligent frame sampling to answer complex questions about video content.

## Overview

VideoAgent addresses the challenge of understanding long-form videos (e.g., 3-minute egocentric videos) by:

1. **Iterative Refinement**: Starting with initial frame samples and progressively refining analysis based on confidence levels
2. **Memory-Augmented Processing**: Maintaining structured video memory with frame captions, event descriptions, and video overviews
3. **Multi-Level Captioning**: Generating hierarchical descriptions from visual details to temporal event understanding
4. **Intelligent Resampling**: Dynamically selecting additional frames from relevant video segments when initial analysis is insufficient

The system is evaluated on the [EgoSchema](https://egoschema.github.io/) benchmark, a challenging dataset of 500 egocentric videos with multiple-choice questions requiring long-form temporal understanding.

## Features

- Multi-model architecture with separate scheduler and viewer models
- Configurable iterative refinement with confidence-based stopping
- LLM response caching for efficient re-runs
- Parallel video processing with multiprocessing support
- Comprehensive evaluation metrics and logging
- Standardized output structure for reproducible experiments

## Project Structure

```
VideoAgent/
├── configs/                      # Configuration files
│   ├── default.yaml              # Default experiment configuration
│   └── models.yaml               # Model pricing and capability reference
│
├── scripts/                      # Shell scripts and utilities
│   ├── template/
│   │   └── eval.sh               # Evaluation script template
│   ├── run_default.sh            # Default experiment runner
│   └── test.sh                   # Test experiment runner
│
├── video_agent/                  # Main Python package
│   ├── cli.py                    # Command-line interface
│   ├── agent.py                  # Main VideoAgent orchestrator
│   ├── core/
│   │   └── video_memory.py       # VideoMemory class for frame/analysis state
│   ├── processors/
│   │   ├── caption_processor.py  # CaptionProcessor for frame analysis
│   │   └── question_processor.py # QuestionProcessor for Q&A
│   └── utils/
│       ├── api.py                # AIMLClient for LLM API interactions
│       ├── config.py             # Configuration management
│       ├── cache.py              # CacheManager for response caching
│       ├── logging_utils.py      # Logging utilities
│       ├── video.py              # Video processing utilities
│       └── parsing.py            # Text and JSON parsing utilities
│
├── data/                         # Input data directory
│   └── EgoSchema_test/           # EgoSchema test dataset
│       ├── annotations.json      # Video annotations and questions
│       ├── video_list.txt        # List of valid video IDs
│       └── videos/               # Video files (.mp4)
│
├── cache/                        # LLM response cache (gitignored)
│
└── results/                      # Experiment outputs (gitignored)
    └── [experiment_name]/
        ├── logging.log           # Full evaluation log
        ├── result.json           # Results with Q&A + summary stats
        ├── metrics.csv           # Performance metrics table
        ├── summary.txt           # Human-readable summary
        ├── accuracy.txt          # Accuracy metrics
        ├── experiment_config.yaml
        └── videos/               # Per-video outputs
            └── [video_id]/
                ├── frames/       # Sampled frames as PNG
                ├── memory.txt    # Video memory state
                ├── question.txt  # Formatted question
                └── result.json   # Video-specific results
```

## Quick Start

### Prerequisites

- Python 3.9+
- API key for LLM service (supports OpenAI-compatible APIs via [AIML API](https://aimlapi.com/))
- Video dataset (EgoSchema or compatible format)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/VideoAgent.git
cd VideoAgent
```

2. **Create and activate a virtual environment**

```bash
# Using conda (recommended)
conda create -n videoagent python=3.10
conda activate videoagent

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Environment Configuration

1. **Create environment file**

```bash
# Create .env file in project root
touch .env
```

2. **Add your API credentials**

```bash
# .env
AIML_API_KEY=your_api_key_here
AIML_BASE_URL=https://api.aimlapi.com/v1  # Optional, this is the default
```

### Prepare Dataset

1. Download the EgoSchema dataset videos
2. Place videos in `data/EgoSchema_test/videos/`
3. Ensure `annotations.json` and `video_list.txt` are in `data/EgoSchema_test/`

### Running Experiments

#### Option 1: Using the Template Script (Recommended)

1. **Copy the evaluation template**

```bash
cp scripts/template/eval.sh scripts/my_experiment.sh
```

2. **Edit configuration parameters**

```bash
vim scripts/my_experiment.sh
```

Key parameters to configure:

```bash
# Model Configuration
SCHEDULER_MODEL="gpt-4o-mini"    # Model for Q&A and evaluation
VIEWER_MODEL="gpt-4o-mini"       # Model for frame captioning (must support vision)

# Experiment Settings
ROUND_NAME="my_experiment"       # Experiment identifier
COUNT=100                        # Number of videos (0 = full dataset)
MAX_ROUNDS=3                     # Maximum refinement rounds
MAX_PROCESSES=4                  # Parallel workers

# Options
USE_CACHE="true"                 # Reuse cached LLM responses
DETAILED="true"                  # Enable verbose logging
```

3. **Run the experiment**

```bash
chmod +x scripts/my_experiment.sh
./scripts/my_experiment.sh
```

#### Option 2: Using CLI Directly

```bash
python -m video_agent.cli \
    --config default \
    --experiment-name "test_run" \
    --scheduler-model "gpt-4o-mini" \
    --viewer-model "gpt-4o-mini" \
    --max-videos 10 \
    --max-rounds 3 \
    --max-processes 4 \
    --llm-logging
```

#### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Base configuration file name | `default` |
| `--experiment-name` | Name for the experiment | `default_experiment` |
| `--scheduler-model` | Model for Q&A and evaluation | `gpt-4o-mini-2024-07-18` |
| `--viewer-model` | Model for frame captioning | `gpt-4o-mini-2024-07-18` |
| `--max-videos` | Number of videos to process (-1 = all) | `-1` |
| `--max-rounds` | Maximum refinement rounds | `5` |
| `--max-processes` | Number of parallel workers | `10` |
| `--llm-logging` | Enable LLM interaction logging | `false` |
| `--no-cache` | Disable LLM response caching | `false` |
| `--video-list` | Path to video list file | from config |
| `--annotation-file` | Path to annotations JSON | from config |
| `--video-dir` | Path to video directory | from config |

## Configuration

### Default Configuration (`configs/default.yaml`)

```yaml
# Model settings
scheduler_model: "gpt-4o-mini-2024-07-18"
viewer_model: "gpt-4o-mini-2024-07-18"
llm_temperature: 0.7
llm_max_tokens: 10000

# Processing settings
max_rounds: 5
max_retrieved_frames: 5
min_retrieved_frames: 2
default_initial_frames: 5
input_frame_interval: 30
caption_method: "multi_level"

# Dataset paths
dataset_dir: "data/EgoSchema_test"
video_dir: "data/EgoSchema_test/videos"
annotation_file: "data/EgoSchema_test/annotations.json"
test_video_list_file: "data/EgoSchema_test/video_list.txt"

# Output settings
output_dir: "results"
cache_dir: "cache"
use_cache: true

# Processing
multi_process: true
max_processes: 10
max_test_videos: -1
```

### Supported Models

The system supports any OpenAI-compatible API. Common options include:

**Vision Models** (for viewer_model - must support image input):
- `gpt-4o`, `gpt-4o-mini` - OpenAI
- `anthropic/claude-4.5-sonnet` - Anthropic
- `google/gemini-2.5-flash` - Google
- `x-ai/grok-4-1-fast-non-reasoning` - xAI
- `alibaba/qwen-vl-max-latest` - Alibaba

**Text Models** (for scheduler_model):
- Any of the vision models above
- `deepseek/deepseek-chat` - DeepSeek
- `alibaba/qwen-turbo-latest` - Alibaba

See `configs/models.yaml` or `scripts/template/eval.sh` for full model list with pricing.

## Output Structure

Each experiment creates a directory with the following structure:

```
results/<EXPERIMENT_NAME>__<SCHEDULER>_viewer_<VIEWER>_numbers_<COUNT>/
├── logging.log           # Full evaluation log
├── result.json           # Detailed results with Q&A
├── metrics.csv           # Performance metrics table
├── summary.txt           # Human-readable summary with case analysis
├── accuracy.txt          # Quick accuracy reference
├── experiment_config.yaml
└── videos/
    └── [video_id]/
        ├── frames/       # Sampled frames
        ├── memory.txt    # Video memory state
        ├── question.txt  # Question with options
        └── result.json   # Per-video result
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Final answer accuracy |
| First Round Accuracy | Accuracy after first round only |
| Improvement Rate | % of initially wrong answers that became correct |
| Case Types | MAINTAINED, IMPROVED, DEGRADED, FAILED |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      VideoAgent                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Caption    │    │   Question   │    │    Video     │  │
│  │  Processor   │    │  Processor   │    │   Memory     │  │
│  │  (Viewer)    │    │ (Scheduler)  │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         └───────────────────┼───────────────────┘           │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │   AIMLClient    │                      │
│                    │   (LLM API)     │                      │
│                    └─────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Processing Pipeline

1. **Initialization**: Load video, sample initial frames
2. **Caption Generation**: Generate multi-level captions for sampled frames
3. **Question Answering**: Answer question based on video memory
4. **Confidence Evaluation**: Assess answer confidence (1-3 scale)
5. **Iteration**: If confidence < 3 and rounds < max_rounds:
   - Generate segment resampling strategy
   - Sample additional frames from relevant segments
   - Return to step 2
6. **Result**: Return final answer with analysis

## License

This project is for research purposes.

## Acknowledgments

- [EgoSchema](https://egoschema.github.io/) for the benchmark dataset
- [AIML API](https://aimlapi.com/) for unified LLM API access

