# VideoAgent

A multi-tool video understanding agent built with LangGraph for long-form video question answering.

## Overview

VideoAgent is a ReAct-style agent that combines large language models with specialized video analysis tools to answer complex questions about video content. The agent:

1. **Receives video context** with initial frame captions and video description
2. **Iteratively selects and calls tools** (sampling, detection, Q&A, etc.)
3. **Submits answer** when confident or when reaching tool call limit

The system is evaluated on [EgoSchema](https://egoschema.github.io/), a challenging benchmark of 500 egocentric videos requiring long-form temporal understanding.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VideoAgent                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         LangGraph Agent                               │   │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────────────────────┐  │   │
│  │  │   Agent    │ ─▶ │   Tools    │ ─▶ │  Force Answer (if needed)  │  │   │
│  │  │    Node    │ ◀─ │    Node    │    │                            │  │   │
│  │  └────────────┘    └────────────┘    └────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          Tool Manager                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Sampling   │  │  Detection   │  │     Q&A      │  ...          │   │
│  │  │    Tools     │  │    Tools     │  │    Tools     │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

- **ReAct-style Agent**: LLM-driven tool selection with reasoning
- **Multi-Tool Support**: Configurable set of video analysis tools
- **Auto-Captioning**: Frame-returning tools automatically caption results
- **Multi-GPU Support**: Resource management for parallel processing
- **Caching**: Model-aware caching for captions and descriptions

## Project Structure

```
VideoAgent/
├── video_agent_tools/              # Main Python package
│   ├── __init__.py                 # Package exports
│   ├── cli.py                      # Command-line interface
│   ├── evaluation.py               # Batch evaluation framework
│   ├── graph.py                    # LangGraph agent definition
│   ├── prompts.py                  # Agent prompts and templates
│   ├── state.py                    # State definitions
│   ├── tools.py                    # Tool manager
│   ├── resource_management/        # GPU resource management
│   │   ├── core.py                 # Base resource classes
│   │   ├── gpu_manager.py          # Multi-GPU management
│   │   ├── tool_server.py          # Central tool server
│   │   └── tool_client.py          # Worker tool client
│   └── utils/                      # Utility modules
│       ├── logging.py              # Logging utilities
│       ├── tool_cache.py           # Tool result caching
│       └── video.py                # Video processing
│
├── tools/                          # Tool implementations
│   ├── interface_base.py           # Base Interface class
│   └── interface/                  # Tool interfaces
│       ├── __init__.py             # Tool registry
│       ├── image_captioning.py     # OmniCaptioner, API captioning
│       ├── internvideo2_5_interface.py  # InternVideo2.5 Q&A
│       ├── object_detection.py     # YOLO-World, YOLOE
│       ├── object_description.py   # DAM region description
│       ├── temporal_spatial_understanding.py  # TStar, VideoTree
│       ├── view_frame.py           # Frame viewing
│       └── visual_qa.py            # General VQA
│
├── configs/                        # Configuration files
│   ├── default.yaml                # Default settings
│   └── models.yaml                 # Model reference
│
├── scripts/                        # Evaluation scripts
│   └── template/
│       └── eval.sh                 # Evaluation template
│
├── data/                           # Dataset directory
│   └── EgoSchema_test/
│       ├── annotations.json        # Questions and answers
│       ├── video_list.txt          # Video IDs
│       └── videos/                 # Video files (gitignored)
│
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
└── .gitignore                      # Git ignore rules
```

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (for local tools)
- API key for LLM service (supports OpenAI-compatible APIs)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/VideoAgent.git
cd VideoAgent
```

2. **Create environment**

```bash
conda create -n videoagent python=3.10
conda activate videoagent
```

3. **Install dependencies**

```bash
pip install -r requirements.txt

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. **Configure API key**

```bash
cp .env.example .env
# Edit .env and add your API key:
# AIML_API_KEY=your_api_key_here
```

### Prepare Dataset

1. Download EgoSchema videos from [EgoSchema](https://egoschema.github.io/)
2. Place videos in `data/EgoSchema_test/videos/`
3. Ensure `annotations.json` and `video_list.txt` are in `data/EgoSchema_test/`

### Running Experiments

#### Option 1: Using Evaluation Script (Recommended)

```bash
# Copy and configure the template
cp scripts/template/eval.sh scripts/my_experiment.sh

# Edit configuration (model, tools, etc.)
vim scripts/my_experiment.sh

# Run evaluation
chmod +x scripts/my_experiment.sh
./scripts/my_experiment.sh
```

Key settings in `eval.sh`:

```bash
# Agent model
AGENT_MODEL="x-ai/grok-4-1-fast-reasoning"

# Enabled tools
TOOLS="internvideo_general_qa,temporal_sample_frames,view_frame,detect_objects"

# Experiment settings
COUNT=100               # Number of videos (0 = all)
MAX_TOOL_CALLS=20       # Max tool calls per video
INITIAL_FRAMES=5        # Initial frames to caption
```

#### Option 2: Using CLI Directly

```bash
python -m video_agent_tools.cli \
    --model "gpt-4o-mini" \
    --tools "temporal_sample_frames,view_frame,detect_objects" \
    --max-tool-calls 15 \
    --max-videos 10 \
    --annotation-file data/EgoSchema_test/annotations.json \
    --video-dir data/EgoSchema_test/videos \
    --experiment-name "test_run"
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | LLM model for agent | `gpt-4o-mini` |
| `--tools` | Comma-separated tool list | See default |
| `--max-tool-calls` | Max tool calls per video | `10` |
| `--max-parallel-tools` | Max tools per turn | `3` |
| `--initial-frames` | Frames to caption initially | `5` |
| `--max-videos` | Number of videos (-1 = all) | `-1` |
| `--num-workers` | Parallel workers | `1` |
| `--captioner` | Caption model | `omni-captioner` |
| `--experiment-name` | Experiment identifier | `tools_agent` |

## Available Tools

### Q&A Tools
| Tool | Description | Backend |
|------|-------------|---------|
| `internvideo_general_qa` | General video Q&A (128 frames) | InternVideo2.5 |
| `internvideo_description` | Video summary + action timeline | InternVideo2.5 |
| `general_vqa` | General visual QA | API-based |

### Frame Sampling
| Tool | Description | Backend |
|------|-------------|---------|
| `temporal_sample_frames` | Sample diverse frames temporally | VideoTree |
| `temporal_spatial_sample_frames` | Find frames with specific objects | TStar |

### Detection & Description
| Tool | Description | Backend |
|------|-------------|---------|
| `detect_objects` | Detect specific object categories | YOLO-World |
| `detect_all_objects` | Detect all objects | YOLOE |
| `describe_region` | Describe specific region | DAM |

### Frame & Caption
| Tool | Description | Backend |
|------|-------------|---------|
| `view_frame` | View specific frame(s) | Returns captions |
| `detailed_captioning` | Detailed caption via API | MLLM |

## Output Structure

```
results/<experiment_name>__<model>_videos_<count>_<date>/
├── logging.log           # Full evaluation log
├── result.json           # Complete results with per-video details
├── metrics.csv           # Performance metrics
├── summary.txt           # Human-readable summary
├── accuracy.txt          # Quick accuracy reference
├── experiment_config.yaml
└── videos/               # Per-video outputs
    └── <video_id>/
        ├── frames/       # Sampled frames (PNG)
        ├── llm.log       # Full LLM interaction log
        └── result.json   # Video-specific result
```

## Configuration

### Environment Variables

Create `.env` file with:

```bash
# Required: API key (one of these)
AIML_API_KEY=your_aiml_api_key
# or
OPENAI_API_KEY=your_openai_key

# Optional: Custom API base URL
AIML_BASE_URL=https://api.aimlapi.com/v1
```

### Supported Models

The agent supports any OpenAI-compatible API. Recommended models:

| Model | Provider | Use Case |
|-------|----------|----------|
| `gpt-4o-mini` | OpenAI | Good value, fast |
| `x-ai/grok-4-1-fast-reasoning` | xAI | Fast reasoning |
| `anthropic/claude-4-sonnet` | Anthropic | Strong reasoning |
| `google/gemini-2.5-flash` | Google | Fast, 1M context |

## Extending VideoAgent

### Adding New Tools

1. Create interface class in `tools/interface/`:

```python
from tools.interface_base import Interface, InterfaceCategory

class MyNewTool(Interface):
    NAME = "my_new_tool"
    CATEGORY = InterfaceCategory.DETECTION
    FUNCTIONALITY = "What this tool does"
    
    AGENT_NAME = "my_new_tool"
    AGENT_DESCRIPTION = "Description for agent"
    AGENT_INPUT_SCHEMA = {
        "query": {"type": "str", "required": True, "description": "Input query"}
    }
    
    def __call__(self, video, query, **kwargs):
        # Implementation
        return {"result": "..."}
    
    @classmethod
    def format_output_for_agent(cls, output):
        return output["result"]
```

2. Register in `tools/interface/__init__.py`:

```python
INTERFACE_MAPPING["my_new_tool"] = MyNewTool
```

3. Enable in evaluation script:

```bash
TOOLS="temporal_sample_frames,my_new_tool"
```

## Acknowledgments

- [EgoSchema](https://egoschema.github.io/) - Benchmark dataset
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent framework
- [InternVideo2.5](https://github.com/OpenGVLab/InternVideo) - Video understanding
- [VideoTree](https://github.com/Ziyang412/VideoTree) - Frame sampling
- [TStar](https://github.com/TStar-Labs/TStar) - Temporal-spatial understanding

## License

This project is for research purposes.
