# VideoAgent

A multi-tool video understanding agent built with LangGraph for long-form video question answering.

## Overview

VideoAgent is a modular agentic framework that employs a large language model (LLM) as a central controller for perception, decision-making, and action execution. The system adopts a **ReAct-style workflow** for iterative, query-driven evidence gathering, enabling efficient and accurate analysis over long videos without relying on predefined workflows or heavy pre-computation.

### Key Features

- **Iterative Reasoning**: ReAct-based workflow that progressively refines hypotheses through temporally grounded evidence gathering
- **Flexible Tool Orchestration**: Dynamic coordination of specialized vision experts via a unified interface for problem-oriented temporal localization and visual perception
- **Hierarchical Memory**: Structured memory organization (Task Context → Video Memory → Tool History → Reasoning State) for coherent long-term reasoning
- **Multi-GPU Support**: Centralized tool server with GPU-aware resource management for parallel processing
- **Model-Aware Caching**: Efficient caching for captions and descriptions to reduce redundant computation

### Performance

Evaluated on the [EgoSchema](https://egoschema.github.io/) benchmark (500 egocentric videos, ~3 minutes each):

| Method | Accuracy | Avg. Frames |
|--------|----------|-------------|
| GPT-4V | 63.5% | - |
| InternVideo2.5 | 63.5% | 128 |
| Tarsier | 68.6% | 128 |
| **VideoAgent (Ours)** | **70.8%** | **22.5** |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VideoAgent                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      LangGraph Agent (ReAct)                          │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────────┐   │  │
│  │  │    Agent    │ ─▶ │    Tools    │ ─▶ │ Force Answer (if needed) │   │  │
│  │  │     Node    │ ◀─ │     Node    │    │                          │   │  │
│  │  └─────────────┘    └─────────────┘    └──────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     │                                        │
│  ┌──────────────────────────────────┼────────────────────────────────────┐  │
│  │                    Hierarchical Memory                                │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌───────────────┐   │  │
│  │  │Task Context  │ │ Video Memory │ │Tool History│ │Reasoning State│   │  │
│  │  │(Q + Choices) │ │(Frames+Caps) │ │  (Q&A Log) │ │ (Hypotheses)  │   │  │
│  │  └──────────────┘ └──────────────┘ └────────────┘ └───────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                     │                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Tool Manager                                  │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │   Q&A    │ │Retrieval │ │Observation│ │Detection │ │  ...     │    │  │
│  │  │  Tools   │ │  Tools   │ │   Tools  │ │  Tools   │ │          │    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
VideoAgent/
├── video_agent_tools/              # Main agent package
│   ├── cli.py                      # Command-line interface
│   ├── evaluation.py               # Batch evaluation framework
│   ├── graph.py                    # LangGraph agent (ReAct workflow)
│   ├── prompts.py                  # Agent prompts and templates
│   ├── state.py                    # State & memory definitions
│   ├── tools.py                    # Tool manager
│   ├── resource_management/        # Multi-GPU resource management
│   │   ├── gpu_manager.py          # GPU allocation & scheduling
│   │   ├── tool_server.py          # Centralized tool server
│   │   └── tool_client.py          # Worker tool client
│   └── utils/
│       ├── logging.py              # Structured logging
│       ├── tool_cache.py           # Model-aware caching
│       └── video.py                # Video processing utilities
│
├── tools/                          # Tool interface layer
│   ├── interface_base.py           # Base Interface class
│   ├── interface/                  # Tool interfaces (see below)
│   └── models/                     # Model weights (gitignored)
│
├── configs/                        # Configuration files
├── scripts/template/eval.sh        # Evaluation script template
├── data/EgoSchema_test/            # Dataset directory
├── requirements.txt
└── .env.example
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yuanyunchen/VideoAgent.git
cd VideoAgent
```

### 2. Create Environment

```bash
conda create -n videoagent python=3.10
conda activate videoagent
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For local tools (optional, see Model Setup below)
pip install transformers accelerate ultralytics
```

### 4. Configure API Key

```bash
cp .env.example .env
# Edit .env:
# AIML_API_KEY=your_api_key_here
```

### 5. Prepare Dataset

```bash
# Download EgoSchema videos from https://egoschema.github.io/
# Place videos in data/EgoSchema_test/videos/
```

## Tool Interface System

VideoAgent uses a **decoupled architecture** separating the Interface Layer from the Model Layer. The agent interacts with abstract interfaces, allowing seamless model updates without changing agent logic.

### Interface Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Agent Layer                              │
│   (Sees only tool descriptions, input schemas, formatted output) │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Interface Layer (tools/interface/)          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ AGENT_NAME      │  │AGENT_DESCRIPTION│  │ AGENT_INPUT_    │   │
│  │AGENT_DESCRIPTION│  │ AGENT_INPUT_    │  │ SCHEMA          │   │
│  │ format_output() │  │ SCHEMA          │  │ format_output() │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Model Layer (tools/models/)                 │
│  InternVideo2.5 | VideoTree | TStar | YOLO-World | DAM | ...     │
└──────────────────────────────────────────────────────────────────┘
```

### Available Tools

| Category | Tool | Interface | Backend Model |
|----------|------|-----------|---------------|
| **Q&A** | `internvideo_general_qa` | `InternVideoGeneralQA` | InternVideo2.5-Chat-8B |
| | `internvideo_description` | `InternVideoDescription` | InternVideo2.5-Chat-8B |
| | `general_vqa` | `GeneralVQA` | API-based MLLM |
| | `temporal_spatial_qa` | `TStarTemporalSpatialQA` | TStar + LLM |
| **Retrieval** | `temporal_sample_frames` | `VideoTreeSampling` | VideoTree (CLIP) |
| | `temporal_spatial_sample_frames` | `TStarSampling` | TStar (MobileCLIP) |
| **Observation** | `view_frame` | `ViewFrame` | - |
| | `caption_image` | `OmniCaptionerCaptioning` | OmniCaptioner |
| | `detailed_captioning` | `APICaptioning` | API-based MLLM |
| | `describe_region` | `DAMDescription` | DAM (Describe Anything) |
| **Detection** | `detect_objects` | `YOLOWorldDetection` | YOLO-World |
| | `detect_all_objects` | `YOLOEPromptFreeDetection` | YOLOE |

## Model Setup

Local tools require downloading model weights to `tools/models/`. Each tool interface specifies its required model.

### Required Models

#### InternVideo2.5 (for `internvideo_general_qa`, `internvideo_description`)

```bash
# Download from HuggingFace
cd tools/models
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVideo2_5_Chat_8B
```

#### OmniCaptioner (for `caption_image`)

```bash
cd tools/models
git clone https://huggingface.co/U4R/OmniCaptioner
```

#### VideoTree (for `temporal_sample_frames`)

VideoTree uses CLIP embeddings. The interface automatically downloads CLIP weights on first use.

#### TStar (for `temporal_spatial_sample_frames`, `temporal_spatial_qa`)

```bash
cd tools/models
git clone https://github.com/TStar-Labs/TStar

# Download MobileCLIP weights
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt
```

#### YOLO-World (for `detect_objects`)

```bash
pip install ultralytics
# Weights are downloaded automatically on first use
```

#### DAM (for `describe_region`)

```bash
cd tools/models
git clone https://github.com/tsinghua-fib-lab/Describe-Anything-Model describe-anything

# Follow DAM installation instructions in its README
```

### API-Only Mode

If you don't want to set up local models, you can use API-only tools:

```bash
TOOLS="general_vqa,view_frame,detailed_captioning"
CAPTIONER="gpt-4o-mini"  # Use API for captioning
```

## Running Experiments

### Using Evaluation Script (Recommended)

```bash
# Copy template
cp scripts/template/eval.sh scripts/my_experiment.sh

# Edit configuration
vim scripts/my_experiment.sh

# Key settings:
AGENT_MODEL="x-ai/grok-4-1-fast-reasoning"
TOOLS="internvideo_general_qa,temporal_sample_frames,view_frame,detect_objects"
COUNT=100               # Number of videos (0 = all)
MAX_TOOL_CALLS=20       # Max iterations
INITIAL_FRAMES=5        # Initial context

# Run
chmod +x scripts/my_experiment.sh
./scripts/my_experiment.sh
```

### Using CLI

```bash
python -m video_agent_tools.cli \
    --model "gpt-4o-mini" \
    --tools "temporal_sample_frames,view_frame,general_vqa" \
    --max-tool-calls 15 \
    --max-videos 10 \
    --annotation-file data/EgoSchema_test/annotations.json \
    --video-dir data/EgoSchema_test/videos \
    --experiment-name "test_run"
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | LLM model for agent reasoning | `gpt-4o-mini` |
| `--tools` | Comma-separated tool list | See eval.sh |
| `--max-tool-calls` | Max tool calls per video | `10` |
| `--max-parallel-tools` | Max tools per turn | `3` |
| `--initial-frames` | Frames to caption initially | `5` |
| `--captioner` | Captioner (`omni-captioner` or API model) | `omni-captioner` |
| `--num-workers` | Parallel workers | `1` |
| `--max-videos` | Number of videos (-1 = all) | `-1` |

## Output Structure

```
results/<experiment_name>__<model>_videos_<count>_<date>/
├── logging.log           # Full evaluation log
├── result.json           # Complete results
├── metrics.csv           # Performance metrics
├── summary.txt           # Human-readable summary
├── accuracy.txt          # Quick accuracy
├── experiment_config.yaml
└── videos/
    └── <video_id>/
        ├── frames/       # Sampled frames (PNG)
        ├── llm.log       # Full LLM interaction log
        └── result.json   # Per-video result
```

## Extending VideoAgent

### Adding a New Tool

1. **Create interface class** in `tools/interface/`:

```python
from tools.interface_base import Interface, InterfaceCategory

class MyNewTool(Interface):
    NAME = "my_new_tool"
    CATEGORY = InterfaceCategory.DETECTION
    FUNCTIONALITY = "What this tool does"
    
    # Agent-facing metadata
    AGENT_NAME = "my_new_tool"
    AGENT_DESCRIPTION = "Description shown to agent"
    AGENT_INPUT_SCHEMA = {
        "query": {"type": "str", "required": True, "description": "Input query"},
        "num_results": {"type": "int", "required": False, "default": 5}
    }
    
    def initialize(self):
        # Load model weights
        self.model = load_model("tools/models/my_model")
    
    def __call__(self, video, query, num_results=5, **kwargs):
        # Execute tool
        result = self.model.process(video, query)
        return {"result": result, "count": len(result)}
    
    @classmethod
    def format_output_for_agent(cls, output):
        # Format output as text for agent consumption
        return f"Found {output['count']} results: {output['result']}"
```

2. **Register** in `tools/interface/__init__.py`:

```python
from tools.interface.my_tool import MyNewTool

INTERFACE_MAPPING["my_new_tool"] = MyNewTool
```

3. **Enable** in evaluation:

```bash
TOOLS="temporal_sample_frames,my_new_tool"
```

## Configuration

### Environment Variables

```bash
# .env file
AIML_API_KEY=your_api_key      # Required: API key for LLM
AIML_BASE_URL=https://api.aimlapi.com/v1  # Optional: API endpoint

# Or use OpenAI directly
OPENAI_API_KEY=your_openai_key
```

### Supported LLM Models

| Model | Provider | Notes |
|-------|----------|-------|
| `gpt-4o-mini` | OpenAI | Good value |
| `gpt-4o` | OpenAI | Best quality |
| `x-ai/grok-4-1-fast-reasoning` | xAI | Fast reasoning |
| `anthropic/claude-4-sonnet` | Anthropic | Strong reasoning |
| `google/gemini-2.5-flash` | Google | 1M context |

## Acknowledgments

- [EgoSchema](https://egoschema.github.io/) - Benchmark dataset
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent framework
- [InternVideo2.5](https://github.com/OpenGVLab/InternVideo) - Video understanding
- [VideoTree](https://github.com/Ziyang412/VideoTree) - Frame sampling
- [TStar](https://github.com/TStar-Labs/TStar) - Temporal-spatial understanding
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World) - Open-vocabulary detection
- [DAM](https://github.com/tsinghua-fib-lab/Describe-Anything-Model) - Region description

## License

This project is for research purposes.
