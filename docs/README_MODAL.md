# Question Answering System - Modal Deployment

This project contains a Question Answering system built with Gradio and deployed on Modal, a serverless cloud platform optimized for AI/ML workloads.

Repository: [https://github.com/Meetpatel006/qa-system.git](https://github.com/Meetpatel006/qa-system.git)

## Files Overview

- `main.py` - Original local Gradio implementation
- `modal_deploy.py` - Modal deployment script with GPU-accelerated inference
- `setup_modal.py` - Automated setup script for Modal deployment
- `modal_requirements.txt` - Dependencies for Modal deployment
- `requirements.txt` - Local development dependencies
- `models/question_answering.py` - Core QA model implementation with training options

## Model Architecture

The system uses a fine-tuned DistilBERT model with the following specifications:
- Base model: `distilbert/distilbert-base-uncased`
- Training data: SQuAD dataset (5000 examples subset)
- Training parameters:
  - Batch size: 16
  - Epochs: 3
  - Learning rate: 2e-5
  - Max sequence length: 384
- Supported frameworks: PyTorch and TensorFlow
- Automatic model pushing to HuggingFace Hub

## Quick Start

### Option 1: Automated Setup
Run the setup script that will handle everything for you:

```bash
python setup_modal.py
```

### Option 2: Manual Setup

1. **Install Modal**:
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```bash
   modal setup
   ```
   Follow the instructions to create an account and authenticate.

3. **Deploy the application**:
   ```bash
   modal deploy modal_deploy.py
   ```

## Development

For development with hot-reloading (changes are automatically deployed):

```bash
modal serve modal_deploy.py
```

## Features

- **GPU Acceleration**: Uses T4 GPU for fast inference
- **Auto-scaling**: Scales from 0 to handle any number of requests
- **Cost Effective**: Only pay for actual usage
- **Modern UI**: Beautiful Gradio interface with examples
- **Error Handling**: Fallback to default model if custom model fails
- **Health Monitoring**: Built-in health check endpoint

## Model Information

The system uses the `RedRepter/my_awesome_qa_model` from Hugging Face. If this model is unavailable, it automatically falls back to `distilbert-base-cased-distilled-squad`.

## Deployment Configuration

- **GPU**: T4 (cost-effective for inference)
- **Scaling**: 1 warm container minimum, scales up as needed
- **Timeout**: 5-minute scale-down window
- **Concurrency**: Up to 1000 concurrent requests per container

## Useful Modal Commands

- **Check deployment status**: `modal app list`
- **View logs**: `modal logs qa-system`
- **Stop deployment**: `modal app stop qa-system`
- **Check costs**: Visit your Modal dashboard
- **Shell access**: `modal shell modal_deploy.py`

## Architecture

```
User Request → Modal Load Balancer → Gradio UI (CPU) → QA Model (GPU) → Response
```

The system architecture includes:
1. Front-end: Gradio UI for user interaction
2. Model Backend:
   - DistilBERT-based QA model
   - Support for both PyTorch and TensorFlow
   - Multiple inference methods (pipeline, PyTorch, TensorFlow)
3. Deployment:
   - GPU-accelerated inference
   - Auto-scaling capabilities
   - Resource optimization

## Cost Optimization

- Uses T4 GPUs (most cost-effective for this workload)
- Automatically scales to zero when not in use
- Keeps one container warm for fast response times
- Separates UI and inference for resource efficiency

## Support

For Modal-specific issues, check:
- [Modal Documentation](https://modal.com/docs)
- [Modal Examples](https://github.com/modal-labs/modal-examples)
- [Modal Discord Community](https://discord.gg/modal)

For application issues, check the logs with `modal logs qa-system`.
