# Fine-tuning Guide for Question Answering System

This guide provides detailed instructions for fine-tuning the Question Answering model using our Modal-based training infrastructure.

## Overview

The fine-tuning system is built on Modal's cloud infrastructure, providing:
- GPU-accelerated training (NVIDIA A10G)
- REST API for training and inference
- Automatic model versioning
- Training metrics monitoring
- HuggingFace Hub integration

## Prerequisites

- Modal account (Sign up at [modal.com](https://modal.com))
- HuggingFace account and API token
- Python 3.10 or higher
- Git (for version control)

## Setup

1. **Install Modal**:
```bash
pip install modal
```

2. **Configure Modal**:
```bash
modal token new
modal login
```

3. **Set Environment**:
Configure your HuggingFace token in `deployment/finetune_modal.py`:
```python
HUGGINGFACE_TOKEN = "your_huggingface_token_here"
```

## Fine-tuning Configuration

The system supports the following configuration parameters:

```python
# Default configuration
MODEL_NAME = "distilbert/distilbert-base-uncased"  # Base model
NUM_SAMPLES = 5000        # Training samples
NUM_EPOCHS = 3           # Training epochs
BATCH_SIZE = 16          # Batch size
LEARNING_RATE = 2e-5     # Learning rate
PUSH_TO_HUB = True       # Push to HuggingFace Hub
```

## Training Methods

### 1. Using the REST API

Start the API server:
```bash
modal serve deployment/finetune_modal.py
```

Send a training request:
```bash
curl -X POST https://modal-fastapi-url.com/train -H "Content-Type: application/json" -d '{
    "model_name": "distilbert/distilbert-base-uncased",
    "num_samples": 5000,
    "num_epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "push_to_hub": true
}'
```

### 2. Using CLI

Run training directly:
```bash
python -c "from deployment.finetune_modal import train_locally; train_locally()"
```

## Training Process

1. **Data Preparation**
   - Loads SQuAD dataset
   - Preprocesses text data
   - Tokenizes input
   - Creates train/test split

2. **Training Loop**
   - GPU acceleration with A10G
   - Progress monitoring
   - Automatic checkpointing
   - Evaluation on test set

3. **Model Storage**
   - Saves to Modal volume
   - Pushes to HuggingFace Hub (optional)
   - Maintains version history

## Monitoring Training

The system provides comprehensive monitoring through:

1. **Console Output**
   - Training progress
   - Loss metrics
   - Accuracy scores
   - Evaluation results

2. **Tensorboard Integration**
   - Real-time loss curves
   - Accuracy metrics
   - Model parameters
   - Learning rate tracking

## Inference

### Using the API

```bash
curl -X POST https://modal-fastapi-url.com/inference -H "Content-Type: application/json" -d '{
    "question": "Your question here",
    "context": "Your context text here"
}'
```

### Using CLI

```bash
python -c "from deployment.finetune_modal import inference_locally; inference_locally()"
```

## Best Practices

1. **Model Selection**
   - Start with `distilbert-base-uncased` for fast iteration
   - Consider larger models for production use
   - Test different architectures

2. **Training Parameters**
   - Adjust batch size based on GPU memory
   - Start with learning rate of 2e-5
   - Monitor training loss for stability

3. **Data Handling**
   - Use appropriate subset size for testing
   - Ensure quality of training data
   - Validate input preprocessing

## Troubleshooting

1. **Out of Memory Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Choose a smaller model

2. **Training Issues**
   - Check learning rate
   - Verify data preprocessing
   - Monitor loss curves

3. **Deployment Problems**
   - Verify Modal setup
   - Check API endpoints
   - Validate request format

## Support

For additional help:
1. Check Modal docs: [docs.modal.com](https://docs.modal.com)
2. HuggingFace docs: [huggingface.co/docs](https://huggingface.co/docs)
3. Create GitHub issues for bugs
4. Join our Discord community for discussions
