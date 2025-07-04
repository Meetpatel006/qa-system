# Question Answering System

A production-ready Question Answering (QA) system that combines the power of Hugging Face's transformers, Modal's cloud infrastructure, and modern web interfaces. This system provides accurate answers to questions based on given context, with support for both local development and cloud deployment.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EjQuv3j1sorDrkch-dGuYYmNJSgthW0N?usp=sharing)

## Quick Links
- [Local Setup Guide](docs/README_LOCAL.md)
- [Modal Deployment Guide](docs/README_MODAL.md)
- [Fine-tuning Guide](docs/README_FINETUNING.md)
- [Repository](https://github.com/Meetpatel006/qa-system.git)

## Key Features

🤖 **Advanced Model Architecture**
- State-of-the-art DistilBERT-based question answering
- Support for both PyTorch and TensorFlow
- Fine-tuned on SQuAD dataset
- Automatic model versioning and HuggingFace Hub integration

☁️ **Cloud Infrastructure**
- GPU-accelerated training with Modal's A10G instances
- Scalable FastAPI-based REST API
- Automatic model versioning and storage
- Comprehensive monitoring and metrics

💻 **Development Experience**
- User-friendly Gradio web interface
- Detailed documentation and examples
- Multiple deployment options (local, cloud, Colab)
- Easy-to-use CLI and API interfaces

## Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/Meetpatel006/qa-system.git
cd qa-system
```

2. **Choose Your Path**
- 🚀 For quick experimentation: [Open in Colab](https://colab.research.google.com/drive/1EjQuv3j1sorDrkch-dGuYYmNJSgthW0N?usp=sharing)
- 💻 For local development: Follow our [Local Setup Guide](docs/README_LOCAL.md)
- ☁️ For cloud deployment: Check our [Modal Deployment Guide](docs/README_MODAL.md)
- 🔧 For model fine-tuning: See our [Fine-tuning Guide](docs/README_FINETUNING.md)

## Project Structure

```bash
├── main.py                 # Gradio web interface
├── deployment/
│   ├── modal_deploy.py     # Modal deployment script
│   └── finetune_modal.py   # Fine-tuning system
├── models/
│   └── question_answering.py  # Core QA implementation
├── docs/
│   ├── README_LOCAL.md     # Local setup guide
│   ├── README_MODAL.md     # Modal deployment guide
│   └── README_FINETUNING.md # Fine-tuning guide
└── requirements/
    ├── requirements.txt    # Local dependencies
    └── modal_requirements.txt # Modal dependencies
```

## System Architecture

The Question Answering system (`models/question_answering.py`) provides:

The system combines multiple components for a robust QA solution:

### Core Components
- **Base Model**: Fine-tuned DistilBERT optimized for QA tasks
- **Training System**: GPU-accelerated fine-tuning with comprehensive monitoring
- **API Layer**: FastAPI-based REST interface for training and inference
- **UI Layer**: Gradio web interface for easy interaction

### Deployment Options
- **Local**: Run everything on your machine with optional GPU support
- **Cloud**: Deploy on Modal's infrastructure with A10G GPUs
- **Hybrid**: Mix local development with cloud training
- **Colab**: Quick experimentation and prototyping

## Usage Examples

### Simple Question Answering
```python
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
answer = qa_model(question=question, context=context)
# Answer: "13" (confidence: 0.98)
```

### Using the REST API
```bash
# Training a model
curl -X POST https://modal-fastapi-url.com/train \
  -H "Content-Type: application/json" \
  -d '{"num_epochs": 3, "batch_size": 16}'

# Running inference
curl -X POST https://modal-fastapi-url.com/inference \
  -H "Content-Type: application/json" \
  -d '{"question": "your question", "context": "your context"}'
```

### Using the Web Interface
1. Start the Gradio server:
```bash
python main.py
```
2. Open http://localhost:7860 in your browser
3. Enter your question and context
4. Get instant answers with confidence scores

## Development

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (optional for local development)
- Modal account (for cloud deployment)
- HuggingFace account (for model pushing)

### Available Tools
- 🔧 Fine-tuning scripts for custom datasets
- 📊 Training monitoring with Tensorboard
- 🚀 Deployment scripts for Modal
- 🧪 Testing utilities and examples

For detailed development instructions, check:
- [Fine-tuning Guide](docs/README_FINETUNING.md)
- [Modal Deployment Guide](docs/README_MODAL.md)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/Meetpatel006/qa-system/issues)
- 💬 [Discussions](https://github.com/Meetpatel006/qa-system/discussions)

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
