# Question Answering System

This is a Question Answering (QA) system built using Hugging Face's transformers library and Gradio for the user interface. The system allows users to input a question and context, and it returns relevant answers along with confidence scores.

## Features

- Question answering using transformer-based models
- User-friendly web interface using Gradio
- Support for context-based question answering
- Confidence score for each answer
- Pre-loaded example questions and contexts
- Support for both local and cloud deployment

## Project Structure

```
├── main.py              # Main application with Gradio interface
├── modal_deploy.py      # Modal deployment configuration
├── requirements.txt     # Python dependencies for local setup
└── modal_requirements.txt  # Dependencies for Modal deployment
```

## Getting Started

For detailed setup instructions, please refer to:
- [Local/Colab Setup](docs/README_LOCAL.md) - For running the system locally or on Google Colab
- [Modal Deployment](docs/README_MODAL.md) - For deploying the system using Modal

## Example Usage

```python
Question: "How many programming languages does BLOOM support?"
Context: "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
Answer: "13"
```

## License

This project is open-source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
