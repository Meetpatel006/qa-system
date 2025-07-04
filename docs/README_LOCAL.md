# Local and Google Colab Setup Guide

This guide explains how to set up and run the Question Answering System locally or on Google Colab.

## Local Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
```bash
git clone <your-repo-url>
cd qa-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Locally

1. Start the application:
```bash
python main.py
```

2. Open your web browser and navigate to:
- Local URL: http://localhost:7860
- Or use the temporary public URL provided by Gradio

## Google Colab Setup

1. Upload the following files to your Google Drive:
   - `main.py`
   - `requirements.txt`

2. Create a new Colab notebook and mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Navigate to your project directory:
```python
%cd /content/drive/MyDrive/path/to/your/project
```

4. Install dependencies:
```python
!pip install -r requirements.txt
```

5. Run the application:
```python
!python main.py
```

6. Click on the public URL provided by Gradio to access the interface

## Troubleshooting

### Common Issues

1. Model Loading Errors:
   - Ensure you have a stable internet connection
   - Check if you have enough disk space
   - Verify that all dependencies are correctly installed

2. Port Already in Use:
   - Change the port number in `main.py`
   - Kill any process using port 7860

### Getting Help

If you encounter any issues:
1. Check the error messages in the console
2. Verify that all prerequisites are met
3. Create an issue in the GitHub repository
