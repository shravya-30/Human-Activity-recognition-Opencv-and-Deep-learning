# Setup Instructions

This guide will help you set up the Human Activity Recognition project on your local machine.

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/human-activity-recognition.git
cd human-activity-recognition
```

### 2. Create Virtual Environment

Using venv (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Or using conda:

```bash
conda create -n activity-recognition python=3.8
conda activate activity-recognition
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including:
- TensorFlow/Keras for deep learning
- OpenCV for video processing
- NumPy and Pandas for data handling
- Jupyter for interactive notebooks

### 4. Download Pre-trained Model

Download the pre-trained model from the project repository and place it in the `models/` directory:

```bash
# Create models directory if it doesn't exist
mkdir -p models

# Place your pretrained_model.h5 here
```

### 5. Verify Installation

Test the installation by running the demo notebook:

```bash
jupyter notebook notebooks/activity_recognition_demo.ipynb
```

Or test with a Python script:

```bash
python -c "from src.video_processor import VideoProcessor; print('✓ Installation successful')"
```

## File Structure Setup

Ensure your project has the following structure:

```
human-activity-recognition/
├── data/
│   └── sample_videos/          # Add your test videos here
├── models/
│   └── pretrained_model.h5     # Download and place here
├── src/
│   ├── __init__.py
│   ├── video_processor.py
│   ├── activity_classifier.py
│   └── utils.py
├── notebooks/
│   └── activity_recognition_demo.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Quick Start

### Using Python Script

```python
from src.activity_classifier import ActivityRecognizer
from src.video_processor import VideoProcessor

# Initialize components
processor = VideoProcessor()
recognizer = ActivityRecognizer(model_path='models/pretrained_model.h5')

# Process video and predict
frames = processor.preprocess_video('data/sample_videos/example.mp4')
predictions = recognizer.predict(frames, return_top_k=3)

# Display results
for activity, confidence in predictions:
    print(f"{activity}: {confidence:.2%}")
```

### Using Jupyter Notebook

```bash
jupyter notebook notebooks/activity_recognition_demo.ipynb
```

The notebook provides a step-by-step walkthrough of the entire pipeline.

## Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:** Ensure you've activated the virtual environment and run:
```bash
pip install --upgrade tensorflow
```

### Issue: "No such file or directory: 'models/pretrained_model.h5'"

**Solution:** Download the pre-trained model and place it in the `models/` directory.

### Issue: OpenCV errors with video processing

**Solution:** Install opencv-contrib-python:
```bash
pip install --upgrade opencv-contrib-python
```

## Environment Variables (Optional)

Create a `.env` file for configuration:

```
MODEL_PATH=models/pretrained_model.h5
DATA_PATH=data/
OUTPUT_PATH=results/
BATCH_SIZE=32
```


---


