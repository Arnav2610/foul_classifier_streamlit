# Soccer Foul Classifier App

This Streamlit application analyzes soccer videos to classify incidents as either flops/dives or legitimate fouls (with warnings, yellow cards, or red cards).

## Features

- Upload soccer video clips for analysis
- Automatic pose detection using YOLOv8
- Classification of soccer incidents into four categories:
  - Flop/Dive
  - Foul (Warning)
  - Foul (Yellow Card)
  - Foul (Red Card)
- Visualization of pose detection results
- Frame-by-frame analysis with probability distribution

## Requirements

- Python 3.8+
- Required Python packages (listed in requirements.txt)

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have the model files:
   - `soccer_classifier.pkl` (trained classifier)
   - `yolov8n-pose.pt` (will be downloaded automatically on first run)

## Running the app

Run the Streamlit app with:

```
streamlit run app.py
```

The app will be available at http://localhost:8501 in your web browser.

## How to use

1. Upload a soccer video clip using the file uploader
2. Click "Analyze Video" to classify the incident
3. Optionally, click "Generate Pose Visualization" to see the pose detection visualization
4. View the results in the respective tabs

## How it works

1. The application uses YOLOv8 for pose detection in each frame
2. Features are extracted from the detected skeletons (velocity, acceleration, torso angle, etc.)
3. A pre-trained Random Forest classifier predicts the type of incident
4. Results are aggregated across frames to determine the final classification 