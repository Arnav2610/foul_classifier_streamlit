import os
import tempfile
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
import torch

# Set page config - hide sidebar
st.set_page_config(
    page_title="Soccer Foul Classifier",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme with black background and sidebar hidden
st.markdown("""
<style>
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Dark theme styling */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    
    /* Header styling */
    .main-header {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .sub-header {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Verdict boxes with minimal design */
    .verdict-box {
        padding: 15px;
        border-radius: 4px;
        font-weight: 600;
        margin: 15px 0;
        border-left: 3px solid;
    }
    
    .verdict-flop {
        background-color: #132a13;
        color: #4ade80;
        border-color: #22c55e;
    }
    
    .verdict-warning {
        background-color: #2b2113;
        color: #fbbf24;
        border-color: #eab308;
    }
    
    .verdict-yellow {
        background-color: #2b2312;
        color: #facc15;
        border-color: #eab308;
    }
    
    .verdict-red {
        background-color: #2a1314;
        color: #f87171;
        border-color: #ef4444;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #0ea5e9;
        color: white;
        font-weight: 500;
        border: none;
        border-radius: 4px;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #0c2231;
        color: #7dd3fc;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    
    /* Card container for dark theme */
    .card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 6px;
        border: 1px solid #2d3748;
        margin: 15px 0;
    }
    
    /* Section headers */
    .section-header {
        color: #e2e8f0;
        font-size: 1.4rem;
        margin: 10px 0;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        margin-top: 40px;
        padding-top: 10px;
        border-top: 1px solid #2d3748;
        color: #94a3b8;
        font-size: 0.8rem;
        text-align: center;
    }
    
    /* About section */
    .about-section {
        background-color: #1e1e1e;
        padding: 25px;
        border-radius: 6px;
        margin-top: 30px;
        border-top: 1px solid #2d3748;
    }
    
    .about-header {
        color: #e2e8f0;
        font-size: 1.2rem;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    /* Clean up other Streamlit elements for dark theme */
    .stProgress > div > div > div > div {
        background-color: #0ea5e9;
    }
    
    /* Style for the dataframe in dark mode */
    .stDataFrame {
        background-color: #1e1e1e;
    }
    
    div[data-testid="stDecoration"] {
        display: none;
    }
    
    /* Make file uploader visible in dark theme */
    .stFileUploader > div > label {
        color: white;
    }
    
    .stFileUploader > div {
        color: white;
        background-color: #1e1e1e;
        border: 1px dashed #4a5568;
    }
    
    /* Tab styling for dark theme */
    button[data-baseweb="tab"] {
        color: #a0aec0;
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0ea5e9;
    }
    
    /* Make tooltips and other overlays visible on dark background */
    div[data-stale="false"] {
        color: #1e1e1e;
    }
</style>
""", unsafe_allow_html=True)

# Define the same functions as in train.py for feature extraction
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n-pose.pt')

@st.cache_resource
def load_classifier():
    return joblib.load('soccer_classifier.pkl')

def extract_features(video_path):
    """
    Extracts features from a video using multi-skeleton detection.
    """
    model = load_yolo_model()
    video = cv2.VideoCapture(video_path)
    previous_keypoints = None
    previous_time = None
    features = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    frame_count = 0
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Run YOLO pose estimation on the frame
        results = model.predict(frame, conf=0.3, show=False, verbose=False)
        current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # time in seconds
        
        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # Gather all skeletons (each must have at least 13 keypoints)
            current_skeletons = []
            for skeleton in results[0].keypoints.xy:
                skel = skeleton.tolist()
                if len(skel) < 13:
                    continue
                current_skeletons.append(skel)
            
            # If previous frame data exists, compute features for each skeleton that can be matched
            if previous_keypoints is not None and current_skeletons:
                dt = current_time - previous_time
                if dt > 0:
                    num_skel = min(len(previous_keypoints), len(current_skeletons))
                    for i in range(num_skel):
                        prev_skel = previous_keypoints[i]
                        curr_skel = current_skeletons[i]
                        # Compute per-keypoint vertical velocity (using y-coordinate)
                        velocities = [(curr_skel[j][1] - prev_skel[j][1]) / dt for j in range(len(curr_skel))]
                        # Compute a rough acceleration (using difference of velocities)
                        accelerations = [(velocities[j] - ((prev_skel[j][1] - curr_skel[j][1]) / dt)) / dt for j in range(len(curr_skel))]
                        # Compute torso angle using keypoints 5 (shoulder) and 11 (hip)
                        shoulder = curr_skel[5]
                        hip = curr_skel[11]
                        torso_angle = np.arctan2(hip[1] - shoulder[1], hip[0] - shoulder[0]) * (180 / np.pi)
                        # Detect contact (if adjacent keypoints are too close)
                        contact_detected = (euclidean_distance(curr_skel[5], curr_skel[6]) < 50 or
                                            euclidean_distance(curr_skel[11], curr_skel[12]) < 50)
                        reaction_time = dt
                        
                        # Compute average values over keypoints for velocity and acceleration
                        feature_vector = [
                            np.mean(velocities),
                            np.mean(accelerations),
                            torso_angle,
                            int(contact_detected),
                            reaction_time
                        ]
                        # Append feature vector
                        features.append(feature_vector)
            
            # Update previous frame's skeletons and time
            previous_keypoints = current_skeletons
            previous_time = current_time
        
        # Update progress bar
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        progress_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    progress_text.empty()
    video.release()
    return features

def process_video(video_path):
    # Extract features from the video
    features = extract_features(video_path)
    if not features:
        st.error("No features could be extracted from the video. Make sure it contains visible people.")
        return None
    
    # Convert features to DataFrame
    feature_columns = ['velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time']
    df = pd.DataFrame(features, columns=feature_columns)
    
    # Load the classifier and make predictions
    clf = load_classifier()
    
    # Make predictions for each frame
    predictions = clf.predict(df)
    probabilities = clf.predict_proba(df)
    
    # Map numeric labels to class names
    class_names = {
        0: "Flop/Dive",
        1: "Foul (Warning)",
        2: "Foul (Yellow Card)",
        3: "Foul (Red Card)"
    }
    
    # Count predictions for each class
    prediction_counts = np.bincount(predictions.astype(int), minlength=4)
    
    # Calculate average probability for each class
    avg_probabilities = np.mean(probabilities, axis=0)
    
    # Get the most frequent prediction
    most_frequent = np.argmax(prediction_counts)
    
    return {
        'predictions': predictions,
        'prediction_counts': prediction_counts,
        'class_names': class_names,
        'most_frequent': most_frequent,
        'avg_probabilities': avg_probabilities
    }

def create_annotated_video(input_path, output_path):
    """Creates an annotated video with pose skeleton visualization"""
    model = load_yolo_model()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error(f"Error opening video")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'XVID'),
                          fps,
                          (frame_width, frame_height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_text = st.empty()
    progress_bar = st.progress(0)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run pose estimation and use YOLO's built-in plotting method
        results = model.predict(frame, conf=0.3, show=False)
        annotated_frame = results[0].plot()  # This plots all skeletons in the frame
        out.write(annotated_frame)
        
        # Update progress bar
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        progress_text.text(f"Processing frame {frame_count}/{total_frames}")
    
    progress_text.empty()
    cap.release()
    out.release()
    return output_path

def get_verdict_class(category):
    if category == 0:
        return "verdict-box verdict-flop"  # Flop/Dive
    elif category == 1:
        return "verdict-box verdict-warning"  # Warning
    elif category == 2:
        return "verdict-box verdict-yellow"  # Yellow card
    else:
        return "verdict-box verdict-red"  # Red card

# Main app
def main():
    # Main content
    st.markdown('<h1 class="main-header">⚽ Soccer Foul Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a soccer video to analyze for flop or legitimate foul</p>', unsafe_allow_html=True)
    
    # Simple file uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display video
        st.video(temp_path)
        
        # Create simplified tabs
        tab1, tab2 = st.tabs(["Analysis", "Pose Visualization"])
        
        with tab1:
            analyze_button = st.button("Analyze Video", use_container_width=True)
            
            if analyze_button:
                with st.spinner("Analyzing video..."):
                    results = process_video(temp_path)
                    
                    if results:
                        # Display the most frequent prediction
                        most_frequent_class = results['class_names'][results['most_frequent']]
                        verdict_class = get_verdict_class(results['most_frequent'])
                        
                        st.markdown(f'<div class="{verdict_class}">Verdict: {most_frequent_class}</div>', unsafe_allow_html=True)
                        
                        # Create a simple table for prediction distribution
                        st.markdown('<h3 class="section-header">Prediction Distribution</h3>', unsafe_allow_html=True)
                        
                        # Calculate percentages
                        total_frames = sum(results['prediction_counts'])
                        percentages = [count/total_frames*100 for count in results['prediction_counts']]
                        
                        # Create a DataFrame for the results
                        results_df = pd.DataFrame({
                            'Class': [results['class_names'][i] for i in range(4)],
                            'Frame Count': results['prediction_counts'],
                            'Percentage (%)': [f"{p:.2f}" for p in percentages]
                        })
                        
                        # Display as a table
                        st.dataframe(
                            results_df,
                            column_config={
                                "Percentage (%)": st.column_config.ProgressColumn(
                                    "Percentage (%)",
                                    format="%.2f%%",
                                    min_value=0,
                                    max_value=100,
                                ),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Simple info box
                        st.markdown('<div class="info-box">The final verdict is determined by the most frequent prediction across all analyzed frames.</div>', unsafe_allow_html=True)
        
        with tab2:
            visualize_button = st.button("Generate Pose Visualization", use_container_width=True)
            
            if visualize_button:
                with st.spinner("Generating visualization..."):
                    output_path = os.path.join(temp_dir, "annotated_" + uploaded_file.name)
                    annotated_video = create_annotated_video(temp_path, output_path)
                    
                    if annotated_video:
                        st.markdown('<h3 class="section-header">Pose Detection</h3>', unsafe_allow_html=True)
                        st.success("Visualization complete")
                        st.video(annotated_video)
                        
                        st.markdown('<div class="info-box">This visualization shows the pose detection used to analyze player movements.</div>', unsafe_allow_html=True)
                    else:
                        st.error("Failed to generate visualization.")
    
    # About and classification info at the bottom instead of side column
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="about-header">About This App</h3>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #a0aec0; font-size: 0.9rem;">
        This app analyzes soccer videos to determine if an incident is a legitimate foul or 
        a dive/flop by analyzing player movements. Using AI pose detection and machine learning,
        it extracts key features from player movements including velocity, acceleration, torso angle,
        player contact, and reaction time.
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 class="about-header">Classification Categories</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div style="color: #a0aec0; font-size: 0.9rem;">
        <p style="margin-bottom: 5px;"><span style="color: #4ade80; font-weight: 600;">• Flop/Dive:</span> Player simulating a foul</p>
        <p style="margin-bottom: 5px;"><span style="color: #fbbf24; font-weight: 600;">• Warning Foul:</span> Minor foul, verbal warning</p>
        <p style="margin-bottom: 5px;"><span style="color: #facc15; font-weight: 600;">• Yellow Card Foul:</span> Moderate severity</p>
        <p style="margin-bottom: 5px;"><span style="color: #f87171; font-weight: 600;">• Red Card Foul:</span> Serious foul play</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">Powered by YOLOv8 & Machine Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 