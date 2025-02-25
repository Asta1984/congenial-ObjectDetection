import cv2
import requests
import yt_dlp
import numpy as np
import time
import json
import random

# YouTube video URL - you can easily change this
#YOUTUBE_URL = "https://youtu.be/5YuQQwLGTxA?si=QPHdDJ_knAkUfVUb"
YOUTUBE_URL = "https://www.youtube.com/watch?v=1EiC9bvVGnk"  # Street scene with cars/people

# Colors for different classes (to make visualization better)
COLORS = {
    'person': (0, 255, 0),      # Green
    'car': (0, 0, 255),         # Red
    'bicycle': (255, 0, 0),     # Blue
    'motorbike': (255, 255, 0), # Cyan
    'bus': (255, 0, 255),       # Magenta
    'truck': (0, 255, 255),     # Yellow
    'dog': (255, 128, 0),       # Orange
    'cat': (128, 0, 255),       # Purple
}

def get_random_color():
    """Generate a random color for classes not in our predefined list"""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def get_youtube_stream(url):
    """Extracts the direct video stream URL using yt-dlp"""
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']  # Get the direct video URL

print(f"Processing YouTube video: {YOUTUBE_URL}")
print("Getting YouTube stream URL...")
VIDEO_SOURCE = get_youtube_stream(YOUTUBE_URL)
print(f"Got stream URL (truncated): {VIDEO_SOURCE[:50]}...")

# Use OpenCV to directly capture the video stream
print("Opening video stream...")
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
else:
    print("Video stream opened successfully!")

# Get video properties for debugging
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video dimensions: {frame_width}x{frame_height}")

# Create resizable window
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 1280, 720)

frame_count = 0
start_time = time.time()
fps = 0
api_success_count = 0
api_failure_count = 0

# For tracking detections
last_detections = []
detection_history = []  # Store last 5 frames of detections for smoothing
smooth_factor = 3  # Number of frames to average

while True:
    # Read frame
    ret, frame = cap.read()
    
    if not ret:
        print("End of stream or error reading frame")
        break
    
    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        # Print stats every second
        print(f"FPS: {fps:.2f}, API Success: {api_success_count}, API Failures: {api_failure_count}")
    
    # Skip frames if we're running too slow (optional)
    # if frame_count % 2 != 0:  # Process every other frame
    #    continue
    
    # Create a copy of the frame for drawing
    display_frame = frame.copy()
    
    # Encode frame to send to Flask
    _, img_encoded = cv2.imencode(".jpg", frame)
    
    try:
        # Send frame to Flask backend
        response = requests.post(
            "http://localhost:5000/predict", 
            files={"image": img_encoded.tobytes()},
            timeout=1.0  # Add timeout to prevent hanging
        )
        
        if response.status_code == 200:
            api_success_count += 1
            response_data = response.json()
            detections = response_data.get("detections", [])
            inference_time = response_data.get("inference_time", 0)
            
            # Update detection history for smoothing
            detection_history.append(detections)
            if len(detection_history) > smooth_factor:
                detection_history.pop(0)
            
            # Log detection count periodically
            if frame_count % 30 == 0:  # Log every ~30 frames
                print(f"Received {len(detections)} detections, inference time: {inference_time:.3f}s")
                if len(detections) > 0:
                    print(f"Top detection: {detections[0]['class']} with confidence {detections[0]['conf']:.2f}")
            
            # Store for next frame
            last_detections = detections
            
            # Draw bounding boxes - using all detections from current frame
            for det in detections:
                try:
                    x, y, w, h = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
                    conf = float(det.get("conf", 0))
                    class_name = det.get("class", "unknown")
                    
                    # Get color for this class
                    color = COLORS.get(class_name, get_random_color())
                    
                    # Ensure coordinates are valid
                    if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w < display_frame.shape[1] and y + h < display_frame.shape[0]:
                        # Draw bounding box with thicker line for higher confidence
                        thickness = max(1, int(conf * 5))
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                        
                        # Draw filled background for label text
                        label = f"{class_name} {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(display_frame, (x, y - text_height - 5), (x + text_width, y), color, -1)
                        
                        # Draw label text in white
                        cv2.putText(display_frame, label, (x, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as bbox_err:
                    print(f"Error drawing bbox: {bbox_err}")
        else:
            api_failure_count += 1
            print(f"Server error: {response.status_code}")
            print(f"Response content: {response.text[:100]}")
            
    except requests.exceptions.RequestException as e:
        api_failure_count += 1
        print(f"Request error: {e}")
        
        # If API fails, use last detections (with faded appearance)
        for det in last_detections:
            try:
                x, y, w, h = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
                class_name = det.get("class", "unknown")
                
                # Use a faded color to indicate these are from previous frame
                color = COLORS.get(class_name, (128, 128, 128))
                faded_color = tuple([c//2 for c in color])  # Make color lighter
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), faded_color, 1)
                cv2.putText(display_frame, f"{class_name} (prev)", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, faded_color, 1)
            except Exception as e:
                pass
    
    # Display FPS and inference info
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display_frame, f"API: {api_success_count}/{api_success_count+api_failure_count}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("YOLO Detection", display_frame)

    # Break loop on 'q' key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):  # Save current frame on 's' key press
        cv2.imwrite(f"frame_{int(time.time())}.jpg", display_frame)
        print("Frame saved")

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Program finished. API success: {api_success_count}, API failures: {api_failure_count}")