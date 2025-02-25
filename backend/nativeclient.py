import cv2
import requests
import time
import random

# Specify your local video file path
VIDEO_PATH = "E:/I made a website that makes websites [ofHGE-85EIA].webm"  # Change this to your video file location

# Predefined colors for some classes
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
    """Return a random color for classes not defined."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

# Get video dimensions and fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video dimensions: {frame_width}x{frame_height} at {fps:.2f} FPS")

# Create a resizable window
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Detection", 1280, 720)

frame_count = 0
api_success_count = 0
api_failure_count = 0

# Process every n-th frame (adjust if needed)
process_every_n_frames = max(1, int(fps / 5))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        continue

    display_frame = frame.copy()

    # Encode frame to send to the Flask backend
    ret_enc, img_encoded = cv2.imencode(".jpg", frame)
    if not ret_enc:
        print("Failed to encode frame.")
        continue

    try:
        # Send frame to Flask backend for YOLO prediction
        response = requests.post(
            "http://localhost:5000/predict",
            files={"image": img_encoded.tobytes()},
            timeout=1.0
        )

        if response.status_code == 200:
            api_success_count += 1
            data = response.json()
            detections = data.get("detections", [])
            inference_time = data.get("inference_time", 0)
            
            # Debug: Print detection info for the first detection if available
            if detections:
                print(f"Frame {frame_count}: Top detection: {detections[0]['class']} with confidence {detections[0]['conf']:.2f}")
            else:
                print(f"Frame {frame_count}: No detections.")
            
            # Draw bounding boxes
            for det in detections:
                try:
                    # Get bounding box coordinates
                    x = int(det["x"])
                    y = int(det["y"])
                    w = int(det["w"])
                    h = int(det["h"])
                    conf = float(det.get("conf", 0))
                    class_name = det.get("class", "unknown")
                    
                    # Debug: Print coordinates
                    print(f"Detection: {class_name}, conf: {conf:.2f}, box: ({x}, {y}, {w}, {h})")
                    
                    # Choose a color and increase thickness for low confidence
                    color = COLORS.get(class_name, get_random_color())
                    thickness = max(2, int(conf * 10))  # Increased thickness
                    
                    # Draw rectangle and label if coordinates are within frame bounds
                    if x >= 0 and y >= 0 and (x + w) < frame_width and (y + h) < frame_height:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                        label = f"{class_name} {conf:.2f}"
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display_frame, (x, y - text_h - 4), (x + text_w, y), color, -1)
                        cv2.putText(display_frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception as e:
                    print(f"Error drawing bbox: {e}")
        else:
            api_failure_count += 1
            print(f"Server error: {response.status_code}, {response.text[:100]}")
    except Exception as e:
        api_failure_count += 1
        print(f"Request error: {e}")

    # Display FPS and API stats on the frame
    cv2.putText(display_frame, f"API Success: {api_success_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the processed frame
    cv2.imshow("YOLO Detection", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite(f"frame_{int(time.time())}.jpg", display_frame)
        print("Frame saved.")

cap.release()
cv2.destroyAllWindows()
print(f"Processing complete. API success: {api_success_count}, API failures: {api_failure_count}")


