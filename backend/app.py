from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from torchvision import transforms
from flask_cors import CORS
from utils import cellboxes_to_boxes
from model import Yolov1
import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load YOLO model
print("Loading YOLO model...")
model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to("cpu")
model.load_state_dict(torch.load("yolo_model.pth", map_location="cpu"))
model.eval()
print("Model loaded successfully!")

# Configure transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# COCO class names for YOLO v1 (common 20-class subset)
CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Lowered confidence threshold
CONFIDENCE_THRESHOLD = 0.7  

# Track request performance
total_requests = 0
total_time = 0
request_times = []

def predict(image):
    """Runs YOLO model on input frame"""
    orig_height, orig_width = image.shape[:2]  # Get original frame size
    
    # 🔹 Transform input image (resize & normalize)
    img_tensor = transform(image).unsqueeze(0)  # Convert to tensor
    
    # 🔹 Run YOLO model
    start_time = time.time()
    with torch.no_grad():
        output = model(img_tensor)
    inference_time = time.time() - start_time
    
    # 🔹 Convert model output to bounding boxes
    bboxes = cellboxes_to_boxes(output)
    results = []

    for box in bboxes[0]:
        class_idx, conf, x, y, w, h = box  # YOLO outputs (normalized)

        if conf > CONFIDENCE_THRESHOLD:
            # 🔹 Convert to top-left corner format
            x_min = int((x - w / 2) * orig_width)  # Scale to image size
            y_min = int((y - h / 2) * orig_height)
            w_scaled = int(w * orig_width)
            h_scaled = int(h * orig_height)

            # 🔹 Ensure bounding box is within valid image dimensions
            x_min = max(0, min(x_min, orig_width - 1))
            y_min = max(0, min(y_min, orig_height - 1))
            w_scaled = max(1, min(w_scaled, orig_width - x_min))
            h_scaled = max(1, min(h_scaled, orig_height - y_min))

            # 🔹 Get class name
            class_name = CLASSES[int(class_idx)] if int(class_idx) < len(CLASSES) else f"class_{int(class_idx)}"

            results.append({
                "class": class_name,
                "class_id": int(class_idx),
                "conf": float(conf),
                "x": x_min,
                "y": y_min,
                "w": w_scaled,
                "h": h_scaled
            })

    print(f"Inference time: {inference_time:.3f}s, Found {len(results)} objects with conf > {CONFIDENCE_THRESHOLD}")
    if results:
        print(f"Top detection: {results[0]['class']} with confidence {results[0]['conf']:.2f}")

    return results, inference_time


@app.route("/predict", methods=["POST"])
def predict_video():
    """Receives image, runs YOLO, and returns JSON"""
    global total_requests, total_time, request_times
    
    start_time = time.time()
    try:
        # Get image file from request
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image provided"}), 400
        
        # Convert to OpenCV format
        img_np = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Save a sample frame occasionally for debugging
        if total_requests % 100 == 0:
            cv2.imwrite(f"debug_frame_{total_requests}.jpg", frame)
            print(f"Saved debug frame {total_requests}")
            
        # Run prediction
        results, inference_time = predict(frame)
        
        # Update stats
        total_requests += 1
        request_time = time.time() - start_time
        total_time += request_time
        request_times.append(request_time)
        
        # Print stats every 100 requests
        if total_requests % 100 == 0:
            avg_time = total_time / total_requests
            print(f"Processed {total_requests} requests, avg time: {avg_time:.3f}s")
            # Calculate p95 latency
            if len(request_times) >= 20:
                p95 = sorted(request_times[-100:])[int(95*len(request_times[-100:])/100)]
                print(f"p95 latency: {p95:.3f}s")
        
        return jsonify({
            "detections": results,
            "inference_time": inference_time,
            "request_time": request_time
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Returns server statistics"""
    if total_requests == 0:
        return jsonify({"status": "No requests processed yet"})
    
    avg_time = total_time / total_requests
    p95 = 0
    if len(request_times) >= 20:
        p95 = sorted(request_times[-100:])[int(95*len(request_times[-100:])/100)]
    
    return jsonify({
        "total_requests": total_requests,
        "average_time": avg_time,
        "p95_latency": p95,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })

@app.route("/threshold", methods=["POST"])
def set_threshold():
    """Allows changing confidence threshold at runtime"""
    global CONFIDENCE_THRESHOLD
    
    try:
        data = request.json
        new_threshold = float(data.get("threshold", 0.2))
        
        if 0 <= new_threshold <= 1.0:
            CONFIDENCE_THRESHOLD = new_threshold
            return jsonify({"status": "success", "new_threshold": CONFIDENCE_THRESHOLD})
        else:
            return jsonify({"error": "Threshold must be between 0 and 1"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"Starting server with confidence threshold: {CONFIDENCE_THRESHOLD}")
    app.run(host="0.0.0.0", port=5000, debug=True)