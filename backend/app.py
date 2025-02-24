from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from torchvision import transforms
from flask_cors import CORS
from utils import cellboxes_to_boxes

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend to communicate

# Load trained YOLO model
model = torch.load("yolo_deploy.pth", map_location="cpu")
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

def predict(image):
    """Runs YOLO model on input frame"""
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    
    # Convert model output to bounding boxes
    bboxes = cellboxes_to_boxes(output)  # Convert to [class, conf, x, y, w, h]
    results = []

    for box in bboxes[0]:
        class_idx, conf, x, y, w, h = box
        if conf > 0.5:
            results.append({"class": int(class_idx), "conf": float(conf),
                            "x": float(x), "y": float(y), "w": float(w), "h": float(h)})

    return results

@app.route("/predict", methods=["POST"])
def predict_video():
    """Receives video frame, runs inference, and returns JSON response"""
    file = request.files["image"]
    img_np = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    results = predict(frame)
    return jsonify({"detections": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
