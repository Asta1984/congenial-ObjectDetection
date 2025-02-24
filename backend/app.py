from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from torchvision import transforms
from flask_cors import CORS
from utils import cellboxes_to_boxes
import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
import asyncio

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load YOLO model
model = torch.load("yolo_deploy.pth", map_location="cpu")
model.eval()

# WebRTC Peer Connection
pcs = set()
relay = MediaRelay()

class YOLOVideoTrack(VideoStreamTrack):
    """Processes video frames and applies YOLO object detection"""
    
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = relay.subscribe(track)

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO on frame
        results = predict(img)

        # Draw bounding boxes
        for det in results:
            x, y, w, h = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"Class {det['class']} ({det['conf']:.2f})", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert back to WebRTC format
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        return new_frame

def predict(image):
    """Runs YOLO model on input frame"""
    transform = torch.nn.Sequential(
        torch.nn.functional.interpolate(size=(448, 448)),
        torch.nn.functional.to_tensor()
    )
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)

    # Convert to bounding boxes
    bboxes = cellboxes_to_boxes(output)
    results = []
    for box in bboxes[0]:
        class_idx, conf, x, y, w, h = box
        if conf > 0.5:
            results.append({"class": int(class_idx), "conf": float(conf),
                            "x": float(x), "y": float(y), "w": float(w), "h": float(h)})
    return results

@app.route("/offer", methods=["POST"])
async def offer():
    """Handles WebRTC peer connection setup"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(YOLOVideoTrack(track))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return jsonify({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
