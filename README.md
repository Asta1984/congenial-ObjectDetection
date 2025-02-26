# 🚀 Congenial Object Detection (YOLO from Scratch)

This repository contains a **custom object detection system** built from scratch using **PyTorch, Flask, WebRTC, and React**. The model is trained on the **Pascal VOC dataset** and can perform real-time detection on **images, videos, webcams, YouTube streams, and RTMP feeds**.

## 📜 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1️⃣ Running the Backend (Flask + PyTorch)](#1️⃣-running-the-backend-flask--pytorch)
  - [2️⃣ Running the Frontend (React + WebRTC)](#2️⃣-running-the-frontend-react--webrtc)
  - [3️⃣ Running the Python Client for Local Video](#3️⃣-running-the-python-client-for-local-video)
  - [4️⃣ Running Object Detection on YouTube Videos](#4️⃣-running-object-detection-on-youtube-videos)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Features
✅ **YOLO-Based Custom Object Detection** (No Pretrained Models!)  
✅ **Supports Local Images, Videos, Webcams, RTMP, and YouTube**  
✅ **Flask API for Model Inference**  
✅ **WebRTC-Based React Frontend for Real-Time Processing**  
✅ **Customizable Confidence Threshold**  
✅ **Runs in Google Colab, Kaggle, and Local Systems**  

---

## ⚡ Installation

### 🔹 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Asta1984/congenial-ObjectDetection.git
cd congenial-ObjectDetection
🔹 2️⃣ Set Up the Python Environment
Install dependencies using requirements.txt:

bash
Copy
Edit
pip install -r requirements.txt
If using a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
🔹 3️⃣ Set Up the Frontend (React + Vite)
bash
Copy
Edit
cd frontend
npm install
🚀 Usage
1️⃣ Running the Backend (Flask + PyTorch)
bash
Copy
Edit
cd backend
python app.py
This will start the Flask server at http://localhost:5000.

2️⃣ Running the Frontend (React + WebRTC)
bash
Copy
Edit
cd frontend
npm run dev
Then, open http://localhost:5173 in your browser.

📌 Note: Make sure your backend (app.py) is running before launching the frontend.

3️⃣ Running the Python Client for Local Video
To test detection on a local video file, modify VIDEO_PATH in video_client.py:

python
Copy
Edit
VIDEO_PATH = "E:/path_to_your_video.mp4"
Then, run:

bash
Copy
Edit
python video_client.py
4️⃣ Running Object Detection on YouTube Videos
To test detection on a YouTube video, modify YOUTUBE_URL in youtube_client.py:

python
Copy
Edit
YOUTUBE_URL = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
Then, run:

bash
Copy
Edit
python youtube_client.py
📁 Project Structure
graphql
Copy
Edit
congenial-ObjectDetection/
│── backend/                    # Flask API for object detection
│   ├── app.py                   # Main Flask app
│   ├── model.py                  # YOLO model definition
│   ├── utils.py                  # Helper functions (NMS, IoU, etc.)
│   ├── requirements.txt          # Backend dependencies
│
│── frontend/                     # React + WebRTC frontend
│   ├── src/
│   ├── package.json              # Frontend dependencies
│   ├── vite.config.js            # Vite config
│
│── clients/                      # Python clients for testing
│   ├── video_client.py           # Test detection on local video files
│   ├── youtube_client.py         # Test detection on YouTube videos
│
│── models/                        # Trained YOLO models
│   ├── yolo_model.pth            # Saved model weights
│
│── dataset/                       # Pascal VOC dataset (if needed)
│
│── README.md                      # This documentation
🔬 Model Details
Dataset: Trained on Pascal VOC (20 classes)
Architecture: Custom CNN-based YOLO model
Input Size: 448x448
Loss Function: Custom YOLO loss
Bounding Box Format: (x_center, y_center, width, height)
📡 API Endpoints
Endpoint	Method	Description
/predict	POST	Detect objects in an uploaded image
/offer	POST	WebRTC video stream processing
/stats	GET	Get API performance stats
/threshold	POST	Adjust confidence threshold dynamically
📌 Example predict API Call

bash
Copy
Edit
curl -X POST -F "image=@test.jpg" http://localhost:5000/predict
🔥 Troubleshooting
1️⃣ Flask App Not Running?

Ensure you are in the backend directory and run python app.py.
2️⃣ WebRTC Video Not Loading?

Open Chrome DevTools (F12) and check for errors.
Try clearing browser cache or using a different browser.
3️⃣ CUDA Not Available?

Run torch.cuda.is_available() inside Python.
If False, make sure PyTorch is installed with GPU support.
🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.
Clone your fork.
Create a new branch (git checkout -b new-feature).
Commit your changes (git commit -m "Added new feature").
Push to GitHub (git push origin new-feature).
Open a Pull Request.
📜 License
This project is licensed under the MIT License.
Feel free to modify and use it for your own projects! 🚀
