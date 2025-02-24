import cv2
import requests

# RTMP Stream URL or Webcam (0 for default cam)
VIDEO_SOURCE = "rtmp://your-rtmp-server/live/stream"

cap = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _, img_encoded = cv2.imencode(".jpg", frame)
    response = requests.post("http://localhost:5000/predict", files={"image": img_encoded.tobytes()})
    detections = response.json()["detections"]

    # Draw bounding boxes
    for det in detections:
        x, y, w, h = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
        label = f"Class {det['class']} ({det['conf']:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
