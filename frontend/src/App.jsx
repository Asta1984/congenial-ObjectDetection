import React, { useState, useEffect, useRef } from "react";
import ReactPlayer from "react-player";
import SimplePeer from "simple-peer";
import axios from "axios";

function App() {
    const videoRef = useRef(null);
    const remoteVideoRef = useRef(null);
    const [peer, setPeer] = useState(null);
    const [videoURL, setVideoURL] = useState("");
    const [isStreaming, setIsStreaming] = useState(false);

    useEffect(() => {
        if (isStreaming) {
            startWebRTC();
        }
    }, [isStreaming]);

    async function startWebRTC() {
        try {
            console.log("🎥 Requesting camera access...");
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }

            console.log("✅ Camera stream received!");

            const peer = new SimplePeer({
                initiator: true,
                trickle: false,
                stream: stream,
                config: {
                    iceServers: [
                        { urls: 'stun:stun.l.google.com:19302' },
                        // Add more STUN/TURN servers as needed
                    ],
                },
            });

            peer.on("signal", async (data) => {
                console.log("📡 Sending WebRTC offer...", data);
                try {
                    const response = await axios.post("http://localhost:5000/offer", data);
                    console.log("✅ Received WebRTC response!", response.data);
                    peer.signal(response.data);
                } catch (error) {
                    console.error("❌ WebRTC connection failed:", error);
                }
            });

            peer.on("stream", (remoteStream) => {
                console.log("🔄 Received processed video stream!");
                if (remoteVideoRef.current) {
                    remoteVideoRef.current.srcObject = remoteStream;
                }
            });

            setPeer(peer);
        } catch (error) {
            console.error("❌ WebRTC Error:", error);
        }
    }

    const handleURLChange = (event) => {
        setVideoURL(event.target.value);
    };

    const startStreaming = () => {
        console.log("Video URL:", videoURL);
        console.log("ReactPlayer.canPlay(videoURL):", ReactPlayer.canPlay(videoURL));

        if (ReactPlayer.canPlay(videoURL)) {
            console.log("▶️ Starting stream...");
            setIsStreaming(true);
        } else {
            alert("❌ Invalid video URL! Please enter a valid YouTube, RTMP, or IP camera stream.");
        }
    };

    return (
        <div className="container">
            <h1>Live YOLO Object Detection</h1>

            <input
                type="text"
                placeholder="Enter Video URL (YouTube, RTMP, IP Camera, Local File)"
                value={videoURL}
                onChange={handleURLChange}
                className="input-box"
            />
            <button onClick={startStreaming} className="start-button">
                Start Streaming
            </button>

            {isStreaming && (
                <div className="video-container">
                    <h2>Original Video Stream:</h2>
                    <video ref={videoRef} autoPlay playsInline width="640" height="360" />

                    <h2>Processed YOLO Detection Stream:</h2>
                    <video ref={remoteVideoRef} autoPlay playsInline width="640" height="360" />
                </div>
            )}
        </div>
    );
}

export default App;