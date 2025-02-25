import React, { useState, useEffect, useRef } from "react";
import ReactPlayer from "react-player";
import SimplePeer from "simple-peer";
import axios from "axios";

function App() {
    const videoRef = useRef(null);
    const remoteVideoRef = useRef(null);
    const [peer, setPeer] = useState(null);
    const [videoURL, setVideoURL] = useState(""); // User-defined video URL
    const [isStreaming, setIsStreaming] = useState(false); // Track streaming state

    useEffect(() => {
        if (isStreaming) {
            startWebRTC();
        }
    }, [isStreaming]); // Only start WebRTC when streaming is enabled

    async function startWebRTC() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            videoRef.current.srcObject = stream;

            const peer = new SimplePeer({
                initiator: true,
                trickle: false,
                stream: stream
            });

            peer.on("signal", async (data) => {
                const response = await axios.post("http://localhost:5000/offer", data);
                peer.signal(response.data);
            });

            peer.on("stream", (remoteStream) => {
                remoteVideoRef.current.srcObject = remoteStream;
            });

            setPeer(peer);
        } catch (error) {
            console.error("WebRTC Error:", error);
        }
    }

    const handleURLChange = (event) => {
        setVideoURL(event.target.value);
    };

    const startStreaming = () => {
        if (ReactPlayer.canPlay(videoURL)) {
            setIsStreaming(true);
        } else {
            alert("Invalid video URL! Please enter a valid YouTube, RTMP, or IP camera stream.");
        }
    };

    return (
        <div>
            <h1>Live YOLO Object Detection</h1>

            {/* Input for custom video URL */}
            <input
                type="text"
                placeholder="Enter Video URL (YouTube, RTMP, IP Camera, Local File)"
                value={videoURL}
                onChange={handleURLChange}
                style={{ width: "60%", padding: "10px", marginBottom: "10px" }}
            />
            <button onClick={startStreaming} style={{ padding: "10px", cursor: "pointer" }}>
                Start Streaming
            </button>

            {/* Conditionally render the video player */}
            {isStreaming && (
                <div>
                    <h2>Original Video Stream:</h2>
                    <ReactPlayer
                        url={videoURL}
                        playing
                        controls
                        width="640px"
                        height="360px"
                        ref={videoRef}
                    />
                    
                    <h2>Processed YOLO Detection Stream:</h2>
                    <video ref={remoteVideoRef} autoPlay playsInline width="640" height="360" />
                </div>
            )}
        </div>
    );
}

export default App;
