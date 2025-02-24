import React, { useState, useEffect, useRef } from "react";
import SimplePeer from "simple-peer";
import axios from "axios";

function App() {
    const videoRef = useRef(null);
    const remoteVideoRef = useRef(null);
    const [peer, setPeer] = useState(null);

    useEffect(() => {
        startWebRTC();
    }, []);

    async function startWebRTC() {
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then((stream) => {
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
            });
    }

    return (
        <div>
            <h1>Live YOLO Object Detection</h1>
            <video ref={videoRef} autoPlay playsInline width="640" height="480" />
            <h2>Processed Video:</h2>
            <video ref={remoteVideoRef} autoPlay playsInline width="640" height="480" />
        </div>
    );
}

export default App;
