import React from "react";
import "./LiveStream.css";

const LiveStream = () => {
  const videoFeeds = [
    "http://localhost:5001/video_feed", // Replace with your actual feed URLs
    "http://localhost:5001/video_feed",
    "http://localhost:5001/video_feed",
    "http://localhost:5001/video_feed",
    "http://localhost:5001/video_feed",
    "http://localhost:5001/video_feed",
  ];

  return (
    <div className="live-stream-container">
      <h2>Live Stream</h2>
      <div className="video-grid">
        {videoFeeds.map((feed, index) => (
          <div key={index} className="video-box">
            <img
              src={feed}
              alt={`Live Stream ${index + 1}`}
              style={{ width: "100%", height: "auto", borderRadius: "10px" }}
            />
          </div>
        ))}
      </div>
    </div>
  );
};

export default LiveStream;
