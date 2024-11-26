// src/components/RoomLayout.js

import React, { useState, useEffect } from "react";
import Draggable from "react-draggable";
import axios from 'axios';
import roomLayout from "../assets/room-layout.png";

const RoomLayout = () => {
  const [cameras, setCameras] = useState([]);

  useEffect(() => {
    // Fetch existing camera data from backend
    const fetchCameras = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_API_BASE_URL}/cameras`);
        setCameras(response.data.map(camera => ({
          id: camera.camera_id,
          x: camera.x_coordinate,
          y: camera.y_coordinate,
          angle: camera.angle,
          camera_name: camera.camera_name,
          room_no: camera.room_no,
          floor: camera.floor,
        })));
      } catch (error) {
        console.error('Error fetching cameras:', error);
      }
    };
    fetchCameras();
  }, []);

  // Add a new camera
  const addCamera = () => {
    const newCamera = { id: Date.now(), x: 100, y: 100, angle: 0 };
    setCameras([...cameras, newCamera]);
  };

  // Remove the last camera
  const removeCamera = () => {
    if (cameras.length > 0) {
      const updatedCameras = cameras.slice(0, -1);
      setCameras(updatedCameras);
    }
  };

  // Rotate the camera
  const rotateCamera = (id, direction) => {
    setCameras((prevCameras) =>
      prevCameras.map((camera) =>
        camera.id === id
          ? { ...camera, angle: (camera.angle + direction * 15) % 360 }
          : camera
      )
    );
  };

  // Save cameras to backend
  const saveCameras = async () => {
    try {
      const cameraData = cameras.map(camera => ({
        camera_id: camera.id,
        camera_name: camera.camera_name || `Camera ${camera.id}`,
        x_coordinate: camera.x,
        y_coordinate: camera.y,
        angle: camera.angle,
        room_no: camera.room_no || 'Unknown',
        floor: camera.floor || 0,
      }));
      await axios.post(`${process.env.REACT_APP_API_BASE_URL}/cameras`, cameraData);
      alert('Cameras saved successfully.');
    } catch (error) {
      console.error('Error saving cameras:', error);
      alert('Failed to save cameras.');
    }
  };

  return (
    <div>
      <h2>Room Layout</h2>
      <div
        style={{
          position: "relative",
          width: "800px",
          height: "600px",
          backgroundImage: `url(${roomLayout})`,
          backgroundSize: "cover",
          margin: "20px auto",
          border: "1px solid black",
        }}
      >
        {cameras.map((camera) => (
          <Draggable
            key={camera.id}
            position={{ x: camera.x, y: camera.y }}
            onStop={(e, data) => {
              setCameras((prevCameras) =>
                prevCameras.map((c) =>
                  c.id === camera.id
                    ? { ...c, x: data.x, y: data.y }
                    : c
                )
              );
            }}
          >
            <div
              style={{
                position: "absolute",
                transform: `translate(-50%, -50%)`,
                cursor: "pointer",
              }}
            >
              {/* Camera Pin */}
              <div
                style={{
                  position: "relative",
                  width: "20px",
                  height: "20px",
                  backgroundColor: "red",
                  borderRadius: "50%",
                  border: "2px solid black",
                }}
              >
                {/* Direction Line */}
                <div
                  style={{
                    position: "absolute",
                    top: "50%",
                    left: "50%",
                    width: "2px",
                    height: "40px",
                    backgroundColor: "black",
                    transform: `rotate(${camera.angle}deg) translate(-50%, -100%)`,
                    transformOrigin: "50% 0%",
                  }}
                ></div>
              </div>

              {/* Rotation Controls */}
              <button
                style={{
                  display: "block",
                  margin: "5px auto",
                  fontSize: "12px",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  color: "blue",
                }}
                onClick={() => rotateCamera(camera.id, 1)}
              >
                ↻
              </button>
              <button
                style={{
                  display: "block",
                  margin: "5px auto",
                  fontSize: "12px",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  color: "blue",
                }}
                onClick={() => rotateCamera(camera.id, -1)}
              >
                ↺
              </button>
            </div>
          </Draggable>
        ))}
      </div>
      <div style={{ textAlign: "center", marginTop: "10px" }}>
        <button
          onClick={addCamera}
          style={{
            marginRight: "10px",
            backgroundColor: "#1976D2",
            color: "white",
            border: "none",
            padding: "10px 20px",
            cursor: "pointer",
          }}
        >
          Add Camera
        </button>
        <button
          onClick={removeCamera}
          style={{
            backgroundColor: "#D32F2F",
            color: "white",
            border: "none",
            padding: "10px 20px",
            cursor: "pointer",
          }}
        >
          Remove Camera
        </button>
        <button
          onClick={saveCameras}
          style={{
            marginLeft: "10px",
            backgroundColor: "#4CAF50",
            color: "white",
            border: "none",
            padding: "10px 20px",
            cursor: "pointer",
          }}
        >
          Save Cameras
        </button>
      </div>
    </div>
  );
};

export default RoomLayout;
