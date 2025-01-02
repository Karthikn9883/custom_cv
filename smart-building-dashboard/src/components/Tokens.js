// src/components/Tokens.js

import React, { useEffect, useState, useRef } from "react";
import { toast } from "react-toastify"; // Import only 'toast' without 'ToastContainer'

const Tokens = ({ title, type }) => {
  const [tokenCount, setTokenCount] = useState(0);
  const prevTokenCount = useRef(0);
  const audioRef = useRef(new Audio("/notification_sound.mp3"));
  const [error, setError] = useState(null);

  // Request and store notification permission on mount
  useEffect(() => {
    if (
      Notification.permission !== "granted" &&
      Notification.permission !== "denied"
    ) {
      Notification.requestPermission().then((permission) => {
        console.log(`Browser notification permission: ${permission}`);
      });
    }
  }, []);

  // Continuously fetch token counts from backend
  useEffect(() => {
    const fetchTokenCounts = async () => {
      try {
        console.log(`Fetching token counts for "${title}" (type="${type}")...`);

        // Adjust URL if your backend is at a different location
        const response = await fetch("http://localhost:5001/token_counts");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        console.log("Token data fetched:", data);

        // data should look like: { "pending": number, "resolved": number }
        let newCount = 0;

        // Map the backend fields to your desired type
        if (type === "active") {
          // backend 'pending' -> Active tokens in your UI
          newCount = data.pending;
        } else if (type === "resolved") {
          // backend 'resolved' -> Resolved tokens in your UI
          newCount = data.resolved;
        } else {
          console.warn(
            `Unexpected "type" prop: "${type}". Must be "active" or "resolved".`
          );
        }

        console.log(`[${title}] Fetched count =`, newCount);

        // Compare with previous token count
        if (newCount > prevTokenCount.current) {
          handleTokenIncrement(newCount - prevTokenCount.current);
        } else if (newCount < prevTokenCount.current && type === "active") {
          // If active tokens decreased => some tokens were resolved
          handleTokenDecrement(prevTokenCount.current - newCount);
        }

        setTokenCount(newCount);
        setError(null);
        prevTokenCount.current = newCount; // update the ref for next iteration
      } catch (err) {
        console.error("Error fetching token counts:", err);
        setError("Failed to fetch token counts.");
      }
    };

    // Fetch once on mount
    fetchTokenCounts();

    // Continuously fetch every 5 seconds
    const interval = setInterval(fetchTokenCounts, 5000);

    // Cleanup on unmount
    return () => clearInterval(interval);
  }, [title, type]);

  // Handle increments (e.g., new tokens created)
  const handleTokenIncrement = (increment) => {
    const message = `🚀 ${increment} new ${
      type === "active" ? "Active" : "Resolved"
    } Token${increment > 1 ? "s" : ""}!`;
    showToastNotification(message);
    playSound();
    console.log(message);
  };

  // Handle decrements (e.g., tokens got resolved)
  const handleTokenDecrement = (decrement) => {
    const message = `✅ ${decrement} ${
      type === "active" ? "Active" : "Resolved"
    } Token${decrement > 1 ? "s" : ""} resolved!`;
    showToastNotification(message);
    playSound();
    console.log(message);
  };

  // Show toast notifications using react-toastify
  const showToastNotification = (message) => {
    const toastOptions = {
      position: "top-right",
      autoClose: 5000,
      hideProgressBar: false,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
    };

    // Differentiate styles if desired
    if (type === "active") {
      toast.success(message, toastOptions);
    } else if (type === "resolved") {
      toast.info(message, toastOptions);
    } else {
      toast(message, toastOptions);
    }
  };

  // Play a notification sound (optional)
  const playSound = () => {
    if (audioRef.current) {
      audioRef.current.play().catch((err) => {
        console.error("Error playing sound:", err);
      });
    }
  };

  return (
    <div
      style={{
        border: "1px solid #ccc",
        padding: "10px",
        borderRadius: "5px",
        width: "220px",
        textAlign: "center",
        margin: "10px",
      }}
    >
      <h3>{title}</h3>
      {error ? (
        <p style={{ color: "red" }}>{error}</p>
      ) : (
        <p style={{ fontSize: "2em", margin: "20px 0" }}>{tokenCount}</p>
      )}
      {/* Removed <ToastContainer /> */}
    </div>
  );
};

export default Tokens;