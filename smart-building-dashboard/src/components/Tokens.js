import React, { useEffect, useState } from "react";

const Tokens = ({ title }) => {
  const [tokenCount, setTokenCount] = useState(0);

  useEffect(() => {
    const fetchTokenCounts = async () => {
      try {
        const response = await fetch("http://localhost:5001/token_counts");
        const data = await response.json();
        console.log("Token data fetched:", data); // Debug log

        // Update count based on the title
        if (title === "Active Tokens") {
          setTokenCount(data.active_tokens_count);
        } else if (title === "Resolved Tokens") {
          setTokenCount(data.resolved_tokens_count);
        }
      } catch (error) {
        console.error("Error fetching token counts:", error);
      }
    };

    fetchTokenCounts();

    // Optionally refresh data periodically (e.g., every 5 seconds)
    const interval = setInterval(fetchTokenCounts, 5000);
    return () => clearInterval(interval); // Cleanup on component unmount
  }, [title]);

  return (
    <div style={{ border: "1px solid #ccc", padding: "10px", borderRadius: "5px" }}>
      <h3>{title}</h3>
      <p>{tokenCount}</p>
    </div>
  );
};

export default Tokens;
