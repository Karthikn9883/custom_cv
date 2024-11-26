import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  CategoryScale,
} from "chart.js";
import { Card, Typography } from "@mui/material";

// Register Chart.js components
ChartJS.register(LineElement, PointElement, LinearScale, Title, Tooltip, Legend, CategoryScale);

const PowerConsumption = () => {
  // Data for the chart
  const data = {
    labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    datasets: [
      {
        label: "Power Usage (kWh)",
        data: [30, 45, 28, 50, 35, 60],
        borderColor: "#1976D2",
        backgroundColor: "rgba(25, 118, 210, 0.1)",
        borderWidth: 2,
        tension: 0.4,
        pointBackgroundColor: "#1976D2",
      },
    ],
  };

  // Chart options
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: "top",
      },
      tooltip: {
        enabled: true,
      },
    },
    scales: {
      x: {
        type: "category",
        title: {
          display: true,
          text: "Months",
          font: {
            size: 12,
            weight: "bold",
          },
        },
      },
      y: {
        title: {
          display: true,
          text: "Usage (kWh)",
          font: {
            size: 12,
            weight: "bold",
          },
        },
        min: 20,
        max: 70,
        ticks: {
          stepSize: 10,
        },
      },
    },
  };

  return (
    <Card
      style={{
        padding: "15px",
        backgroundColor: "#FFFFFF",
        boxShadow: "0 2px 5px rgba(0, 0, 0, 0.1)",
        height: "250px",
      }}
    >
      <Typography
        variant="h6"
        style={{
          marginBottom: "10px",
          color: "#1976D2",
          fontSize: "1rem",
        }}
      >
        Power Consumption
      </Typography>
      <div style={{ height: "180px", overflow: "hidden" }}>
        <Line data={data} options={options} />
      </div>
    </Card>
  );
};

export default PowerConsumption;
