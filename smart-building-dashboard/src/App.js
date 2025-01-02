// src/App.js

import React, { useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import LiveStream from "./components/LiveStream";
import PowerConsumption from "./components/PowerConsumption";
import WeatherUpdates from "./components/WeatherUpdates";
import Tokens from "./components/Tokens";
import RoomLayout from "./components/RoomLayout";
import ManageWorkers from "./components/ManageWorkers"; // Import ManageWorkers
import { Grid } from "@mui/material";
import { ToastContainer } from "react-toastify"; // Import ToastContainer
import "react-toastify/dist/ReactToastify.css"; // Import Toastify CSS

function App() {
  const [activeSection, setActiveSection] = useState("dashboard");

  return (
    <div style={{ display: "flex" }}>
      <Sidebar setActiveSection={setActiveSection} />
      <div style={{ flex: 1, padding: "20px" }}>
        <Header />
        {activeSection === "dashboard" && (
          <Grid container spacing={2} style={{ marginTop: "20px" }}>
            <Grid item xs={4}>
              <PowerConsumption />
            </Grid>
            <Grid item xs={4}>
              <WeatherUpdates />
            </Grid>
            {/* Pass "active" and "resolved" as the type prop */}
            <Grid item xs={4}>
              <Tokens title="Active Tokens" type="active" />
            </Grid>
            <Grid item xs={4}>
              <Tokens title="Resolved Tokens" type="resolved" />
            </Grid>
            <LiveStream />
          </Grid>
        )}
        {activeSection === "manageCams" && (
          <Grid container spacing={2} style={{ marginTop: "20px" }}>
            <Grid item xs={12}>
              <RoomLayout />
            </Grid>
          </Grid>
        )}
        {activeSection === "manageWorkers" && (
          <Grid container spacing={2} style={{ marginTop: "20px" }}>
            <Grid item xs={12}>
              <ManageWorkers />
            </Grid>
          </Grid>
        )}
        {/* Place ToastContainer once at the root of your app */}
        <ToastContainer />
      </div>
    </div>
  );
}

export default App;