import React, { useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import LiveStream from "./components/LiveStream";
import PowerConsumption from "./components/PowerConsumption";
import WeatherUpdates from "./components/WeatherUpdates";
import Tokens from "./components/Tokens";
import RoomLayout from "./components/RoomLayout";
import { Grid } from "@mui/material";

function App() {
  // Define activeSection state to manage which section is displayed
  const [activeSection, setActiveSection] = useState("dashboard");

  return (
    <div style={{ display: "flex" }}>
      {/* Pass setActiveSection as a prop to Sidebar */}
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
      </div>
    </div>
  );
}

export default App;
