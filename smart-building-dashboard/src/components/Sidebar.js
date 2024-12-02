import React from "react";
import { Button, Stack } from "@mui/material";

const Sidebar = ({ setActiveSection, handleLogout }) => {
  return (
    <div style={{ padding: "20px", background: "#f4f4f4", height: "100vh" }}>
      <Stack spacing={2}>
        <Button
          variant="contained"
          color="primary"
          onClick={() => setActiveSection("dashboard")}
        >
          Dashboard
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={() => setActiveSection("manageCams")}
        >
          Manage Cams
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={() => setActiveSection("manageWorkers")}
        >
          Manage Workers
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={() => setActiveSection("settings")}
        >
          Settings
        </Button>
        <Button variant="contained" color="secondary" onClick={handleLogout}>
          Logout
        </Button>
      </Stack>
    </div>
  );
};

export default Sidebar;
