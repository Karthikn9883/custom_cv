// src/components/Sidebar.js
import React from 'react';
import { Button, Box } from '@mui/material';

const Sidebar = () => {
  const options = ['Occupancy', 'Manage Cams', 'Manage Workers', 'Settings'];

  return (
    <Box style={{ padding: '20px', backgroundColor: '#F4F6F8', height: '100vh' }}>
      {options.map((option, index) => (
        <Button
          key={index}
          fullWidth
          variant="contained"
          style={{
            marginBottom: '10px',
            backgroundColor: '#1976D2',
            color: '#FFFFFF',
          }}
        >
          {option}
        </Button>
      ))}
    </Box>
  );
};

export default Sidebar;
