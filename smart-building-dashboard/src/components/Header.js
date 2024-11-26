// src/components/Header.js
import React from 'react';
import { Typography } from '@mui/material';

const Header = () => {
  return (
    <div style={{ padding: '10px 20px', background: '#FFFFFF', borderBottom: '2px solid #E0E0E0' }}>
      <Typography variant="h4" style={{ fontWeight: 'bold', color: '#1976D2' }}>
        Smart Building Dashboard
      </Typography>
    </div>
  );
};

export default Header;
