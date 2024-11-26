// src/components/WeatherUpdates.js
import React from 'react';
import { Card, Typography } from '@mui/material';

const WeatherUpdates = () => {
  return (
    <Card style={{ padding: '20px', backgroundColor: '#E3F2FD' }}>
      <Typography variant="h6" style={{ color: '#1976D2' }}>
        Weather Updates
      </Typography>
      <Typography variant="body2" style={{ marginTop: '10px' }}>
        Sunny, 28°C
      </Typography>
    </Card>
  );
};

export default WeatherUpdates;
