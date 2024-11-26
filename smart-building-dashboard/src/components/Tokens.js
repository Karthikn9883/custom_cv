// src/components/Tokens.js
import React from 'react';
import { Card, Typography } from '@mui/material';

const Tokens = ({ title, count }) => {
  return (
    <Card style={{ padding: '20px', backgroundColor: '#FFFFFF' }}>
      <Typography variant="h6">{title}</Typography>
      <Typography variant="h4" style={{ color: '#1976D2' }}>
        {count}
      </Typography>
    </Card>
  );
};

export default Tokens;
