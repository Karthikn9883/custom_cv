import React from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import LiveStream from './components/LiveStream';
import PowerConsumption from './components/PowerConsumption';
import WeatherUpdates from './components/WeatherUpdates';
import Tokens from './components/Tokens';
import { Grid } from '@mui/material';

function App() {
  return (
    <div style={{ display: 'flex' }}>
      <Sidebar />
      <div style={{ flex: 1, padding: '20px' }}>
        <Header />
        <Grid container spacing={2} style={{ marginTop: '20px' }}>
          <Grid item xs={4}>
            <PowerConsumption />
          </Grid>
          <Grid item xs={4}>
            <WeatherUpdates />
          </Grid>
          <Grid item xs={4}>
            <Tokens title="Active Tokens" count={12} />
          </Grid>
          <Grid item xs={4}>
            <Tokens title="Resolved Tokens" count={45} />
          </Grid>
        </Grid>
        <LiveStream />
      </div>
    </div>
  );
}

export default App;
