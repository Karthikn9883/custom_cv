// src/components/LiveStream.js
import React from 'react';
import { Card, Grid, Typography } from '@mui/material';
import ReactPlayer from 'react-player';

const LiveStream = () => {
  const cameraFeeds = [
    'http://localhost:8080/stream.m3u8', // Replace with converted HLS/HTTPS link if necessary
    'rtsp://smart:1234@192.168.1.207:554/avstream/channel=7/stream=0.sdp',
    'rtsp://smart:1234@192.168.1.207:554/avstream/channel=9/stream=0.sdp',
    'rtsp://smart:1234@192.168.1.207:554/avstream/channel=10/stream=0.sdp',
    'rtsp://smart:1234@192.168.1.207:554/avstream/channel=11/stream=0.sdp',
    'rtsp://smart:1234@192.168.1.207:554/avstream/channel=12/stream=0.sdp',
  ];

  return (
    <div>
      <Typography variant="h5" style={{ margin: '20px 0', color: '#1976D2' }}>
        Live Stream
      </Typography>
      <Grid container spacing={2}>
        {cameraFeeds.map((feed, index) => (
          <Grid item xs={4} key={index}>
            <Card style={{ height: '200px', backgroundColor: '#F5F5F5', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <ReactPlayer
                url={feed}
                playing={true}
                controls={true}
                width="100%"
                height="100%"
                config={{
                  file: {
                    attributes: {
                      crossOrigin: 'anonymous',
                    },
                  },
                }}
                style={{ borderRadius: '8px', overflow: 'hidden' }}
              />
            </Card>
          </Grid>
        ))}
      </Grid>
    </div>
  );
};

export default LiveStream;
