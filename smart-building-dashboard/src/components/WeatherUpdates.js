import React, { useState, useEffect } from "react";
import { Card, Typography } from "@mui/material";

// Mapping weather codes to human-readable descriptions
const weatherConditions = {
  0: "Clear",
  1: "Mainly Clear",
  2: "Partly Cloudy",
  3: "Overcast",
  45: "Foggy",
  48: "Depositing Rime Fog",
  51: "Drizzle: Light",
  53: "Drizzle: Moderate",
  55: "Drizzle: Dense",
  61: "Rain: Slight",
  63: "Rain: Moderate",
  65: "Rain: Heavy",
  71: "Snow: Slight",
  73: "Snow: Moderate",
  75: "Snow: Heavy",
  80: "Rain Showers: Slight",
  81: "Rain Showers: Moderate",
  82: "Rain Showers: Violent",
  95: "Thunderstorm: Slight",
  96: "Thunderstorm: Slight Hail",
  99: "Thunderstorm: Heavy Hail",
};

const WeatherUpdates = () => {
  const [weatherData, setWeatherData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  // Fetch weather data
  const fetchWeather = async (latitude, longitude) => {
    try {
      const response = await fetch(
        `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current_weather=true`
      );
      if (!response.ok) {
        throw new Error('Failed to fetch weather data.');
      }
      const data = await response.json();
      setWeatherData(data.current_weather);
      setLoading(false);
    } catch (err) {
      console.error(err);
      setError('Failed to load weather data.');
      setLoading(false);
    }
  };

  // Get user's location
  useEffect(() => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          fetchWeather(latitude, longitude);
        },
        (err) => {
          console.error(err);
          setError('Unable to retrieve your location.');
          setLoading(false);
        }
      );
    } else {
      setError('Geolocation is not supported by your browser.');
      setLoading(false);
    }
  }, []);

  return (
    <Card
      style={{
        padding: "15px",
        backgroundColor: "#FFFFFF",
        boxShadow: "0 2px 5px rgba(0, 0, 0, 0.1)",
        height: "250px",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        textAlign: "center",
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
        Weather in Your Location
      </Typography>
      {loading ? (
        <Typography variant="body2">Fetching weather data...</Typography>
      ) : error ? (
        <Typography variant="body2" color="error">
          {error}
        </Typography>
      ) : (
        <>
          <Typography variant="body1" style={{ fontSize: "1.2rem", fontWeight: "bold" }}>
            Temperature: {weatherData.temperature}°C
          </Typography>
          <Typography variant="body2" style={{ fontSize: "1rem" }}>
            Condition: {weatherConditions[weatherData.weathercode] || "Unknown"}
          </Typography>
        </>
      )}
    </Card>
  );
};

export default WeatherUpdates;