import React, { useEffect, useRef } from 'react';
import { Line } from 'react-chartjs-2';

const PowerConsumption = () => {
  const chartRef = useRef(null);

  const data = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    datasets: [
      {
        label: 'Power Usage (kWh)',
        data: [30, 45, 28, 50, 35, 40],
        borderColor: '#1976D2',
        borderWidth: 2,
        tension: 0.4,
      },
    ],
  };

  useEffect(() => {
    const chartInstance = chartRef.current;

    return () => {
      // Cleanup: Destroy the chart instance before unmounting
      if (chartInstance) {
        chartInstance.destroy();
      }
    };
  }, []);

  return (
    <div>
      <Line ref={chartRef} data={data} />
    </div>
  );
};

export default PowerConsumption;
