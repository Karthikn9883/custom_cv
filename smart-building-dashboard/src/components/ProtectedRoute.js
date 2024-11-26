// frontend/src/components/ProtectedRoute.js

import React from 'react';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ authToken, children }) => {
  return authToken ? children : <Navigate to="/login" replace />;
};

export default ProtectedRoute;
