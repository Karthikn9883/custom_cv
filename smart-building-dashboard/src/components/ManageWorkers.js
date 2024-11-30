// src/components/ManageWorkers.js

import React, { useState, useEffect } from "react";
import axios from "axios";
import { Table, Button, TextField, Dialog, DialogTitle, DialogContent, DialogActions } from "@mui/material";

const ManageWorkers = () => {
  const [workers, setWorkers] = useState([]);
  const [newWorker, setNewWorker] = useState({ name: "", email: "", number: "" });
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  // Fetch workers from the backend
  useEffect(() => {
    const fetchWorkers = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_API_BASE_URL}/workers`);
        setWorkers(response.data);
      } catch (error) {
        console.error("Error fetching workers:", error);
      }
    };
    fetchWorkers();
  }, []);

  // Add a new worker
  const handleAddWorker = async () => {
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_BASE_URL}/workers`, {
        name: newWorker.name,
        email: newWorker.email,
        number: newWorker.number,
        status: "free",
      });
      setWorkers([...workers, response.data]);
      setNewWorker({ name: "", email: "", number: "" });
      setIsDialogOpen(false);
    } catch (error) {
      console.error("Error adding worker:", error);
    }
  };

  // Delete a worker
  const handleDeleteWorker = async (id) => {
    try {
      await axios.delete(`${process.env.REACT_APP_API_BASE_URL}/workers/${id}`);
      setWorkers(workers.filter((worker) => worker.worker_id !== id));
    } catch (error) {
      console.error("Error deleting worker:", error);
    }
  };

  // Evenly distribute workload while assigning tasks
  const assignWorker = async () => {
    const freeWorkers = workers.filter((worker) => worker.status === "free");
    if (freeWorkers.length > 0) {
      const workerToAssign = freeWorkers[Math.floor(Math.random() * freeWorkers.length)];
      try {
        await axios.put(`${process.env.REACT_APP_API_BASE_URL}/workers/${workerToAssign.worker_id}`, {
          status: "occupied",
        });
        setWorkers(
          workers.map((worker) =>
            worker.worker_id === workerToAssign.worker_id
              ? { ...worker, status: "occupied" }
              : worker
          )
        );
        alert(`Task assigned to ${workerToAssign.worker_name}`);
      } catch (error) {
        console.error("Error assigning worker:", error);
      }
    } else {
      alert("No free workers available.");
    }
  };

  return (
    <div>
      <h2>Manage Workers</h2>
      <Table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Email</th>
            <th>Number</th>
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {workers.map((worker) => (
            <tr key={worker.worker_id}>
              <td>{worker.worker_id}</td>
              <td>{worker.worker_name}</td>
              <td>{worker.worker_email}</td>
              <td>{worker.worker_number}</td>
              <td>{worker.status}</td>
              <td>
                <Button
                  variant="contained"
                  color="error"
                  onClick={() => handleDeleteWorker(worker.worker_id)}
                >
                  Delete
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
      <Button variant="contained" color="primary" onClick={() => setIsDialogOpen(true)}>
        Add Worker
      </Button>
      <Button
        variant="contained"
        color="success"
        style={{ marginLeft: "10px" }}
        onClick={assignWorker}
      >
        Assign Worker
      </Button>

      {/* Dialog for adding a new worker */}
      <Dialog open={isDialogOpen} onClose={() => setIsDialogOpen(false)}>
        <DialogTitle>Add Worker</DialogTitle>
        <DialogContent>
          <TextField
            label="Name"
            fullWidth
            margin="normal"
            value={newWorker.name}
            onChange={(e) => setNewWorker({ ...newWorker, name: e.target.value })}
          />
          <TextField
            label="Email"
            fullWidth
            margin="normal"
            value={newWorker.email}
            onChange={(e) => setNewWorker({ ...newWorker, email: e.target.value })}
          />
          <TextField
            label="Number"
            fullWidth
            margin="normal"
            value={newWorker.number}
            onChange={(e) => setNewWorker({ ...newWorker, number: e.target.value })}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIsDialogOpen(false)} color="secondary">
            Cancel
          </Button>
          <Button onClick={handleAddWorker} color="primary">
            Add
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default ManageWorkers;
