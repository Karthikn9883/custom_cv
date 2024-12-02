import React, { useState, useEffect } from "react";
import axios from "axios";
import {
    Table,
    TableHead,
    TableRow,
    TableCell,
    TableBody,
    Button,
    TextField,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
} from "@mui/material";

const ManageWorkers = () => {
    const [workers, setWorkers] = useState([]);
    const [newWorker, setNewWorker] = useState({ name: "", email: "", number: "" });
    const [isDialogOpen, setIsDialogOpen] = useState(false);

    // Fetch workers from the backend
    const fetchWorkers = async () => {
        try {
            const response = await axios.get(`${process.env.REACT_APP_API_BASE_URL}/workers`);
            setWorkers(response.data);
        } catch (error) {
            console.error("Error fetching workers:", error);
        }
    };

    useEffect(() => {
        fetchWorkers();
    }, []);

    // Add a new worker
    const handleAddWorker = async () => {
        try {
            await axios.post(`${process.env.REACT_APP_API_BASE_URL}/workers`, newWorker);
            fetchWorkers(); // Refresh workers
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
            fetchWorkers(); // Refresh workers
        } catch (error) {
            console.error("Error deleting worker:", error);
        }
    };

    // Update a worker's status
    const handleUpdateStatus = async (id, status) => {
        try {
            await axios.put(`${process.env.REACT_APP_API_BASE_URL}/workers/${id}`, { status });
            fetchWorkers(); // Refresh workers
        } catch (error) {
            console.error("Error updating worker:", error);
        }
    };

    return (
        <div>
            <h2>Manage Workers</h2>
            <Table>
                <TableHead>
                    <TableRow>
                        <TableCell>ID</TableCell>
                        <TableCell>Name</TableCell>
                        <TableCell>Email</TableCell>
                        <TableCell>Number</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Actions</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {workers.map((worker) => (
                        <TableRow key={worker.worker_id}>
                            <TableCell>{worker.worker_id}</TableCell>
                            <TableCell>{worker.worker_name}</TableCell>
                            <TableCell>{worker.worker_email}</TableCell>
                            <TableCell>{worker.worker_number}</TableCell>
                            <TableCell>{worker.status}</TableCell>
                            <TableCell>
                                <Button
                                    variant="contained"
                                    color={worker.status === "free" ? "primary" : "secondary"}
                                    onClick={() =>
                                        handleUpdateStatus(worker.worker_id, worker.status === "free" ? "occupied" : "free")
                                    }
                                >
                                    {worker.status === "free" ? "Assign" : "Release"}
                                </Button>
                                <Button
                                    variant="contained"
                                    color="error"
                                    onClick={() => handleDeleteWorker(worker.worker_id)}
                                    style={{ marginLeft: "10px" }}
                                >
                                    Delete
                                </Button>
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
            <Button variant="contained" color="primary" onClick={() => setIsDialogOpen(true)} style={{ marginTop: "20px" }}>
                Add Worker
            </Button>
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
