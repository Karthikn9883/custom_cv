# RealTimeObjectDetection
with yolo-world and Nvidia NanoSam

# SCOPE: Smart Building Computer Vision System


This repository contains the source code for the SCOPE (Smart Campus Operations & Predictive Environment) project, also known as the Smart Building Computer Vision Project.The system leverages AI-powered computer vision and IoT to transform reactive building maintenance into a proactive, autonomous process. By detecting and responding to facilities issues in real-time, the project aims to cut operational costs, raise safety and cleanliness, and improve energy efficiency.

## Key Features (MVP1)
The current development focus is on delivering the core functionalities for MVP1:

  * **Trash Management:** Monitors trash bins using computer vision to detect overflow and dispatch cleanup alerts, preventing hygiene issues and reducing manual checks.
  * **Lost and Found Detection:** Identifies and logs common unattended items (e.g., electronics, clothing) to streamline the lost and found process for building occupants.
  * **Spill Detection:** Uses segmentation to identify spills and leaks on floors, triggering immediate alerts for swift cleanup to prevent slip-and-fall accidents.

## System Architecture

The project is built on a modern, scalable architecture designed for real-time edge computing:

  * **Edge Device:** NVIDIA Jetson (Nano or Orin Nano) for real-time AI inference on camera streams.
  * **Vision Model:** Utilizes a **YOLO-World** model for flexible, open-vocabulary object detection, allowing for rapid prototyping without extensive data annotation.
  * **Communication:** An **MQTT** message broker (Eclipse Mosquitto) serves as the central hub for coordinating events between detectors, robots, and dashboards.
  * **Robotics:** The system is designed to integrate with a **ROS 2**-based autonomous robot for ticket resolution (e.g., cleaning spills).

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

  * Git
  * Python 3.10+

### Installation & Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/arshad8049/RealTimeObjectDetection.git
    cd RealTimeObjectDetection
    ```

2.  **Create and activate a Python virtual environment:**

    ```bash
    # Create the virtual environment
    python3 -m venv venv

    # Activate it (on macOS/Linux)
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Important: Handling the Model (`.pt`) File

This project uses a pre-trained YOLO-World model. The first time you run the detection script, the `ultralytics` library will automatically download the model weights file (`yolov8s-world.pt`) for you.

**Please note the following:**

  * **File Size:** This file is large (over 100 MB).
  * **`.gitignore`:** The `.gitignore` file in this repository is **intentionally configured to ignore all `.pt` files.**
  * **Why it's ignored:** Model weights are considered compiled artifacts, not source code. Storing large binary files in Git is inefficient and against best practices. Any developer who clones the repository can automatically download the correct model by simply running the script. This keeps the repository lightweight and clean.
  * **Action Required:** None. This process is automatic. Simply run the code, let it download, and know that Git is correctly ignoring the file.

### Configuration

All settings are managed in the `configs/main_config.yaml` file. Here you can change the detection prompts, confidence threshold, and input source.

```yaml
# configs/main_config.yaml

# --- YOLO-World Settings ---
yolo_model: 'yolov8s-world.pt'

# --- Detection Prompts ---
# Add or remove any objects you want to detect
detection_prompts:
  - 'overflowing trash bin'
  - 'spill on floor'
  - 'unattended backpack'
  # ... and so on

# --- Input Source ---
# Default source when running without arguments. Use '0' for webcam.
input_source: 0

# --- MQTT Broker Settings ---
mqtt:
  enabled: false # Set to true to enable MQTT publishing
  broker_address: "localhost"
  port: 1883
  detection_topic: "smart-building/detections"
```

## Usage

Run all commands from the root directory of the project (`RealTimeObjectDetection`).

  * **To run the live webcam feed:**

    ```bash
    python mart_building_cv/src/detection/detector.py
    ```

  * **To process a specific image or video file:**

    ```bash
    python mart_building_cv/src/detection/detector.py --input /path/to/your/image.jpg
    ```

  * **To set a custom confidence threshold:**

    ```bash
    python mart_building_cv/src/detection/detector.py --conf 0.5
    ```