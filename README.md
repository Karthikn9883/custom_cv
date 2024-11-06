Multi-Camera YOLO Object Detection Setup

This project enables real-time object detection across multiple cameras on macOS using YOLOv5 from Ultralytics. The setup includes threading for handling multiple cameras and a main display loop for simultaneous feeds.

Prerequisites

Python 3.8 or higher – Install it from python.org.
Compatible Mac system – Ensure macOS permissions are set for camera access.
Three connected cameras – This setup is designed to handle multiple camera feeds.
Setup Instructions

Follow these steps to set up the project environment and install dependencies.

1. Clone the Repository
Clone this project or set up a new folder for your code.

bash
Copy code
git clone https://github.com/your-repo/multi-camera-yolo.git
cd multi-camera-yolo
2. Set Up a Virtual Environment
It’s recommended to use a virtual environment to keep dependencies isolated.

bash
Copy code
python3 -m venv yolov11_env
Activate the virtual environment:

macOS/Linux:
bash
Copy code
source yolov11_env/bin/activate
Windows:
bash
Copy code
yolov11_env\Scripts\activate
3. Install Dependencies
Use the requirements.txt file to install all necessary packages.

bash
Copy code
pip install -r requirements.txt
4. Run the Script
To start the multi-camera object detection, ensure that all cameras are connected and then run:

bash
Copy code
python cam.py
Exiting the Program
Press q in any displayed camera feed window to exit the program.

Troubleshooting

OpenCV Errors: If you encounter errors like Unknown C++ exception from OpenCV, this is often due to multithreaded cv2.imshow() usage. This code structure avoids imshow() calls in threads, so ensure you're running the latest code.
Performance: Running multiple camera feeds with YOLO can be resource-intensive. Try reducing the resolution of each camera if you experience lag.
Mac Camera Permissions: Ensure that your terminal or IDE has permission to access the camera under System Preferences > Security & Privacy > Camera.
Additional Notes

Supported Cameras: This setup supports up to three cameras. If you need more, adjust the code to handle additional queues and threads.
Model Options: You’re using yolov5s for a balance of speed and accuracy. Other models (e.g., yolov5n) are available if you need a faster, lightweight alternative.
License

This project is licensed under the MIT License.