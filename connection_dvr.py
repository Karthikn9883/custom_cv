import cv2
import torch
import threading
from queue import Queue
import os

# Set environment variable for FFMPEG (if necessary for RTSP)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# Load the YOLO model from the Ultralytics hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# Queue to store frames for the RTSP stream
frame_queue = Queue()

# RTSP URL for the DVR camera stream (update as needed)
rtsp_url = "rtsp://smart:1234@192.168.1.207:554/avstream/channel=7/stream=1.sdp"

# Function to process video from the RTSP stream
def process_rtsp_stream(rtsp_url, model, frame_queue):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"Cannot open RTSP stream at {rtsp_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to retrieve frame from RTSP stream.")
            break

        # Perform object detection on the frame
        results = model(frame)
        
        # Render the results on the frame
        annotated_frame = results.render()[0]
        
        # Put the annotated frame in the queue for display in the main thread
        if not frame_queue.full():
            frame_queue.put(annotated_frame)

    cap.release()

# Create and start a thread for processing the RTSP stream
thread = threading.Thread(target=process_rtsp_stream, args=(rtsp_url, model, frame_queue))
thread.start()

# Main display loop
while True:
    # Display the frame from the RTSP stream
    if not frame_queue.empty():
        frame = frame_queue.get()
        cv2.imshow('RTSP Stream with YOLO', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Wait for the thread to complete
thread.join()
cv2.destroyAllWindows()
