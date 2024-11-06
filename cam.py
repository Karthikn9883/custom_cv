import cv2
import torch
import threading
from queue import Queue

# Load the YOLO model from the Ultralytics hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# Queue dictionary to store frames for each camera
frame_queues = {0: Queue(), 1: Queue(), 2: Queue()}

# Function to process video from a specific camera
def process_camera(camera_index, model, frame_queue):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {camera_index} failed to capture video.")
            break

        # Perform object detection on the frame
        results = model(frame)
        
        # Render the results on the frame
        annotated_frame = results.render()[0]
        
        # Put the annotated frame in the queue for display in the main thread
        if not frame_queue.full():
            frame_queue.put(annotated_frame)

    cap.release()

# Create and start threads for each camera
threads = []
for i in range(3):  # Three cameras
    t = threading.Thread(target=process_camera, args=(i, model, frame_queues[i]))
    t.start()
    threads.append(t)

# Main display loop
while True:
    # Display frames from each camera
    for i in range(3):
        if not frame_queues[i].empty():
            frame = frame_queues[i].get()
            cv2.imshow(f'Camera {i}', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Wait for all threads to complete
for t in threads:
    t.join()

cv2.destroyAllWindows()
