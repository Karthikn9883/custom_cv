
import cv2
import torch
import threading
from queue import Queue

# Load the YOLO model from the Ultralytics hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# Queue dictionary to store frames for each camera
frame_queues = {0: Queue(), 1: Queue(), 2: Queue()}

# Define frame skip values for each camera
frame_skip_values = {0: 12, 1: 12, 2: 12}  # Process every 5th frame for each camera


# Function to process video from a specific camera with frame skip
def process_camera(camera_index, model, frame_queue, frame_skip):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {camera_index} failed to capture video.")
            break
        
        # Only process every nth frame based on frame_skip
        if frame_count % frame_skip == 0:
            # Perform object detection on the frame
            results = model(frame)
            # Render the results on the frame
            annotated_frame = results.render()[0]
            # Put the annotated frame in the queue for display in the main thread
            if not frame_queue.full():
                frame_queue.put(annotated_frame)

        frame_count += 1

    cap.release()

# Create and start threads for each camera
threads = []
for i in range(3):  # Three cameras
    t = threading.Thread(target=process_camera, args=(i, model, frame_queues[i], frame_skip_values[i]))
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
