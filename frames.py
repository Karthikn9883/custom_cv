import cv2
import os

# Function to extract frames from a video file and save them as images
def extract_frames_from_video(video_path, output_folder, frame_interval):
    # Get the base name of the video file to use as a folder name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    cap = cv2.VideoCapture(video_path)  # Open the video file
    count = 0
    success, frame = cap.read()
    
    while success:
        if count % frame_interval == 0:
            frame_path = os.path.join(video_output_folder, f"frame{count}.jpg")
            cv2.imwrite(frame_path, frame)
        success, frame = cap.read()
        count += 1

    cap.release()

# Main function to iterate over all video files in the "training data" folder
def process_all_videos(input_folder, output_folder, frame_interval):
    for file_name in os.listdir(input_folder):
        # Check if the file is a video by looking at the extension
        if file_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more formats if needed
            video_path = os.path.join(input_folder, file_name)
            extract_frames_from_video(video_path, output_folder, frame_interval)
            print(f"Processed video: {file_name}")

# Parameters
input_folder = "data/folder/path"
output_folder = "frames_output"  # Folder to store frames
frame_interval = 90  # Extract every 90th frame

# Run the process on all videos
process_all_videos(input_folder, output_folder, frame_interval)
