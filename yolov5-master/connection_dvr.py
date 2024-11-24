# connection_dvr.py

import cv2
import torch
import threading
from queue import Queue
import os
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import time
import logging
from dotenv import load_dotenv
import contextlib

# Load environment variables for sensitive data
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='connection_dvr.log',
    level=logging.INFO,  # You may set this to logging.DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set environment variable for FFMPEG (using TCP for stability)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

# Load the YOLO model
custom_weights_path = "runs/train/exp4/weights/best.pt"  # Path to your custom weights

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the custom YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_weights_path, force_reload=True)
    model.to(device)
    model.eval()
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    exit(1)

# Queue to store frames for the RTSP stream
frame_queue = Queue(maxsize=10)  # Adjust size based on memory constraints

# RTSP URL for the DVR camera stream (update as needed)
rtsp_url = "rtsp://smart:1234@192.168.1.207:554/avstream/channel=7/stream=0.sdp"

# Initialize a lock for database operations (optional)
db_lock = threading.Lock()

# Function to create a new database connection
def init_db():
    db_path = os.path.join('database', 'smart_building.db')
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        logging.info("Database connected successfully.")
        return conn, cursor
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to the database: {e}")
        exit(1)

# Function to log detection into the database
def log_detection(conn, cursor, item, confidence, timestamp, location, worker_id=None, reason='Detected'):
    try:
        cursor.execute('''
            INSERT INTO tokens (token_reason, token_location, token_assigned, token_status, token_start, token_end_time, confidence)
            VALUES (?, ?, ?, 'Pending', ?, NULL, ?)
        ''', (reason, location, worker_id, timestamp, confidence))
        token_id = cursor.lastrowid  # Get the last inserted token_id
        conn.commit()
        logging.info(f"Logged detection: {item} at {location} with confidence {confidence:.2f}, reason: {reason}, token_id: {token_id}")
        return token_id
    except Exception as e:
        logging.error(f"Error during logging detection: {e}", exc_info=True)
        return None

# Function to update token status to "Resolved" or other status
def update_token_status(conn, cursor, token_id, status='Resolved'):
    try:
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            UPDATE tokens
            SET token_status = ?, token_end_time = ?
            WHERE token_id = ?
        ''', (status, end_time, token_id))
        rows_updated = cursor.rowcount
        conn.commit()
        logging.info(f"Token ID {token_id} has been updated to status '{status}'. Rows affected: {rows_updated}")
        return rows_updated > 0
    except Exception as e:
        logging.error(f"Error during updating token status: {e}", exc_info=True)
        return False

# Function to send email notifications
def send_email_notification(worker_email, item, confidence, timestamp, location, status='Pending'):
    # Email configuration from environment variables
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')  # Replace with your SMTP server
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))  # Replace with your SMTP port (usually 587 for TLS)
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')  # Replace with your email
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')  # Replace with your email password or App Password

    FROM_EMAIL = os.getenv('FROM_EMAIL', SMTP_USERNAME)  # Sender email
    TO_EMAIL = worker_email  # Recipient email

    subject = f"Alert: {item} Detected - {status}"
    body = f"""
A {item} was detected.

Details:
- Item: {item}
- Confidence: {confidence:.2f}
- Time: {timestamp}
- Location: {location}
- Status: {status}
"""

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = FROM_EMAIL
    msg['To'] = TO_EMAIL

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())
            logging.info(f"Email notification sent to {TO_EMAIL} for {item} detected at {timestamp} with status '{status}'.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}", exc_info=True)

# Function to handle email notifications asynchronously
def notification_worker(notification_queue):
    while True:
        notification = notification_queue.get()
        if notification == "STOP":
            logging.info("Notification worker received stop signal.")
            break
        try:
            worker_email, item, confidence, timestamp, location, status = notification
            send_email_notification(worker_email, item, confidence, timestamp, location, status)
        except Exception as e:
            logging.error(f"Error processing notification: {e}", exc_info=True)
        finally:
            notification_queue.task_done()

# Function to assign a worker randomly and update their status to 'occupied'
def assign_worker(conn, cursor):
    try:
        # Select a free worker randomly
        cursor.execute('''
            SELECT worker_id, worker_email FROM workers
            WHERE status = 'free'
            ORDER BY RANDOM()
            LIMIT 1;
        ''')
        result = cursor.fetchone()
        if result:
            worker_id, worker_email = result
            # Update worker status to 'occupied'
            cursor.execute('''
                UPDATE workers
                SET status = 'occupied'
                WHERE worker_id = ?;
            ''', (worker_id,))
            conn.commit()
            logging.info(f"Assigned Worker ID: {worker_id}, Email: {worker_email}")
            return worker_id, worker_email
        else:
            logging.warning("No free workers available for assignment.")
            return None, None  # No workers available
    except Exception as e:
        logging.error(f"Error during worker assignment: {e}", exc_info=True)
        return None, None

# Function to release a worker (set status to 'free')
def release_worker(conn, cursor, worker_id):
    try:
        cursor.execute('''
            UPDATE workers
            SET status = 'free'
            WHERE worker_id = ?;
        ''', (worker_id,))
        rows_updated = cursor.rowcount
        conn.commit()
        logging.info(f"Released Worker ID: {worker_id}, set status to 'free'. Rows affected: {rows_updated}")
        return rows_updated > 0
    except Exception as e:
        logging.error(f"Error during releasing worker: {e}", exc_info=True)
        return False

# Define the list of target items
target_items = ["wallet"]  # Add more items as needed

# Parameters for detection logic
detection_duration_threshold = 10  # seconds
detection_gap_threshold = 2        # seconds
resolution_gap_threshold = 5       # seconds

# Initialize detection states
detection_state = {item: False for item in target_items}
detection_start_time = {item: None for item in target_items}
last_detection_time = {item: None for item in target_items}
detection_stop_time = {item: None for item in target_items}  # New variable
assigned_token = {item: None for item in target_items}

# Function to implement rate limiting for notifications
last_notification_time = {}
notification_cooldown = 60  # seconds

def should_send_notification(item, location):
    global last_notification_time
    key = f"{item}_{location}"
    current_time = time.time()
    if key in last_notification_time:
        elapsed = current_time - last_notification_time[key]
        if elapsed < notification_cooldown:
            logging.debug(f"Notification cooldown active for {item} at {location}.")
            return False
    last_notification_time[key] = current_time
    return True

# Function to process video from the RTSP stream
def process_rtsp_stream(rtsp_url, model, frame_queue, notification_queue):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    reconnect_delay = 5  # seconds
    max_reconnect_attempts = 5
    reconnect_attempts = 0

    # Initialize database connection for this thread
    conn, cursor = init_db()

    if not cap.isOpened():
        logging.error(f"Cannot open RTSP stream at {rtsp_url}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to retrieve frame from RTSP stream. Attempting to reconnect...")
            cap.release()
            reconnect_attempts += 1
            if reconnect_attempts > max_reconnect_attempts:
                logging.error("Max reconnect attempts reached. Exiting video processing thread.")
                break
            time.sleep(reconnect_delay)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logging.error(f"Cannot reopen RTSP stream at {rtsp_url}. Retrying in {reconnect_delay} seconds...")
                continue
            else:
                reconnect_attempts = 0  # Reset on successful reconnection
            continue

        reconnect_attempts = 0  # Reset on successful frame retrieval

        # Perform object detection on the frame
        with torch.cuda.amp.autocast() if device == 'cuda' else contextlib.nullcontext():
            results = model(frame)

        # Render the results on the frame
        annotated_frame = results.render()[0]

        # Get current time
        current_time = datetime.now()

        # Parse detections
        detected_items = {}
        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            confidence = float(conf)
            # Only consider target items with high enough confidence
            if label in target_items and confidence >= 0.5:  # Adjust confidence threshold as needed
                detected_items[label] = confidence

        # Update detection states
        for item in target_items:
            if item in detected_items:
                if not detection_state[item]:
                    detection_state[item] = True
                    if detection_start_time[item] is None:
                        detection_start_time[item] = current_time
                    detection_stop_time[item] = None  # Reset stop time
                    logging.info(f"Detection started for {item} at {current_time}.")
                last_detection_time[item] = current_time
            else:
                if detection_state[item]:
                    if last_detection_time[item]:
                        time_since_last_detection = (current_time - last_detection_time[item]).total_seconds()
                        if time_since_last_detection > detection_gap_threshold:
                            detection_state[item] = False
                            detection_start_time[item] = None
                            detection_stop_time[item] = current_time  # Set stop time
                            logging.info(f"Detection stopped for {item} at {current_time}.")
                    else:
                        # No last detection time, reset state
                        detection_state[item] = False
                        detection_start_time[item] = None
                        detection_stop_time[item] = current_time  # Set stop time

            # Check for raising tokens
            if detection_state[item]:
                total_detection_time = (current_time - detection_start_time[item]).total_seconds()
                if total_detection_time >= detection_duration_threshold and assigned_token[item] is None:
                    logging.info(f"{item} detected continuously for {detection_duration_threshold} seconds. Assigning token.")
                    worker_id, worker_email = assign_worker(conn, cursor)
                    if worker_id and worker_email:
                        token_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        token_id = log_detection(
                            conn, cursor,
                            item=item,
                            confidence=detected_items[item],
                            timestamp=token_timestamp,
                            location='Entrance Hall',  # Replace with dynamic location if needed
                            worker_id=worker_id,
                            reason='Detected'
                        )
                        if token_id:
                            assigned_token[item] = token_id
                            logging.info(f"Token {token_id} assigned to {item}.")
                        else:
                            logging.warning(f"Failed to log detection for {item}.")
                            continue  # Skip further processing for this item

                        if should_send_notification(item, 'Entrance Hall'):
                            notification_queue.put((
                                worker_email,
                                item,
                                detected_items[item],
                                token_timestamp,
                                'Entrance Hall',
                                'Pending'
                            ))
                    else:
                        logging.warning(f"No available workers to assign for {item} detection.")

            # Check for token resolution
            if assigned_token[item] is not None:
                if not detection_state[item]:
                    if detection_stop_time[item]:
                        time_since_detection_stop = (current_time - detection_stop_time[item]).total_seconds()
                        if time_since_detection_stop >= resolution_gap_threshold:
                            logging.info(f"{item} has been absent for {resolution_gap_threshold} seconds. Resolving token.")
                            try:
                                token_id = assigned_token[item]
                                logging.info(f"Attempting to resolve token {token_id}")
                                # Update token status
                                updated = update_token_status(conn, cursor, token_id, status='Resolved')
                                if not updated:
                                    logging.warning(f"Token {token_id} status update failed.")
                                    continue  # Skip further processing if update failed

                                # Get the worker assigned to this token
                                cursor.execute('''
                                    SELECT token_assigned FROM tokens
                                    WHERE token_id = ?
                                ''', (token_id,))
                                result = cursor.fetchone()
                                logging.info(f"Token {token_id} query result: {result}")
                                if result and result[0]:
                                    worker_assigned_id = result[0]
                                    logging.info(f"Worker assigned ID: {worker_assigned_id}")

                                    # Retrieve worker's email
                                    cursor.execute('SELECT worker_email FROM workers WHERE worker_id = ?', (worker_assigned_id,))
                                    worker_email_result = cursor.fetchone()
                                    logging.info(f"Worker email query result: {worker_email_result}")
                                    if worker_email_result:
                                        worker_email = worker_email_result[0]
                                        if should_send_notification(item, 'Entrance Hall'):
                                            notification_queue.put((
                                                worker_email,
                                                item,
                                                0.0,  # Confidence is 0 for resolved
                                                current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                                'Entrance Hall',
                                                'Resolved'
                                            ))
                                    # Release the worker
                                    released = release_worker(conn, cursor, worker_assigned_id)
                                    if not released:
                                        logging.warning(f"Failed to release worker ID {worker_assigned_id}.")
                                    else:
                                        logging.info(f"Token {token_id} for {item} has been resolved and worker released.")
                                else:
                                    logging.warning(f"No worker assigned to token {token_id}.")
                                # Reset token assignment and detection states
                                assigned_token[item] = None
                                detection_state[item] = False
                                detection_start_time[item] = None
                                last_detection_time[item] = None
                                detection_stop_time[item] = None  # Reset stop time
                            except Exception as e:
                                logging.error(f"Error during token resolution: {e}", exc_info=True)
                                continue

        # Put the annotated frame in the queue for display in the main thread
        if not frame_queue.full():
            frame_queue.put(annotated_frame)

    # Close the database connection when the thread finishes
    conn.close()

# Create a queue for notifications
notification_queue = Queue()

# Start the notification worker thread
notification_thread = threading.Thread(target=notification_worker, args=(notification_queue,))
notification_thread.start()

# Create and start a thread for processing the RTSP stream
detection_thread = threading.Thread(target=process_rtsp_stream, args=(rtsp_url, model, frame_queue, notification_queue))
detection_thread.daemon = True  # Daemonize thread to exit when main thread exits
detection_thread.start()

# Function to clean up resources
def cleanup():
    # Signal the notification worker to stop
    notification_queue.put("STOP")

    # Wait for queues to be processed
    notification_queue.join()

    # Wait for threads to finish
    detection_thread.join(timeout=5)
    notification_thread.join(timeout=5)

    cv2.destroyAllWindows()
    logging.info("Shutdown complete.")

# Main display loop with enhanced error handling
try:
    while True:
        # Display the frame from the RTSP stream
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow('RTSP Stream with YOLO', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exit command received. Shutting down.")
            break
except KeyboardInterrupt:
    logging.info("Interrupted by user. Shutting down.")
finally:
    cleanup()
