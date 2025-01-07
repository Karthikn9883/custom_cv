# connection_dvr_multi.py

import os
import cv2
import torch
import threading
import sqlite3
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import time
import logging
import signal
import sys
from queue import Queue
from dotenv import load_dotenv
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import contextlib

# ##############################################################################
# Configuration and Setup
# ##############################################################################

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    filename='connection_dvr_multi.log',
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Database path
DB_PATH = os.getenv('DB_PATH', 'database/smart_building.db')

def connect_db():
    """
    Create a new database connection.
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        logging.info("Database connected successfully.")
        return conn, conn.cursor()
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to the database: {e}")
        sys.exit(1)

# ##############################################################################
# YOLO Model Setup
# ##############################################################################
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

custom_weights_path = os.getenv('YOLO_WEIGHTS_PATH', 'runs/train/exp4/weights/best.pt')  # Path to your custom weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    # Load YOLOv5 model via torch hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=custom_weights_path, force_reload=True)
    model.to(device)
    model.eval()
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    sys.exit(1)

# ##############################################################################
# RTSP Streams Configuration
# ##############################################################################
# Define multiple RTSP URLs (camera_id -> URL). Update these as needed.
RTSP_URLS = {
    0: "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/102",
    1: "rtsp://admin:admin123@192.168.29.195:554/Streaming/Channels/202",
    2: "rtsp://admin:admin123@192.168.29.196:554/Streaming/Channels/302",
    3: "rtsp://admin:admin123@192.168.29.197:554/Streaming/Channels/402"
}

# Create one frame queue per camera
frame_queues = {}
for camera_id in RTSP_URLS:
    frame_queues[camera_id] = Queue(maxsize=10)

# ##############################################################################
# Detection States per Camera
# ##############################################################################
target_items = ["wallet"]  # Add more items as needed

# Initialize detection states for each camera
detection_state = {}
detection_start_time = {}
last_detection_time = {}
detection_stop_time = {}
assigned_token = {}

for cam_id in RTSP_URLS:
    detection_state[cam_id] = {item: False for item in target_items}
    detection_start_time[cam_id] = {item: None for item in target_items}
    last_detection_time[cam_id] = {item: None for item in target_items}
    detection_stop_time[cam_id] = {item: None for item in target_items}
    assigned_token[cam_id] = {item: None for item in target_items}

# Detection parameters
detection_duration_threshold = 10  # seconds
detection_gap_threshold = 2        # seconds
resolution_gap_threshold = 5       # seconds

# ##############################################################################
# Worker / Token Functions
# ##############################################################################
def assign_worker(conn, cursor):
    """
    Assign a free worker randomly.
    """
    try:
        cursor.execute('''
            SELECT worker_id, worker_email FROM workers
            WHERE status = 'free'
            ORDER BY RANDOM()
            LIMIT 1;
        ''')
        result = cursor.fetchone()
        if result:
            worker_id, worker_email = result
            cursor.execute('''
                UPDATE workers
                SET status = 'occupied'
                WHERE worker_id = ?;
            ''', (worker_id,))
            conn.commit()
            logging.info(f"Assigned Worker ID: {worker_id}, Email: {worker_email}")
            return worker_id, worker_email
        else:
            logging.warning("No free workers available.")
            return None, None
    except Exception as e:
        logging.error(f"Error during worker assignment: {e}", exc_info=True)
        return None, None

def release_worker(conn, cursor, worker_id):
    """
    Release a worker (set status to 'free').
    """
    try:
        cursor.execute('''
            UPDATE workers
            SET status = 'free'
            WHERE worker_id = ?;
        ''', (worker_id,))
        rows_updated = cursor.rowcount
        conn.commit()
        if rows_updated > 0:
            logging.info(f"Released Worker ID: {worker_id}, set status to 'free'.")
            return True
        else:
            logging.warning(f"No worker with ID {worker_id} found.")
            return False
    except Exception as e:
        logging.error(f"Error releasing worker: {e}", exc_info=True)
        return False

def log_detection(conn, cursor, item, confidence, timestamp, location, worker_id=None, reason='Detected'):
    """
    Log detection into the tokens table.
    """
    try:
        cursor.execute('''
            INSERT INTO tokens (token_reason, token_location, token_assigned, token_status, token_start, token_end_time, confidence)
            VALUES (?, ?, ?, 'Pending', ?, NULL, ?)
        ''', (reason, location, worker_id, timestamp, confidence))
        token_id = cursor.lastrowid
        conn.commit()
        logging.info(f"Logged detection: {item} at {location} with confidence {confidence:.2f}, token_id: {token_id}")
        return token_id
    except Exception as e:
        logging.error(f"Error logging detection: {e}", exc_info=True)
        return None

def update_token_status(conn, cursor, token_id, status='Resolved'):
    """
    Update token status in the tokens table.
    """
    try:
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            UPDATE tokens
            SET token_status = ?, token_end_time = ?
            WHERE token_id = ?
        ''', (status, end_time, token_id))
        rows_updated = cursor.rowcount
        conn.commit()
        logging.info(f"Token ID {token_id} updated to status '{status}'. Rows: {rows_updated}")
        return rows_updated > 0
    except Exception as e:
        logging.error(f"Error updating token status: {e}", exc_info=True)
        return False

# ##############################################################################
# Email Notification Functions
# ##############################################################################
def send_email_notification(worker_email, item, confidence, timestamp, location, status='Pending'):
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
    FROM_EMAIL = os.getenv('FROM_EMAIL', SMTP_USERNAME)
    TO_EMAIL = worker_email

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
            logging.info(f"Email sent to {TO_EMAIL} for {item} at {timestamp}, status={status}.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}", exc_info=True)

# ##############################################################################
# Notification Worker (Asynchronous Email Sending)
# ##############################################################################
notification_queue = Queue()

def notification_worker():
    while True:
        notification = notification_queue.get()
        if notification == "STOP":
            logging.info("Notification worker stop signal.")
            break
        try:
            worker_email, item, confidence, timestamp, location, status = notification
            send_email_notification(worker_email, item, confidence, timestamp, location, status)
        except Exception as e:
            logging.error(f"Error in notification queue: {e}", exc_info=True)
        finally:
            notification_queue.task_done()

notification_thread = threading.Thread(target=notification_worker, daemon=True)
notification_thread.start()

# ##############################################################################
# Rate Limiting for Notifications
# ##############################################################################
last_notification_time = {}
notification_cooldown = 60  # seconds

def should_send_notification(item, camera_id):
    """
    Rate limiting logic keyed by (item, camera_id).
    """
    global last_notification_time
    key = f"{camera_id}_{item}"
    current_time = time.time()
    if key in last_notification_time:
        elapsed = current_time - last_notification_time[key]
        if elapsed < notification_cooldown:
            logging.debug(f"Notification cooldown active for {item} at camera {camera_id}.")
            return False
    last_notification_time[key] = current_time
    return True

# ##############################################################################
# RTSP Stream Processing (Multi-Camera)
# ##############################################################################
def process_rtsp_stream(camera_id):
    """
    Process a single RTSP stream (camera_id), read frames, run YOLO, handle tokens.
    """
    rtsp_url = RTSP_URLS[camera_id]
    frame_queue = frame_queues[camera_id]

    conn, cursor = connect_db()

    reconnect_delay = 5  # seconds
    max_reconnect_attempts = 5
    reconnect_attempts = 0

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logging.error(f"Camera {camera_id}: Cannot open RTSP stream: {rtsp_url}")
        return

    logging.info(f"Camera {camera_id}: Started processing RTSP stream.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Camera {camera_id}: Failed to retrieve frame. Attempting to reconnect...")
            cap.release()
            reconnect_attempts += 1
            if reconnect_attempts > max_reconnect_attempts:
                logging.error(f"Camera {camera_id}: Max reconnect attempts reached. Exiting thread.")
                break
            time.sleep(reconnect_delay)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                logging.error(f"Camera {camera_id}: Cannot reopen RTSP stream. Retrying...")
                continue
            else:
                reconnect_attempts = 0
            continue

        reconnect_attempts = 0  # Reset after successful frame read

        # Perform YOLO inference
        try:
            with torch.cuda.amp.autocast() if device == 'cuda' else contextlib.nullcontext():
                results = model(frame)
        except Exception as e:
            logging.error(f"Camera {camera_id}: YOLO inference error: {e}", exc_info=True)
            continue

        # Render results on frame
        try:
            annotated_frame = results.render()[0]
        except Exception as e:
            logging.error(f"Camera {camera_id}: Error rendering YOLO results: {e}", exc_info=True)
            annotated_frame = frame

        # Get current time
        current_time = datetime.now()

        # Parse detections
        detected_items = {}
        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            confidence = float(conf)
            if label in target_items and confidence >= 0.5:
                detected_items[label] = confidence

        for item in target_items:
            if item in detected_items:
                # If item newly detected
                if not detection_state[camera_id][item]:
                    detection_state[camera_id][item] = True
                    if detection_start_time[camera_id][item] is None:
                        detection_start_time[camera_id][item] = current_time
                    detection_stop_time[camera_id][item] = None
                    logging.info(f"Camera {camera_id}: Detection started for {item} at {current_time}.")
                last_detection_time[camera_id][item] = current_time
            else:
                # If item was detected but no longer visible
                if detection_state[camera_id][item]:
                    if last_detection_time[camera_id][item]:
                        gap = (current_time - last_detection_time[camera_id][item]).total_seconds()
                        if gap > detection_gap_threshold:
                            detection_state[camera_id][item] = False
                            detection_start_time[camera_id][item] = None
                            detection_stop_time[camera_id][item] = current_time
                            logging.info(f"Camera {camera_id}: Detection stopped for {item} at {current_time}.")

            # Check if token needs to be raised
            if detection_state[camera_id][item]:
                total_time = (current_time - detection_start_time[camera_id][item]).total_seconds()
                if total_time >= detection_duration_threshold and assigned_token[camera_id][item] is None:
                    logging.info(f"Camera {camera_id}: {item} detected for {detection_duration_threshold}s. Raising token.")
                    worker_id, worker_email = assign_worker(conn, cursor)
                    if worker_id and worker_email:
                        token_timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        confidence = detected_items[item]
                        token_id = log_detection(conn, cursor,
                                                 item=item,
                                                 confidence=confidence,
                                                 timestamp=token_timestamp,
                                                 location=f"Camera {camera_id}",
                                                 worker_id=worker_id)
                        if token_id:
                            assigned_token[camera_id][item] = token_id
                            logging.info(f"Camera {camera_id}: Token {token_id} assigned to {item}.")
                            if should_send_notification(item, camera_id):
                                notification_queue.put((
                                    worker_email,
                                    item,
                                    confidence,
                                    token_timestamp,
                                    f"Camera {camera_id}",
                                    'Pending'
                                ))
                        else:
                            logging.warning(f"Camera {camera_id}: Failed to log detection for {item}.")
                    else:
                        logging.warning(f"Camera {camera_id}: No available workers to assign for {item} detection.")

            # Check if token needs to be resolved
            if assigned_token[camera_id][item] is not None and not detection_state[camera_id][item]:
                if detection_stop_time[camera_id][item]:
                    absent_time = (current_time - detection_stop_time[camera_id][item]).total_seconds()
                    if absent_time >= resolution_gap_threshold:
                        logging.info(f"Camera {camera_id}: {item} has been absent for {resolution_gap_threshold}s. Resolving token.")
                        token_id = assigned_token[camera_id][item]
                        updated = update_token_status(conn, cursor, token_id, status='Resolved')
                        if updated:
                            # Find the worker assigned to this token
                            cursor.execute('SELECT token_assigned FROM tokens WHERE token_id = ?', (token_id,))
                            row = cursor.fetchone()
                            if row and row[0]:
                                worker_assigned_id = row[0]
                                # Retrieve worker's email
                                cursor.execute('SELECT worker_email FROM workers WHERE worker_id = ?', (worker_assigned_id,))
                                worker_email_result = cursor.fetchone()
                                if worker_email_result:
                                    worker_email = worker_email_result[0]
                                    if should_send_notification(item, camera_id):
                                        notification_queue.put((
                                            worker_email,
                                            item,
                                            0.0,  # Confidence for Resolved
                                            current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                            f"Camera {camera_id}",
                                            'Resolved'
                                        ))
                            # Release the worker
                            release_worker(conn, cursor, worker_assigned_id)
                        else:
                            logging.warning(f"Camera {camera_id}: Failed to update token {token_id} status.")
                        # Reset token assignment and detection states
                        assigned_token[camera_id][item] = None
                        detection_state[camera_id][item] = False
                        detection_start_time[camera_id][item] = None
                        last_detection_time[camera_id][item] = None
                        detection_stop_time[camera_id][item] = None

        # Enqueue the annotated frame for Flask video feed
        if not frame_queue.full():
            frame_queue.put(annotated_frame)
            logging.debug(f"Camera {camera_id}: Frame enqueued. Queue size: {frame_queue.qsize()}")
        else:
            logging.debug(f"Camera {camera_id}: Frame queue full. Dropping frame.")

    # ##############################################################################
    # Flask Routes for Video Feeds and Other Functionalities
    # ##############################################################################
    @app.route('/video_feed/<int:camera_id>')
    def video_feed(camera_id):
        """
        Return the live video feed for a specific camera.
        """
        if camera_id not in RTSP_URLS:
            return jsonify({"error": "Invalid camera ID"}), 400

        def generate_frames():
            q = frame_queues[camera_id]
            while True:
                try:
                    frame = q.get(timeout=10)  # Wait for a frame
                except:
                    continue  # No frame available, continue waiting
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/cameras', methods=['GET'])
    def get_cameras():
        try:
            conn, cursor = connect_db()
            cursor.execute("SELECT * FROM cameras")
            rows = cursor.fetchall()
            cameras = [dict(row) for row in rows]
            return jsonify(cameras), 200
        except Exception as e:
            logging.error(f"Error fetching cameras: {e}", exc_info=True)
            return {"error": str(e)}, 500
        finally:
            conn.close()

    @app.route('/cameras', methods=['POST'])
    def save_cameras():
        try:
            cameras = request.json  # Expecting a list of cameras
            conn, cursor = connect_db()
            cursor.execute("DELETE FROM cameras")  # Clear existing data before saving
            for camera in cameras:
                cursor.execute('''
                    INSERT INTO cameras (camera_id, camera_name, x_coordinate, y_coordinate, angle, room_no, floor)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    camera['camera_id'],
                    camera.get('camera_name', f'Camera {camera["camera_id"]}'),
                    camera['x_coordinate'],
                    camera['y_coordinate'],
                    camera['angle'],
                    camera.get('room_no', 'Unknown'),
                    camera.get('floor', 0)
                ))
            conn.commit()
            logging.info("Cameras saved successfully.")
            return {"message": "Cameras saved successfully."}, 200
        except Exception as e:
            logging.error(f"Error saving cameras: {e}", exc_info=True)
            return {"error": str(e)}, 500
        finally:
            conn.close()

    @app.route('/workers', methods=['GET'])
    def get_workers():
        try:
            conn, cursor = connect_db()
            cursor.execute("SELECT * FROM workers")
            rows = cursor.fetchall()
            workers = [dict(row) for row in rows]
            return jsonify(workers), 200
        except Exception as e:
            logging.error(f"Error fetching workers: {e}", exc_info=True)
            return {"error": str(e)}, 500
        finally:
            conn.close()

    @app.route('/workers', methods=['POST'])
    def add_worker():
        try:
            data = request.json
            conn, cursor = connect_db()
            cursor.execute('''
                INSERT INTO workers (worker_name, worker_number, worker_email, status)
                VALUES (?, ?, ?, 'free')
            ''', (data['name'], data['number'], data['email']))
            conn.commit()
            logging.info("Worker added successfully.")
            return {"message": "Worker added successfully."}, 201
        except Exception as e:
            logging.error(f"Error adding worker: {e}", exc_info=True)
            return {"error": str(e)}, 500
        finally:
            conn.close()

    @app.route('/workers/<int:worker_id>', methods=['PUT'])
    def update_worker(worker_id):
        try:
            data = request.json
            conn, cursor = connect_db()
            cursor.execute('''
                UPDATE workers
                SET worker_name = ?, worker_number = ?, worker_email = ?, status = ?
                WHERE worker_id = ?
            ''', (data['name'], data['number'], data['email'], data['status'], worker_id))
            conn.commit()
            if cursor.rowcount > 0:
                logging.info("Worker updated successfully.")
                return {"message": "Worker updated successfully."}, 200
            else:
                logging.warning("Worker not found.")
                return {"error": "Worker not found."}, 404
        except Exception as e:
            logging.error(f"Error updating worker: {e}", exc_info=True)
            return {"error": str(e)}, 500
        finally:
            conn.close()

    @app.route('/workers/<int:worker_id>', methods=['DELETE'])
    def delete_worker(worker_id):
        try:
            conn, cursor = connect_db()
            cursor.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))
            conn.commit()
            if cursor.rowcount > 0:
                logging.info("Worker deleted successfully.")
                return {"message": "Worker deleted successfully."}, 200
            else:
                logging.warning("Worker not found.")
                return {"error": "Worker not found."}, 404
        except Exception as e:
            logging.error(f"Error deleting worker: {e}", exc_info=True)
            return {"error": str(e)}, 500
        finally:
            conn.close()

    # ##############################################################################
    # Camera Status Monitoring Route
    # ##############################################################################
    @app.route('/camera_status', methods=['GET'])
    def camera_status():
        """
        API endpoint to check the running status of each camera.
        """
        status = {}
        for cam_id, thread in camera_threads.items():
            status[cam_id] = 'Running' if thread.is_alive() else 'Stopped'
        return jsonify(status), 200

    # ##############################################################################
    # Flask Application Runner
    # ##############################################################################
    def run_flask():
        """
        Runs the Flask application.
        """
        app.run(host='0.0.0.0', port=5001, debug=False)

    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logging.info("Flask server started on port 5001.")

    # ##############################################################################
    # Start Detection Threads for Each Camera
    # ##############################################################################
    camera_threads = {}
    for cam_id in RTSP_URLS:
        t = threading.Thread(target=process_rtsp_stream, args=(cam_id,), daemon=True)
        t.start()
        camera_threads[cam_id] = t
        logging.info(f"Started detection thread for camera {cam_id}.")

    # ##############################################################################
    # Cleanup and Graceful Shutdown
    # ##############################################################################
    def cleanup():
        """
        Cleans up resources and stops threads gracefully.
        """
        logging.info("Initiating cleanup...")
        # Stop the notification worker
        notification_queue.put("STOP")
        notification_queue.join()

        # Attempt to join detection threads
        for cam_id, thread in camera_threads.items():
            thread.join(timeout=2)
            if thread.is_alive():
                logging.warning(f"Camera {cam_id}: Detection thread did not terminate properly.")

        # Close all OpenCV windows
        cv2.destroyAllWindows()
        logging.info("Cleanup complete.")

    def signal_handler(sig, frame):
        """
        Handles termination signals for graceful shutdown.
        """
        logging.info(f"Signal {sig} received. Shutting down...")
        cleanup()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ##############################################################################
    # Main Loop for OpenCV Display (Optional)
    # ##############################################################################
    try:
        while True:
            # Display frames for each camera in separate OpenCV windows
            for cam_id in RTSP_URLS:
                q = frame_queues[cam_id]
                if not q.empty():
                    frame = q.get()
                    try:
                        cv2.imshow(f"Camera {cam_id}", frame)
                        # Optionally, handle 'q' key for shutdown
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            logging.info("Exit command received via OpenCV window. Shutting down.")
                            raise KeyboardInterrupt
                    except Exception as e:
                        logging.error(f"Error displaying frame for Camera {cam_id}: {e}", exc_info=True)
            time.sleep(0.01)  # Small sleep to prevent high CPU usage
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Initiating shutdown...")
    finally:
        cleanup()
