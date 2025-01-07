# # connection_dvr.py

# import cv2
# import torch
# import threading
# from queue import Queue
# import os
# import sqlite3
# import smtplib
# from email.mime.text import MIMEText
# from datetime import datetime
# import time
# import logging
# from dotenv import load_dotenv
# import contextlib
# from flask import Flask, Response, request, jsonify
# from flask_cors import CORS

# ###############################################################################
# #                             Flask & Configuration                           #
# ###############################################################################

# app = Flask(__name__)
# CORS(app)

# # Load environment variables from .env file
# load_dotenv()

# # Database path
# DB_PATH = os.getenv('DB_PATH', 'database/smart_building.db')

# # Set up logging with reduced verbosity to minimize overhead
# logging.basicConfig(
#     filename='connection_dvr.log',
#     level=logging.INFO,  # Changed from DEBUG to INFO for better performance
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# ###############################################################################
# #                          Utility / DB Connection                            #
# ###############################################################################

# def connect_db():
#     """
#     Creates a new database connection (with row dict factory).
#     Each thread should call this to get its own connection.
#     """
#     try:
#         conn = sqlite3.connect(DB_PATH, check_same_thread=False)
#         conn.row_factory = sqlite3.Row  # Enable dictionary-like row access
#         logging.debug("Database connection established.")
#         return conn
#     except sqlite3.Error as e:
#         logging.error(f"Failed to connect to database: {e}")
#         raise e

# def init_db():
#     """
#     Initialize or connect to the DB and return (conn, cursor).
#     """
#     try:
#         conn = connect_db()
#         cursor = conn.cursor()
#         logging.debug("Database cursor initialized.")
#         return conn, cursor
#     except sqlite3.Error as e:
#         logging.error(f"Failed to initialize DB: {e}")
#         raise e

# ###############################################################################
# #                              Flask Endpoints                                #
# ###############################################################################

# @app.route('/video_feed/<int:camera_id>')
# def video_feed(camera_id):
#     """
#     Returns an MJPEG stream for a specific camera.
#     (Raw from the camera, no YOLO overlay.)
#     """
#     # Define camera streams (you can fetch from DB if preferred)
#     camera_streams = {
#         1: "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/102?rtsp_transport=tcp",
#         2: "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/202?rtsp_transport=tcp",
#         3: "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/302?rtsp_transport=tcp",
#         4: "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/402?rtsp_transport=tcp"
#         # Add more cameras as needed
#     }

#     rtsp_url = camera_streams.get(camera_id)
#     if not rtsp_url:
#         logging.warning(f"Camera ID {camera_id} not found.")
#         return f"Camera ID {camera_id} not found.", 404

#     def generate():
#         cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#         if not cap.isOpened():
#             logging.error(f"Cannot open RTSP for camera {camera_id}.")
#             return

#         while True:
#             success, frame = cap.read()
#             if not success:
#                 logging.warning(f"Camera {camera_id}: No frame received.")
#                 break

#             # Encode as JPEG
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 logging.warning(f"Camera {camera_id}: Could not encode frame.")
#                 continue

#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' +
#                    buffer.tobytes() +
#                    b'\r\n')
#         cap.release()

#     return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/token_counts', methods=['GET'])
# def get_token_counts():
#     """
#     Return counts of 'Pending' vs 'Resolved' tokens.
#     """
#     try:
#         conn, cursor = init_db()
#         cursor.execute("SELECT COUNT(*) FROM tokens WHERE token_status='Pending'")
#         pending = cursor.fetchone()[0]

#         cursor.execute("SELECT COUNT(*) FROM tokens WHERE token_status='Resolved'")
#         resolved = cursor.fetchone()[0]

#         conn.close()
#         logging.debug(f"Token counts fetched: Pending={pending}, Resolved={resolved}")
#         return {"pending": pending, "resolved": resolved}, 200
#     except Exception as e:
#         logging.error(f"Error in /token_counts: {e}", exc_info=True)
#         return {"error": str(e)}, 500

# ###############################################################################
# #                           Camera & Worker Endpoints                         #
# ###############################################################################

# @app.route('/cameras', methods=['GET'])
# def get_cameras():
#     """
#     Returns a list of all cameras in the DB table 'cameras'.
#     """
#     try:
#         conn = connect_db()
#         cursor = conn.cursor()
#         cursor.execute("SELECT * FROM cameras")
#         rows = cursor.fetchall()
#         cameras = [dict(row) for row in rows]
#         logging.debug(f"Fetched {len(cameras)} cameras from the database.")
#         return jsonify(cameras), 200
#     except Exception as e:
#         logging.error(f"Error fetching cameras: {e}", exc_info=True)
#         return {"error": str(e)}, 500
#     finally:
#         conn.close()

# @app.route('/cameras', methods=['POST'])
# def save_cameras():
#     """
#     Saves (replaces) the entire cameras table with the provided list of cameras.
#     """
#     try:
#         cameras = request.json
#         conn = connect_db()
#         cursor = conn.cursor()
#         cursor.execute("DELETE FROM cameras")  # Clear existing data
#         logging.debug("Existing cameras deleted from the database.")

#         for c in cameras:
#             cursor.execute("""
#                 INSERT INTO cameras (camera_name, x_coordinate, y_coordinate, angle, room_no, floor)
#                 VALUES (?, ?, ?, ?, ?, ?)
#             """, (
#                 c.get('camera_name', f"Camera {c.get('camera_id', '')}"),
#                 c['x_coordinate'],
#                 c['y_coordinate'],
#                 c['angle'],
#                 c.get('room_no', 'Unknown'),
#                 c.get('floor', 0)
#             ))
#         conn.commit()
#         logging.info(f"Saved {len(cameras)} cameras to the database.")
#         return {"message": "Cameras saved successfully"}, 200
#     except Exception as e:
#         logging.error(f"Error saving cameras: {e}", exc_info=True)
#         return {"error": str(e)}, 500
#     finally:
#         conn.close()

# @app.route('/workers', methods=['GET'])
# def get_workers():
#     """
#     Fetch all workers from the database.
#     """
#     try:
#         conn = connect_db()
#         cursor = conn.cursor()
#         cursor.execute("SELECT * FROM workers")
#         rows = cursor.fetchall()
#         workers = [dict(row) for row in rows]
#         logging.debug(f"Fetched {len(workers)} workers from the database.")
#         return jsonify(workers), 200
#     except Exception as e:
#         logging.error(f"Error fetching workers: {e}", exc_info=True)
#         return {"error": str(e)}, 500
#     finally:
#         conn.close()

# @app.route('/workers', methods=['POST'])
# def add_worker():
#     """
#     Add a new worker to the database.
#     """
#     try:
#         data = request.json
#         conn = connect_db()
#         cursor = conn.cursor()
#         cursor.execute("""
#             INSERT INTO workers (worker_name, worker_number, worker_email)
#             VALUES (?, ?, ?)
#         """, (data['name'], data['number'], data['email']))
#         conn.commit()
#         worker_id = cursor.lastrowid
#         logging.info(f"Added new worker: ID={worker_id}, Email={data['email']}.")
#         return {"message": "Worker added successfully.", "worker_id": worker_id}, 201
#     except sqlite3.IntegrityError as ie:
#         logging.error(f"Integrity Error adding worker: {ie}", exc_info=True)
#         return {"error": "Worker with this email already exists."}, 400
#     except Exception as e:
#         logging.error(f"Error adding worker: {e}", exc_info=True)
#         return {"error": str(e)}, 500
#     finally:
#         conn.close()

# @app.route('/workers/<int:worker_id>', methods=['PUT'])
# def update_worker(worker_id):
#     """
#     Update an existing worker's status or details.
#     """
#     try:
#         data = request.json
#         conn = connect_db()
#         cursor = conn.cursor()
#         cursor.execute("""
#             UPDATE workers
#             SET worker_name=?, worker_number=?, worker_email=?, status=?
#             WHERE worker_id=?
#         """, (data['name'], data['number'], data['email'], data['status'], worker_id))
#         conn.commit()
#         if cursor.rowcount == 0:
#             logging.warning(f"Worker ID {worker_id} not found for update.")
#             return {"error": "Worker not found."}, 404
#         logging.info(f"Worker ID {worker_id} updated successfully.")
#         return {"message": "Worker updated successfully."}, 200
#     except sqlite3.IntegrityError as ie:
#         logging.error(f"Integrity Error updating worker: {ie}", exc_info=True)
#         return {"error": "Worker with this email already exists."}, 400
#     except Exception as e:
#         logging.error(f"Error updating worker: {e}", exc_info=True)
#         return {"error": str(e)}, 500
#     finally:
#         conn.close()

# @app.route('/workers/<int:worker_id>', methods=['DELETE'])
# def delete_worker(worker_id):
#     """
#     Delete a worker from the database.
#     """
#     try:
#         conn = connect_db()
#         cursor = conn.cursor()
#         cursor.execute("DELETE FROM workers WHERE worker_id=?", (worker_id,))
#         conn.commit()
#         if cursor.rowcount == 0:
#             logging.warning(f"Worker ID {worker_id} not found for deletion.")
#             return {"error": "Worker not found."}, 404
#         logging.info(f"Worker ID {worker_id} deleted successfully.")
#         return {"message": "Worker deleted successfully."}, 200
#     except Exception as e:
#         logging.error(f"Error deleting worker: {e}", exc_info=True)
#         return {"error": str(e)}, 500
#     finally:
#         conn.close()

# ###############################################################################
# #                           Token / Worker Logic                              #
# ###############################################################################

# def assign_worker(conn, cursor):
#     """
#     Assign the least-recently-assigned free worker, mark them as 'occupied'.
#     """
#     try:
#         cursor.execute("""
#             SELECT worker_id, worker_email
#             FROM workers
#             WHERE status='free'
#             ORDER BY last_assigned ASC NULLS FIRST
#             LIMIT 1;
#         """)
#         row = cursor.fetchone()
#         if row:
#             worker_id, worker_email = row['worker_id'], row['worker_email']
#             cursor.execute("""
#                 UPDATE workers
#                 SET status='occupied', last_assigned=CURRENT_TIMESTAMP
#                 WHERE worker_id=?;
#             """, (worker_id,))
#             conn.commit()
#             logging.info(f"Assigned worker {worker_id} ({worker_email}).")
#             return worker_id, worker_email
#         else:
#             logging.warning("No free workers available for assignment.")
#             return None, None
#     except Exception as e:
#         logging.error(f"Error in assign_worker: {e}", exc_info=True)
#         return None, None

# def release_worker(conn, cursor, worker_id):
#     """
#     Release the worker (mark as 'free').
#     """
#     try:
#         cursor.execute("""
#             UPDATE workers
#             SET status='free'
#             WHERE worker_id=?;
#         """, (worker_id,))
#         updated = cursor.rowcount
#         conn.commit()
#         if updated:
#             logging.info(f"Released worker {worker_id} and marked as 'free'.")
#             return True
#         else:
#             logging.warning(f"Attempted to release worker {worker_id}, but no rows were updated.")
#             return False
#     except Exception as e:
#         logging.error(f"Error in release_worker: {e}", exc_info=True)
#         return False

# ###############################################################################
# #                     Database Logging & Notification Logic                   #
# ###############################################################################

# def log_detection(conn, cursor, item, confidence, timestamp, location, worker_id=None, reason='Detected'):
#     """
#     Log a detection event into the 'tokens' table.
#     """
#     try:
#         cursor.execute("""
#             INSERT INTO tokens (token_reason, token_location, token_assigned, token_status, token_start, token_end_time, confidence)
#             VALUES (?, ?, ?, 'Pending', ?, NULL, ?)
#         """, (reason, location, worker_id, timestamp, confidence))
#         token_id = cursor.lastrowid
#         conn.commit()
#         logging.info(f"Logged detection: {item} at {location} with confidence={confidence:.2f}, reason={reason}, token_id={token_id}")
#         return token_id
#     except Exception as e:
#         logging.error(f"Error during logging detection: {e}", exc_info=True)
#         return None

# def update_token_status(conn, cursor, token_id, status='Resolved'):
#     """
#     Update a token's status.
#     """
#     try:
#         end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         cursor.execute("""
#             UPDATE tokens
#             SET token_status=?, token_end_time=?
#             WHERE token_id=?;
#         """, (status, end_time, token_id))
#         rows_updated = cursor.rowcount
#         conn.commit()
#         if rows_updated:
#             logging.info(f"Token {token_id} updated to '{status}' with end_time={end_time}.")
#             return True
#         else:
#             logging.warning(f"Token {token_id} not found for updating.")
#             return False
#     except Exception as e:
#         logging.error(f"Error updating token status: {e}", exc_info=True)
#         return False

# def send_email_notification(worker_email, item, confidence, timestamp, location, status='Pending'):
#     """
#     Send an email to the assigned worker about a detection or resolution.
#     """
#     SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
#     SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
#     SMTP_USERNAME = os.getenv('SMTP_USERNAME')
#     SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
#     FROM_EMAIL = os.getenv('FROM_EMAIL', SMTP_USERNAME)
#     TO_EMAIL = worker_email

#     subject = f"Alert: {item} Detected - {status}"
#     body = f"""
# A {item} was detected.

# Details:
# - Item: {item}
# - Confidence: {confidence:.2f}
# - Time: {timestamp}
# - Location: {location}
# - Status: {status}
# """

#     msg = MIMEText(body)
#     msg['Subject'] = subject
#     msg['From'] = FROM_EMAIL
#     msg['To'] = TO_EMAIL

#     try:
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.ehlo()
#             server.starttls()
#             server.ehlo()
#             server.login(SMTP_USERNAME, SMTP_PASSWORD)
#             server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())
#             logging.info(f"Email notification sent to {TO_EMAIL} for {item} at {timestamp} with status='{status}'.")
#     except Exception as e:
#         logging.error(f"Failed to send email: {e}", exc_info=True)

# def notification_worker(notification_queue):
#     """
#     Thread worker to asynchronously process and send email notifications.
#     """
#     while True:
#         notif = notification_queue.get()
#         if notif == "STOP":
#             logging.info("Notification worker stopping.")
#             break
#         try:
#             worker_email, item, confidence, timestamp, location, status = notif
#             send_email_notification(worker_email, item, confidence, timestamp, location, status)
#         except Exception as e:
#             logging.error(f"Notification worker error: {e}", exc_info=True)
#         finally:
#             notification_queue.task_done()

# ###############################################################################
# #                          YOLO & RTSP Processing                             #
# ###############################################################################

# # Path to your custom YOLO model
# model_path = '/Users/karthiknutulapati/Desktop/smartbuilding/cv+v5/yolov5-master/runs/train/exp12/weights/best.pt'

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# try:
#     # Load the custom YOLO model using torch hub
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
#     model.to(device)
#     model.eval()
#     logging.info(f"Custom YOLO model loaded successfully from '{model_path}' on {device}.")
# except Exception as e:
#     logging.error(f"Error loading custom YOLO model: {e}")
#     exit(1)

# # Items you want to detect
# target_items = ["wallet", "spill"]  # Extended to include 'spill'

# # Thresholds for detection logic (in seconds)
# detection_duration_threshold = 10  # Time item must be continuously visible to raise a token
# detection_gap_threshold = 2        # Time item must be absent to consider detection stopped
# resolution_gap_threshold = 5       # Time item must be absent to resolve the token

# # Rate-limiting notifications
# last_notification_time = {}
# notification_cooldown = 60  # seconds

# def should_send_notification(item, location):
#     """
#     Rate-limit notifications on a per-item+location basis.
#     """
#     global last_notification_time
#     key = f"{item}_{location}"
#     now = time.time()
#     if key in last_notification_time:
#         elapsed = now - last_notification_time[key]
#         if elapsed < notification_cooldown:
#             logging.debug(f"Notification cooldown active for {item} at {location}.")
#             return False
#     last_notification_time[key] = now
#     return True

# def process_rtsp_stream(rtsp_url, model, frame_queue, notification_queue):
#     """
#     Thread function that reads RTSP frames, runs YOLO, and manages detection states.
#     Each camera has its own detection dictionaries so tokens are raised independently.
#     """
#     # Per-camera detection states
#     detection_state = {item: False for item in target_items}
#     detection_start_time = {item: None for item in target_items}
#     last_detection_time = {item: None for item in target_items}
#     detection_stop_time = {item: None for item in target_items}
#     assigned_token = {item: None for item in target_items}

#     reconnect_attempts = 0
#     max_reconnect_attempts = 5
#     reconnect_delay = 5  # seconds

#     # Frame skipping to reduce lag
#     frame_count = 0
#     process_every_n_frames = 3  # OPTIMIZATION: Process every 3rd frame

#     # Lower inference size for faster processing
#     INFERENCE_SIZE = 416  # OPTIMIZATION: Reduced from 640 to 416

#     # Initialize DB connection for this thread
#     try:
#         conn, cursor = init_db()
#     except Exception as e:
#         logging.critical(f"Failed to initialize DB for thread handling {rtsp_url}: {e}")
#         return

#     while reconnect_attempts <= max_reconnect_attempts:
#         cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#         if not cap.isOpened():
#             logging.error(f"[{rtsp_url}] Cannot open stream. Retrying in {reconnect_delay}s...")
#             reconnect_attempts += 1
#             time.sleep(reconnect_delay)
#             continue
#         else:
#             logging.info(f"[{rtsp_url}] Stream opened successfully.")
#             reconnect_attempts = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logging.warning(f"[{rtsp_url}] No frame received. Attempting reconnect...")
#                 cap.release()
#                 reconnect_attempts += 1
#                 if reconnect_attempts > max_reconnect_attempts:
#                     logging.error(f"[{rtsp_url}] Exceeded max reconnect attempts. Exiting thread.")
#                     break
#                 time.sleep(reconnect_delay)
#                 break

#             frame_count += 1

#             # Only run inference on every n-th frame to reduce load
#             if frame_count % process_every_n_frames != 0:
#                 continue

#             # YOLO inference with optimized size
#             try:
#                 with torch.cuda.amp.autocast() if device == 'cuda' else contextlib.nullcontext():
#                     results = model(frame, size=INFERENCE_SIZE)  # OPTIMIZATION: Reduced size
#                 annotated_frame = results.render()[0]
#             except Exception as e:
#                 logging.error(f"[{rtsp_url}] Error in YOLO inference: {e}", exc_info=True)
#                 continue

#             current_time = datetime.now()
#             detected_items = {}

#             # Gather YOLO detections
#             for *xyxy, conf, cls_idx in results.xyxy[0]:
#                 label = model.names[int(cls_idx)]
#                 confidence = float(conf)
#                 if label in target_items and confidence >= 0.5:
#                     detected_items[label] = confidence

#             # Per-item detection logic
#             for item in target_items:
#                 if item in detected_items:
#                     # If newly detected
#                     if not detection_state[item]:
#                         detection_state[item] = True
#                         detection_start_time[item] = current_time
#                         detection_stop_time[item] = None
#                         logging.info(f"[{rtsp_url}] Detection started for '{item}' at {current_time}.")
#                     last_detection_time[item] = current_time
#                 else:
#                     # If item was being detected, check if it disappeared
#                     if detection_state[item]:
#                         if last_detection_time[item]:
#                             elapsed_gone = (current_time - last_detection_time[item]).total_seconds()
#                             if elapsed_gone > detection_gap_threshold:
#                                 detection_state[item] = False
#                                 detection_start_time[item] = None
#                                 detection_stop_time[item] = current_time
#                                 logging.info(f"[{rtsp_url}] Detection stopped for '{item}' at {current_time}.")
#                         else:
#                             # No last_detection_time, forcibly reset
#                             detection_state[item] = False
#                             detection_start_time[item] = None
#                             detection_stop_time[item] = current_time

#                 # Check if we should assign a token (continuous detection)
#                 if detection_state[item]:
#                     total_time_detected = (current_time - detection_start_time[item]).total_seconds()
#                     if total_time_detected >= detection_duration_threshold and assigned_token[item] is None:
#                         logging.info(f"[{rtsp_url}] '{item}' detected continuously for {detection_duration_threshold}s. Raising token.")
#                         worker_id, worker_email = assign_worker(conn, cursor)
#                         if worker_id and worker_email:
#                             ts_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
#                             # Check if 'item' is in detected_items before accessing
#                             if item in detected_items:
#                                 confidence = detected_items[item]
#                                 token_id = log_detection(
#                                     conn, cursor,
#                                     item=item,
#                                     confidence=confidence,
#                                     timestamp=ts_str,
#                                     location=rtsp_url,  # Use rtsp_url to identify camera/location
#                                     worker_id=worker_id
#                                 )
#                                 if token_id:
#                                     assigned_token[item] = token_id
#                                     if should_send_notification(item, rtsp_url):
#                                         notification_queue.put((
#                                             worker_email,
#                                             item,
#                                             confidence,
#                                             ts_str,
#                                             rtsp_url,
#                                             'Pending'
#                                         ))
#                                     logging.debug(f"[{rtsp_url}] Assigned token ID {token_id} to worker ID {worker_id}.")
#                                 else:
#                                     logging.warning(f"[{rtsp_url}] Failed to log detection for '{item}'.")
#                             else:
#                                 logging.warning(f"[{rtsp_url}] '{item}' not detected during token assignment.")
#                         else:
#                             logging.warning(f"[{rtsp_url}] No free workers available to assign for '{item}' detection.")

#                 # Check for token resolution
#                 if assigned_token[item] is not None and not detection_state[item]:
#                     # If detection stopped, see if we pass resolution threshold
#                     if detection_stop_time[item]:
#                         absent_time = (current_time - detection_stop_time[item]).total_seconds()
#                         if absent_time >= resolution_gap_threshold:
#                             token_id = assigned_token[item]
#                             logging.info(f"[{rtsp_url}] '{item}' absent for {resolution_gap_threshold}s. Resolving token ID {token_id}.")
#                             updated = update_token_status(conn, cursor, token_id, 'Resolved')
#                             if updated:
#                                 # Retrieve assigned worker ID from token
#                                 cursor.execute("SELECT token_assigned FROM tokens WHERE token_id=?", (token_id,))
#                                 row = cursor.fetchone()
#                                 if row and row['token_assigned']:
#                                     assigned_worker_id = row['token_assigned']
#                                     # Retrieve worker email
#                                     cursor.execute("SELECT worker_email FROM workers WHERE worker_id=?", (assigned_worker_id,))
#                                     email_row = cursor.fetchone()
#                                     if email_row:
#                                         worker_email = email_row['worker_email']
#                                         if should_send_notification(item, rtsp_url):
#                                             notification_queue.put((
#                                                 worker_email,
#                                                 item,
#                                                 0.0,  # Confidence not relevant for resolution
#                                                 current_time.strftime("%Y-%m-%d %H:%M:%S"),
#                                                 rtsp_url,
#                                                 'Resolved'
#                                             ))
#                                 else:
#                                     logging.warning(f"[{rtsp_url}] Token ID {token_id} has no assigned worker.")
#                                 # Release the worker
#                                 released = release_worker(conn, cursor, assigned_worker_id)
#                                 if released:
#                                     logging.debug(f"[{rtsp_url}] Worker ID {assigned_worker_id} released successfully.")
#                                 else:
#                                     logging.warning(f"[{rtsp_url}] Failed to release worker ID {assigned_worker_id}.")
#                                 # Reset detection and token state
#                                 assigned_token[item] = None
#                                 detection_state[item] = False
#                                 detection_start_time[item] = None
#                                 last_detection_time[item] = None
#                                 detection_stop_time[item] = None
#                             else:
#                                 logging.warning(f"[{rtsp_url}] Failed to update token ID {token_id} to 'Resolved'.")

#             # Put the annotated frame into the queue for display
#             if not frame_queue.full():
#                 frame_queue.put(annotated_frame)

#         cap.release()
#         if reconnect_attempts > max_reconnect_attempts:
#             logging.error(f"[{rtsp_url}] Exiting thread after exceeding max reconnect attempts.")
#             break

#     # Close DB connection
#     conn.close()
#     logging.debug(f"[{rtsp_url}] Database connection closed.")

# def run_flask():
#     """
#     Runs the Flask server on a separate thread.
#     """
#     app.run(host='0.0.0.0', port=5001, debug=False)

# ###############################################################################
# #                     Spawning Threads & Main Display Loop                    #
# ###############################################################################

# # Start Flask thread
# flask_thread = threading.Thread(target=run_flask, daemon=True)
# flask_thread.start()
# logging.info("Flask server started on port 5001.")

# # Create a notification queue for asynchronous email alerts
# notification_queue = Queue()
# notification_thread = threading.Thread(target=notification_worker, args=(notification_queue,), daemon=True)
# notification_thread.start()
# logging.info("Notification worker thread started.")

# # List of cameras/RTSP URLs for processing
# camera_urls = [
#     "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/102?rtsp_transport=tcp",
#     "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/202?rtsp_transport=tcp",
#     "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/302?rtsp_transport=tcp",
#     "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/402?rtsp_transport=tcp"
#     # Add more cameras as needed
# ]

# frame_queues = Queue(maxsize=5)
# detection_threads = {}

# # Spawn a detection thread per camera
# for url in camera_urls:
#     frame_queues[url] = Queue(maxsize=2)  # Keep small queue size to reduce lag
#     t = threading.Thread(
#         target=process_rtsp_stream,
#         args=(url, model, frame_queues[url], notification_queue),
#         daemon=True
#     )
#     detection_threads[url] = t
#     t.start()
#     logging.info(f"Detection thread started for camera: {url}")

# def cleanup():
#     """
#     Graceful shutdown: stops threads, cleans up queues.
#     """
#     logging.info("Initiating cleanup process.")
#     # Signal the notification worker to stop
#     notification_queue.put("STOP")
#     notification_queue.join()
#     logging.info("Notification worker signaled to stop.")

#     # Join detection threads
#     for url, t in detection_threads.items():
#         t.join(timeout=5)
#         logging.debug(f"Detection thread for {url} joined.")

#     # Join notification thread
#     notification_thread.join(timeout=5)
#     logging.debug("Notification worker thread joined.")
#     cv2.destroyAllWindows()
#     logging.info("Cleanup complete.")

# try:
#     while True:
#         # Display frames for each camera
#         for url in camera_urls:
#             if not frame_queues[url].empty():
#                 frame = frame_queues[url].get()
#                 cv2.imshow(f"YOLOv5 - {url}", frame)

#         # Quit on 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             logging.info("Exit command received. Shutting down.")
#             break
# except KeyboardInterrupt:
#     logging.info("Interrupted by user. Shutting down.")
# finally:
#     cleanup()


# connection_dvr.py

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
from flask import Flask, Response, request, jsonify
from flask_cors import CORS

###############################################################################
#                             Flask & Configuration                           #
###############################################################################

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

# Database path
DB_PATH = os.getenv('DB_PATH', 'database/smart_building.db')

# Set up logging with enhanced details
logging.basicConfig(
    filename='connection_dvr.log',
    level=logging.DEBUG,  # Changed to DEBUG for more verbose logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

###############################################################################
#                          Utility / DB Connection                            #
###############################################################################

def connect_db():
    """
    Creates a new database connection (with row dict factory).
    Each thread should call this to get its own connection.
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable dictionary-like row access
        logging.debug("Database connection established.")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to database: {e}")
        raise e

def init_db():
    """
    Initialize or connect to the DB and return (conn, cursor).
    """
    try:
        conn = connect_db()
        cursor = conn.cursor()
        logging.debug("Database cursor initialized.")
        return conn, cursor
    except sqlite3.Error as e:
        logging.error(f"Failed to initialize DB: {e}")
        raise e

###############################################################################
#                              Flask Endpoints                                #
###############################################################################

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    """
    Returns an MJPEG stream for a specific camera.
    (Raw from the camera, no YOLO overlay.)
    """
    # Define camera streams (you can fetch from DB if preferred)
    camera_streams = {
        1: "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/202?rtsp_transport=tcp",
        2: "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/302?rtsp_transport=tcp",
        3: "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/402?rtsp_transport=tcp",
    }

    rtsp_url = camera_streams.get(camera_id)
    if not rtsp_url:
        logging.warning(f"Camera ID {camera_id} not found.")
        return f"Camera ID {camera_id} not found.", 404

    def generate():
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logging.error(f"Cannot open RTSP for camera {camera_id}.")
            return

        while True:
            success, frame = cap.read()
            if not success:
                logging.warning(f"Camera {camera_id}: No frame received.")
                break

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logging.warning(f"Camera {camera_id}: Could not encode frame.")
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() +
                   b'\r\n')
        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/token_counts', methods=['GET'])
def get_token_counts():
    """
    Return counts of 'Pending' vs 'Resolved' tokens.
    """
    try:
        conn, cursor = init_db()
        cursor.execute("SELECT COUNT(*) FROM tokens WHERE token_status='Pending'")
        pending = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM tokens WHERE token_status='Resolved'")
        resolved = cursor.fetchone()[0]

        conn.close()
        logging.debug(f"Token counts fetched: Pending={pending}, Resolved={resolved}")
        return {"pending": pending, "resolved": resolved}, 200
    except Exception as e:
        logging.error(f"Error in /token_counts: {e}", exc_info=True)
        return {"error": str(e)}, 500

###############################################################################
#                           Camera & Worker Endpoints                         #
###############################################################################

@app.route('/cameras', methods=['GET'])
def get_cameras():
    """
    Returns a list of all cameras in the DB table 'cameras'.
    """
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cameras")
        rows = cursor.fetchall()
        cameras = [dict(row) for row in rows]
        logging.debug(f"Fetched {len(cameras)} cameras from the database.")
        return jsonify(cameras), 200
    except Exception as e:
        logging.error(f"Error fetching cameras: {e}", exc_info=True)
        return {"error": str(e)}, 500
    finally:
        conn.close()

@app.route('/cameras', methods=['POST'])
def save_cameras():
    """
    Saves (replaces) the entire cameras table with the provided list of cameras.
    """
    try:
        cameras = request.json
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cameras")  # Clear existing data
        logging.debug("Existing cameras deleted from the database.")

        for c in cameras:
            cursor.execute("""
                INSERT INTO cameras (camera_name, x_coordinate, y_coordinate, angle, room_no, floor)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                c.get('camera_name', f"Camera {c.get('camera_id', '')}"),
                c['x_coordinate'],
                c['y_coordinate'],
                c['angle'],
                c.get('room_no', 'Unknown'),
                c.get('floor', 0)
            ))
        conn.commit()
        logging.info(f"Saved {len(cameras)} cameras to the database.")
        return {"message": "Cameras saved successfully"}, 200
    except Exception as e:
        logging.error(f"Error saving cameras: {e}", exc_info=True)
        return {"error": str(e)}, 500
    finally:
        conn.close()

@app.route('/workers', methods=['GET'])
def get_workers():
    """
    Fetch all workers from the database.
    """
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workers")
        rows = cursor.fetchall()
        workers = [dict(row) for row in rows]
        logging.debug(f"Fetched {len(workers)} workers from the database.")
        return jsonify(workers), 200
    except Exception as e:
        logging.error(f"Error fetching workers: {e}", exc_info=True)
        return {"error": str(e)}, 500
    finally:
        conn.close()

@app.route('/workers', methods=['POST'])
def add_worker():
    """
    Add a new worker to the database.
    """
    try:
        data = request.json
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO workers (worker_name, worker_number, worker_email)
            VALUES (?, ?, ?)
        """, (data['name'], data['number'], data['email']))
        conn.commit()
        worker_id = cursor.lastrowid
        logging.info(f"Added new worker: ID={worker_id}, Email={data['email']}.")
        return {"message": "Worker added successfully.", "worker_id": worker_id}, 201
    except sqlite3.IntegrityError as ie:
        logging.error(f"Integrity Error adding worker: {ie}", exc_info=True)
        return {"error": "Worker with this email already exists."}, 400
    except Exception as e:
        logging.error(f"Error adding worker: {e}", exc_info=True)
        return {"error": str(e)}, 500
    finally:
        conn.close()

@app.route('/workers/<int:worker_id>', methods=['PUT'])
def update_worker(worker_id):
    """
    Update an existing worker's status or details.
    """
    try:
        data = request.json
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE workers
            SET worker_name=?, worker_number=?, worker_email=?, status=?
            WHERE worker_id=?
        """, (data['name'], data['number'], data['email'], data['status'], worker_id))
        conn.commit()
        if cursor.rowcount == 0:
            logging.warning(f"Worker ID {worker_id} not found for update.")
            return {"error": "Worker not found."}, 404
        logging.info(f"Worker ID {worker_id} updated successfully.")
        return {"message": "Worker updated successfully."}, 200
    except sqlite3.IntegrityError as ie:
        logging.error(f"Integrity Error updating worker: {ie}", exc_info=True)
        return {"error": "Worker with this email already exists."}, 400
    except Exception as e:
        logging.error(f"Error updating worker: {e}", exc_info=True)
        return {"error": str(e)}, 500
    finally:
        conn.close()

@app.route('/workers/<int:worker_id>', methods=['DELETE'])
def delete_worker(worker_id):
    """
    Delete a worker from the database.
    """
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM workers WHERE worker_id=?", (worker_id,))
        conn.commit()
        if cursor.rowcount == 0:
            logging.warning(f"Worker ID {worker_id} not found for deletion.")
            return {"error": "Worker not found."}, 404
        logging.info(f"Worker ID {worker_id} deleted successfully.")
        return {"message": "Worker deleted successfully."}, 200
    except Exception as e:
        logging.error(f"Error deleting worker: {e}", exc_info=True)
        return {"error": str(e)}, 500
    finally:
        conn.close()

###############################################################################
#                           Token / Worker Logic                              #
###############################################################################

def assign_worker(conn, cursor):
    """
    Assign the least-recently-assigned free worker, mark them as 'occupied'.
    """
    try:
        cursor.execute("""
            SELECT worker_id, worker_email
            FROM workers
            WHERE status='free'
            ORDER BY last_assigned ASC NULLS FIRST
            LIMIT 1;
        """)
        row = cursor.fetchone()
        if row:
            worker_id, worker_email = row['worker_id'], row['worker_email']
            cursor.execute("""
                UPDATE workers
                SET status='occupied', last_assigned=CURRENT_TIMESTAMP
                WHERE worker_id=?;
            """, (worker_id,))
            conn.commit()
            logging.info(f"Assigned worker {worker_id} ({worker_email}).")
            return worker_id, worker_email
        else:
            logging.warning("No free workers available for assignment.")
            return None, None
    except Exception as e:
        logging.error(f"Error in assign_worker: {e}", exc_info=True)
        return None, None

def release_worker(conn, cursor, worker_id):
    """
    Release the worker (mark as 'free').
    """
    try:
        cursor.execute("""
            UPDATE workers
            SET status='free'
            WHERE worker_id=?;
        """, (worker_id,))
        updated = cursor.rowcount
        conn.commit()
        if updated:
            logging.info(f"Released worker {worker_id} and marked as 'free'.")
            return True
        else:
            logging.warning(f"Attempted to release worker {worker_id}, but no rows were updated.")
            return False
    except Exception as e:
        logging.error(f"Error in release_worker: {e}", exc_info=True)
        return False

###############################################################################
#                     Database Logging & Notification Logic                   #
###############################################################################

def log_detection(conn, cursor, item, confidence, timestamp, location, worker_id=None, reason='Detected'):
    """
    Log a detection event into the 'tokens' table.
    """
    try:
        cursor.execute("""
            INSERT INTO tokens (token_reason, token_location, token_assigned, token_status, token_start, token_end_time, confidence)
            VALUES (?, ?, ?, 'Pending', ?, NULL, ?)
        """, (reason, location, worker_id, timestamp, confidence))
        token_id = cursor.lastrowid
        conn.commit()
        logging.info(f"Logged detection: {item} at {location} with confidence={confidence:.2f}, reason={reason}, token_id={token_id}")
        return token_id
    except Exception as e:
        logging.error(f"Error during logging detection: {e}", exc_info=True)
        return None

def update_token_status(conn, cursor, token_id, status='Resolved'):
    """
    Update a token's status.
    """
    try:
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            UPDATE tokens
            SET token_status=?, token_end_time=?
            WHERE token_id=?;
        """, (status, end_time, token_id))
        rows_updated = cursor.rowcount
        conn.commit()
        if rows_updated:
            logging.info(f"Token {token_id} updated to '{status}' with end_time={end_time}.")
            return True
        else:
            logging.warning(f"Token {token_id} not found for updating.")
            return False
    except Exception as e:
        logging.error(f"Error updating token status: {e}", exc_info=True)
        return False

def send_email_notification(worker_email, item, confidence, timestamp, location, status='Pending'):
    """
    Send an email to the assigned worker about a detection or resolution.
    """
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
            logging.info(f"Email notification sent to {TO_EMAIL} for {item} at {timestamp} with status='{status}'.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}", exc_info=True)

def notification_worker(notification_queue):
    """
    Thread worker to asynchronously process and send email notifications.
    """
    while True:
        notif = notification_queue.get()
        if notif == "STOP":
            logging.info("Notification worker stopping.")
            break
        try:
            worker_email, item, confidence, timestamp, location, status = notif
            send_email_notification(worker_email, item, confidence, timestamp, location, status)
        except Exception as e:
            logging.error(f"Notification worker error: {e}", exc_info=True)
        finally:
            notification_queue.task_done()

###############################################################################
#                          YOLO & RTSP Processing                             #
###############################################################################

# Path to your custom YOLO model
model_path = '/Users/karthiknutulapati/Desktop/smartbuilding/cv+v5/yolov5-master/runs/train/exp12/weights/best.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    # Load the custom YOLO model using torch hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.to(device)
    model.eval()
    logging.info(f"Custom YOLO model loaded successfully from '{model_path}' on {device}.")
except Exception as e:
    logging.error(f"Error loading custom YOLO model: {e}")
    exit(1)

# Items you want to detect
target_items = ["wallet"]  # Extend as needed

# Thresholds for detection logic (in seconds)
detection_duration_threshold = 10  # Time item must be continuously visible to raise a token
detection_gap_threshold = 2        # Time item must be absent to consider detection stopped
resolution_gap_threshold = 5       # Time item must be absent to resolve the token

# Rate-limiting notifications
last_notification_time = {}
notification_cooldown = 60  # seconds

def should_send_notification(item, location):
    """
    Rate-limit notifications on a per-item+location basis.
    """
    global last_notification_time
    key = f"{item}_{location}"
    now = time.time()
    if key in last_notification_time:
        elapsed = now - last_notification_time[key]
        if elapsed < notification_cooldown:
            logging.debug(f"Notification cooldown active for {item} at {location}.")
            return False
    last_notification_time[key] = now
    return True

def process_rtsp_stream(rtsp_url, model, frame_queue, notification_queue):
    """
    Thread function that reads RTSP frames, runs YOLO, and manages detection states.
    Each camera has its own detection dictionaries so tokens are raised independently.
    """
    # Per-camera detection states
    detection_state = {item: False for item in target_items}
    detection_start_time = {item: None for item in target_items}
    last_detection_time = {item: None for item in target_items}
    detection_stop_time = {item: None for item in target_items}
    assigned_token = {item: None for item in target_items}

    reconnect_attempts = 0
    max_reconnect_attempts = 5
    reconnect_delay = 5  # seconds

    # Frame skipping to reduce lag
    frame_count = 0
    process_every_n_frames = 1  # Process every frame for real-time detection

    # Initialize DB connection for this thread
    try:
        conn, cursor = init_db()
    except Exception as e:
        logging.critical(f"Failed to initialize DB for thread handling {rtsp_url}: {e}")
        return

    while reconnect_attempts <= max_reconnect_attempts:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logging.error(f"[{rtsp_url}] Cannot open stream. Retrying in {reconnect_delay}s...")
            reconnect_attempts += 1
            time.sleep(reconnect_delay)
            continue
        else:
            logging.info(f"[{rtsp_url}] Stream opened successfully.")
            reconnect_attempts = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"[{rtsp_url}] No frame received. Attempting reconnect...")
                cap.release()
                reconnect_attempts += 1
                if reconnect_attempts > max_reconnect_attempts:
                    logging.error(f"[{rtsp_url}] Exceeded max reconnect attempts. Exiting thread.")
                    break
                time.sleep(reconnect_delay)
                break

            # Skip frames if configured
            frame_count += 1
            if frame_count % process_every_n_frames != 0:
                continue

            # YOLO inference
            try:
                with torch.cuda.amp.autocast() if device == 'cuda' else contextlib.nullcontext():
                    results = model(frame, size=640)
                annotated_frame = results.render()[0]
            except Exception as e:
                logging.error(f"[{rtsp_url}] Error in YOLO inference: {e}", exc_info=True)
                continue

            current_time = datetime.now()
            detected_items = {}

            # Gather YOLO detections
            for *xyxy, conf, cls_idx in results.xyxy[0]:
                label = model.names[int(cls_idx)]
                confidence = float(conf)
                if label in target_items and confidence >= 0.5:
                    detected_items[label] = confidence

            # Per-item detection logic
            for item in target_items:
                if item in detected_items:
                    # If newly detected
                    if not detection_state[item]:
                        detection_state[item] = True
                        detection_start_time[item] = current_time
                        detection_stop_time[item] = None
                        logging.info(f"[{rtsp_url}] Detection started for '{item}' at {current_time}.")
                    last_detection_time[item] = current_time
                else:
                    # If item was being detected, check if it disappeared
                    if detection_state[item]:
                        if last_detection_time[item]:
                            elapsed_gone = (current_time - last_detection_time[item]).total_seconds()
                            if elapsed_gone > detection_gap_threshold:
                                detection_state[item] = False
                                detection_start_time[item] = None
                                detection_stop_time[item] = current_time
                                logging.info(f"[{rtsp_url}] Detection stopped for '{item}' at {current_time}.")
                        else:
                            # No last_detection_time, forcibly reset
                            detection_state[item] = False
                            detection_start_time[item] = None
                            detection_stop_time[item] = current_time

                # Check if we should assign a token (continuous detection)
                if detection_state[item]:
                    total_time_detected = (current_time - detection_start_time[item]).total_seconds()
                    if total_time_detected >= detection_duration_threshold and assigned_token[item] is None:
                        logging.info(f"[{rtsp_url}] '{item}' detected continuously for {detection_duration_threshold}s. Raising token.")
                        worker_id, worker_email = assign_worker(conn, cursor)
                        if worker_id and worker_email:
                            ts_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                            # **Fix:** Check if 'item' is in detected_items before accessing
                            if item in detected_items:
                                confidence = detected_items[item]
                                token_id = log_detection(
                                    conn, cursor,
                                    item=item,
                                    confidence=confidence,
                                    timestamp=ts_str,
                                    location=rtsp_url,  # Use rtsp_url to identify camera/location
                                    worker_id=worker_id
                                )
                                if token_id:
                                    assigned_token[item] = token_id
                                    logging.debug(f"[{rtsp_url}] Assigned token ID {token_id} to worker ID {worker_id}.")
                                    if should_send_notification(item, rtsp_url):
                                        notification_queue.put((
                                            worker_email,
                                            item,
                                            confidence,
                                            ts_str,
                                            rtsp_url,
                                            'Pending'
                                        ))
                                else:
                                    logging.warning(f"[{rtsp_url}] Failed to log detection for '{item}'.")
                            else:
                                logging.warning(f"[{rtsp_url}] '{item}' not detected during token assignment.")
                        else:
                            logging.warning(f"[{rtsp_url}] No free workers available to assign for '{item}' detection.")

                # Check for token resolution
                if assigned_token[item] is not None and not detection_state[item]:
                    # If detection stopped, see if we pass resolution threshold
                    if detection_stop_time[item]:
                        absent_time = (current_time - detection_stop_time[item]).total_seconds()
                        if absent_time >= resolution_gap_threshold:
                            token_id = assigned_token[item]
                            logging.info(f"[{rtsp_url}] '{item}' absent for {resolution_gap_threshold}s. Resolving token ID {token_id}.")
                            updated = update_token_status(conn, cursor, token_id, 'Resolved')
                            if updated:
                                # Retrieve assigned worker ID from token
                                cursor.execute("SELECT token_assigned FROM tokens WHERE token_id=?", (token_id,))
                                row = cursor.fetchone()
                                if row and row['token_assigned']:
                                    assigned_worker_id = row['token_assigned']
                                    # Retrieve worker email
                                    cursor.execute("SELECT worker_email FROM workers WHERE worker_id=?", (assigned_worker_id,))
                                    email_row = cursor.fetchone()
                                    if email_row:
                                        worker_email = email_row['worker_email']
                                        if should_send_notification(item, rtsp_url):
                                            notification_queue.put((
                                                worker_email,
                                                item,
                                                0.0,  # Confidence not relevant for resolution
                                                current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                                rtsp_url,
                                                'Resolved'
                                            ))
                                    else:
                                        logging.warning(f"[{rtsp_url}] Worker email not found for worker ID {assigned_worker_id}.")
                                    # Release the worker
                                    released = release_worker(conn, cursor, assigned_worker_id)
                                    if released:
                                        logging.debug(f"[{rtsp_url}] Worker ID {assigned_worker_id} released successfully.")
                                    else:
                                        logging.warning(f"[{rtsp_url}] Failed to release worker ID {assigned_worker_id}.")
                                else:
                                    logging.warning(f"[{rtsp_url}] Token ID {token_id} has no assigned worker.")
                                # Reset detection and token state
                                assigned_token[item] = None
                                detection_state[item] = False
                                detection_start_time[item] = None
                                last_detection_time[item] = None
                                detection_stop_time[item] = None
                            else:
                                logging.warning(f"[{rtsp_url}] Failed to update token ID {token_id} to 'Resolved'.")

            # Put the annotated frame into the queue for display
            if not frame_queue.full():
                frame_queue.put(annotated_frame)

        cap.release()
        if reconnect_attempts > max_reconnect_attempts:
            logging.error(f"[{rtsp_url}] Exiting thread after exceeding max reconnect attempts.")
            break

    # Close DB connection
    conn.close()
    logging.debug(f"[{rtsp_url}] Database connection closed.")

###############################################################################
#                     Spawning Threads & Main Display Loop                    #
###############################################################################

def run_flask():
    """
    Runs the Flask server on a separate thread.
    """
    app.run(host='0.0.0.0', port=5001, debug=False)

# Start Flask thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
logging.info("Flask server started on port 5001.")

# Create a notification queue for asynchronous email alerts
notification_queue = Queue()
notification_thread = threading.Thread(target=notification_worker, args=(notification_queue,), daemon=True)
notification_thread.start()
logging.info("Notification worker thread started.")

# List of cameras/RTSP URLs for processing
camera_urls = [
    "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/202?rtsp_transport=tcp",
    "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/302?rtsp_transport=tcp",
    "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/402?rtsp_transport=tcp",
]

frame_queues = {}
detection_threads = {}

# Spawn a detection thread per camera
for url in camera_urls:
    frame_queues[url] = Queue(maxsize=2)  # Keep small queue size to reduce lag
    t = threading.Thread(
        target=process_rtsp_stream,
        args=(url, model, frame_queues[url], notification_queue),
        daemon=True
    )
    detection_threads[url] = t
    t.start()
    logging.info(f"Detection thread started for camera: {url}")

def cleanup():
    """
    Graceful shutdown: stops threads, cleans up queues.
    """
    logging.info("Initiating cleanup process.")
    # Signal the notification worker to stop
    notification_queue.put("STOP")
    notification_queue.join()
    logging.info("Notification worker signaled to stop.")

    # Join detection threads
    for url, t in detection_threads.items():
        t.join(timeout=5)
        logging.debug(f"Detection thread for {url} joined.")

    # Join notification thread
    notification_thread.join(timeout=5)
    logging.debug("Notification worker thread joined.")
    cv2.destroyAllWindows()
    logging.info("Cleanup complete.")

try:
    while True:
        # Display frames for each camera
        for url in camera_urls:
            if not frame_queues[url].empty():
                frame = frame_queues[url].get()
                cv2.imshow(f"YOLOv5 - {url}", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exit command received. Shutting down.")
            break
except KeyboardInterrupt:
    logging.info("Interrupted by user. Shutting down.")
finally:
    cleanup()