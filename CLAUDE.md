# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a smart building monitoring system that combines YOLOv5 computer vision for object detection with a React-based dashboard for visualization and management. The system detects objects like wallets, spills, bottles, cups, and people through RTSP camera feeds and manages worker assignments through a token-based notification system.

## Architecture

### Core Components

**YOLOv5 Backend (`yolov5-master/`)**
- Main detection engine using custom-trained YOLOv5 model
- `connection_dvr.py`: Primary backend service handling RTSP streams, object detection, Flask API, and worker management
- Custom model located at: `runs/train/exp12/weights/best.pt`
- Detects 5 classes: bottle, cup, person, spill, wallet (defined in `new_dataset/data.yaml`)

**React Dashboard (`smart-building-dashboard/`)**
- Frontend dashboard for monitoring and management
- Material-UI based interface with live streaming, worker management, and token tracking
- Main components: LiveStream, PowerConsumption, WeatherUpdates, Tokens, RoomLayout, ManageWorkers

**Database**
- SQLite database at `yolov5-master/database/smart_building.db`
- Manages cameras, workers, and detection tokens

### Key Architecture Patterns

**Detection Logic**
- Continuous detection threshold: 10 seconds (item must be visible continuously)
- Detection gap threshold: 2 seconds (absence time to consider detection stopped)
- Resolution gap threshold: 5 seconds (absence time to resolve token)
- Worker assignment system with least-recently-assigned algorithm

**Multi-threading**
- Separate threads for each RTSP camera stream
- Flask API server thread
- Email notification worker thread
- Frame processing with configurable frame skipping (every 3rd frame)

## Development Commands

### React Dashboard
```bash
cd smart-building-dashboard
npm install          # Install dependencies
npm start            # Start development server (http://localhost:3000)
npm run build        # Build for production
npm test             # Run tests
```

### YOLOv5 Backend
```bash
cd yolov5-master
pip install -r requirements.txt  # Install Python dependencies

# Run the main detection system
python connection_dvr.py

# Train new model
python train.py --data new_dataset/data.yaml --weights yolov5s.pt --epochs 100

# Run detection on images/videos
python detect.py --source path/to/images --weights runs/train/exp12/weights/best.pt

# Validate model performance
python val.py --data new_dataset/data.yaml --weights runs/train/exp12/weights/best.pt
```

### Database Management
```bash
cd yolov5-master
python create_tables.py  # Initialize database tables
```

## Configuration

### Environment Variables (YOLOv5)
Create `.env` file in `yolov5-master/` with:
- `DB_PATH`: Database file path (default: database/smart_building.db)
- `SMTP_SERVER`, `SMTP_USERNAME`, `SMTP_PASSWORD`: Email configuration
- `FROM_EMAIL`: Sender email address

### RTSP Camera Configuration
Camera streams are hardcoded in `connection_dvr.py` around line 861-866:
```python
camera_streams = {
    1: "rtsp://admin:admin123@192.168.29.194:554/Streaming/Channels/102?rtsp_transport=tcp",
    2-4: # Additional camera streams
}
```

### Model Configuration
- Custom model path: `/Users/karthiknutulapati/Desktop/smartbuilding/cv+v5/yolov5-master/runs/train/exp12/weights/best.pt`
- Target detection items: `["wallet"]` (configurable in connection_dvr.py line 1245)
- Inference size: 416px (optimized for performance)

## API Endpoints

**Flask Backend (Port 5001)**
- `GET /video_feed/<camera_id>`: MJPEG stream for camera
- `GET /token_counts`: Returns pending/resolved token counts
- `GET /cameras`, `POST /cameras`: Camera management
- `GET /workers`, `POST /workers`, `PUT /workers/<id>`, `DELETE /workers/<id>`: Worker management

## Development Notes

### Performance Optimizations
- Frame skipping: Process every 3rd frame to reduce CPU load
- Reduced inference size: 416px instead of 640px
- Logging level set to INFO instead of DEBUG
- Small frame queues (maxsize=2) to reduce lag

### Database Schema
- `cameras`: Camera positioning and metadata
- `workers`: Worker information and status (free/occupied)
- `tokens`: Detection events with assignment and resolution tracking

### Detection Flow
1. RTSP streams processed continuously per camera
2. YOLO inference on frames
3. Object tracking with time-based logic
4. Worker assignment when detection threshold met
5. Email notifications sent asynchronously
6. Token resolution when object absent for threshold period
7. Worker release back to free pool

### Testing
- React components have test files in `smart-building-dashboard/src/`
- YOLOv5 model validation can be run with `val.py`
- Manual testing through dashboard interface at http://localhost:3000