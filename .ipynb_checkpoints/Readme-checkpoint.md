# 🚀 YOLOv8 Computer Vision Project – Object Detection, Summarization & Real-Time Alerts

This project contains two modular Computer Vision tasks built using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) model:

- **Task 1**: Run object detection on a video, generate per-frame JSON, save annotated frames, compute summary statistics, and visualize object frequency.
- **Task 2**: Run real-time stream detection (webcam or video) and generate alerts when specific classes (like `"person"` or `"car"`) are detected.

---

## 📁 Project Structure

cv_assignment/
├── task1/
│ └── task1.py # Summarization and JSON logging
├── task2/
│ └── task2.py # Real-time alerting system
├── yolov8n.pt # YOLOv8n model (not pushed to GitHub)
├── sample_video.mp4 # Sample test video (optional)
├── run.py # Controller to toggle between tasks
├── requirements.txt # All dependencies
├── Dockerfile # Docker setup
└── output/
└── task1/ or task2/ # Output visualizations, JSON logs