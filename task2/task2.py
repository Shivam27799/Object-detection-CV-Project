# task2/task2.py

from ultralytics import YOLO
import cv2
import os
import json
from datetime import datetime

def run_task2(video_source=0, model_path="yolov8n.pt", alert_class="person", output_dir="output/task2"):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "alerts"), exist_ok=True)

    # Open video stream
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Failed to open stream: {video_source}")
        return

    print(f"📡 Real-time detection started on: {video_source}")
    alert_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream ended or failed.")
            break

        # Predict
        results = model.predict(source=frame, conf=0.3, verbose=False)

        alert_triggered = False
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                # Trigger alert if match
                if label == alert_class:
                    if not alert_triggered:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        alert_filename = os.path.join(output_dir, "alerts", f"alert_{timestamp}.jpg")
                        cv2.imwrite(alert_filename, frame)
                        alert_log.append({
                            "timestamp": timestamp,
                            "alert_class": alert_class,
                            "image": alert_filename
                        })
                        print(f"🚨 ALERT: '{alert_class}' detected! Saved to {alert_filename}")
                        alert_triggered = True

                # Draw box
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show stream
        cv2.imshow("Task 2 - Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save alert log as JSON
    alert_json_path = os.path.join(output_dir, "alerts.json")
    with open(alert_json_path, "w") as f:
        json.dump(alert_log, f, indent=4)

    print(f"\n📁 Alert log saved to: {alert_json_path}")
    print("🛑 Detection stopped.")

if __name__ == "__main__":
    run_task2()