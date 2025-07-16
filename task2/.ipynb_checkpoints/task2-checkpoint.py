# task2/task2.py

from ultralytics import YOLO
import cv2

def run_task2(video_source=0, model_path="yolov8n.pt", alert_class="person"):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Open video stream (0 = webcam, or use video file path)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Failed to open stream: {video_source}")
        return

    print(f"📡 Real-time detection started on: {video_source}")
    alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream ended or failed.")
            break

        # Inference
        results = model.predict(source=frame, conf=0.3, verbose=False)

        # Draw boxes and check for alert_class
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label == alert_class:
                    if not alert_triggered:
                        print(f"🚨 ALERT: '{alert_class}' detected!")
                        alert_triggered = True

                # Optional: Draw box (for visual debug)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show live feed
        cv2.imshow("Task 2 - Live Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Detection stopped.")

if __name__ == "__main__":
    run_task2()