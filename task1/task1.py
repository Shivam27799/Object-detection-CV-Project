# task1/task1.py

from ultralytics import YOLO
import cv2
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def run_task1(video_path="sample_video.mp4", model_path="yolov8n.pt", output_dir="output/task1"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "json"), exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return

    detection_summary = defaultdict(int)
    frame_class_diversity = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.predict(source=frame, conf=0.25, verbose=False)

        frame_data = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                detection_summary[label] += 1
                frame_class_diversity.setdefault(frame_count, set()).add(label)

                # Append to JSON list
                frame_data.append({
                    "class": label,
                    "bbox": xyxy,
                    "confidence": round(conf, 4)
                })

                # Draw box
                cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (xyxy[0], xyxy[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save every 30th frame with output + JSON
        if frame_count % 30 == 0:
            frame_name = f"frame_{frame_count}"
            cv2.imwrite(os.path.join(output_dir, "frames", f"{frame_name}.jpg"), frame)

            # Save JSON
            with open(os.path.join(output_dir, "json", f"{frame_name}.json"), "w") as f:
                json.dump(frame_data, f, indent=4)
            print(f"✅ Saved frame and JSON for: {frame_name}")

    cap.release()

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(detection_summary, f, indent=4)

    # Max diverse frame
    max_diverse_frame = max(frame_class_diversity.items(), key=lambda x: len(x[1]))
    print(f"\n🧠 Frame {max_diverse_frame[0]} had most diverse objects: {len(max_diverse_frame[1])}")
    print("   ➤ Classes:", ', '.join(max_diverse_frame[1]))

    # Bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(detection_summary.keys(), detection_summary.values(), color='skyblue')
    plt.xlabel("Object Class")
    plt.ylabel("Frequency")
    plt.title("Object Frequency Summary (Task 1)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "object_frequency.png")
    plt.savefig(plot_path)
    print(f"📊 Frequency chart saved to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    run_task1()