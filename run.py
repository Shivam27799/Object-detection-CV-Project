# run.py

import argparse
from task1.task1 import run_task1
from task2.task2 import run_task2

def main():
    parser = argparse.ArgumentParser(description="Computer Vision Task Runner")
    parser.add_argument("--task", type=int, choices=[1, 2], required=True, help="Choose task: 1 or 2")
    parser.add_argument("--video", type=str, default="sample_video.mp4", help="Path to video file or camera index")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 model file")
    parser.add_argument("--alert", type=str, default="person", help="Object class to alert on (Task 2 only)")

    args = parser.parse_args()

    if args.task == 1:
        run_task1(video_path=args.video, model_path=args.model)
    elif args.task == 2:
        # Try to convert webcam index if it's a digit
        source = int(args.video) if args.video.isdigit() else args.video
        run_task2(video_source=source, model_path=args.model, alert_class=args.alert)

if __name__ == "__main__":
    main()