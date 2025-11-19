# yolotrack_cpu_lane_tracker.py
"""
CPU-optimized Vehicle Detection + Speed + Lane Counting
- Tracks cars and trucks in left, center, right lanes
- Live speed estimation (km/h)
- No ID labels
- Normal FPS output video
- Lane separators drawn
"""

import cv2
from ultralytics import YOLO

# -----------------------------
# User Settings
# -----------------------------
VIDEO_PATH = "speed.mp4"                # Input video
OUTPUT_PATH = "output_yolotrack_cpu.mp4"  # Output video
MODEL_PATH = "yolov10n.pt"             # YOLOv10 model
REAL_DISTANCE_M = 5.0                   # Distance between lines
CONFIDENCE_THRESHOLD = 0.35
ALLOWED_CLASSES = {"car", "truck"}
SCALE = 0.5                             # Resize factor for faster detection

# -----------------------------
# Main Function
# -----------------------------
def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)  # runs on CPU by default if GPU not detected

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    ret, frame = cap.read()
    if not ret:
        print("❌ Could not open video")
        return
    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Horizontal lines for speed measurement
    LINE_A_Y = int(height * 0.55)
    LINE_B_Y = int(height * 0.8)

    # Lane boundaries (adjust for your camera view)
    LEFT_LANE_X = 0
    CENTER_LANE_X = int(width * 0.4)
    RIGHT_LANE_X = int(width * 0.75)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # Tracking variables
    total_in = {"left":0, "center":0, "right":0}
    total_out = {"left":0, "center":0, "right":0}
    cross_times = {}
    speeds = {}
    counted = set()

    # Preview window
    cv2.namedWindow("Vehicle Count & Speed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Count & Speed", width//2, height//2)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx +=1
        t_now = frame_idx/fps

        small_frame = cv2.resize(frame, (0,0), fx=SCALE, fy=SCALE)
        results = model.track(small_frame, persist=True, verbose=False)

        boxes_list = []
        if results[0].boxes is not None and len(results[0].boxes)>0:
            boxes_list = list(results[0].boxes)

        for box in boxes_list:
            conf = float(box.conf[0])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name not in ALLOWED_CLASSES:
                continue

            # Scale boxes
            x1, y1, x2, y2 = [int(v/SCALE) for v in box.xyxy[0].tolist()]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            track_id = int(box.id[0]) if box.id is not None else f"noid_{frame_idx}_{cx}"

            # Determine lane
            if cx < CENTER_LANE_X:
                lane = "left"
            elif cx < RIGHT_LANE_X:
                lane = "center"
            else:
                lane = "right"

            # Initialize tracking
            if track_id not in cross_times:
                cross_times[track_id] = {"A": None, "B": None, "lane": lane}
                speeds[track_id] = None

            # Line crossings
            if cy >= LINE_A_Y and cross_times[track_id]["A"] is None:
                cross_times[track_id]["A"] = t_now
            if cy >= LINE_B_Y and cross_times[track_id]["B"] is None:
                cross_times[track_id]["B"] = t_now

            # Speed calculation
            A_time = cross_times[track_id]["A"]
            B_time = cross_times[track_id]["B"]
            if A_time is not None and B_time is None:
                dt = t_now - A_time
                speeds[track_id] = round(REAL_DISTANCE_M/dt*3.6,2) if dt>0.01 else None

            # Count once
            if B_time is not None and track_id not in counted:
                direction = "in" if A_time<B_time else "out"
                if direction=="in":
                    total_in[lane] +=1
                else:
                    total_out[lane] +=1
                counted.add(track_id)

            # Draw bounding box and speed
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            speed_text = f"{speeds[track_id]} km/h" if speeds[track_id] else "N/A"
            label = f"{class_name} {speed_text}"
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        # Draw horizontal lines
        cv2.line(frame,(0,LINE_A_Y),(width,LINE_A_Y),(255,0,0),2)
        cv2.line(frame,(0,LINE_B_Y),(width,LINE_B_Y),(0,0,255),2)

        # Draw vertical lane separators
        cv2.line(frame,(CENTER_LANE_X,0),(CENTER_LANE_X,height),(255,255,0),2)
        cv2.line(frame,(RIGHT_LANE_X,0),(RIGHT_LANE_X,height),(255,255,0),2)

        # Draw lane counts
        cv2.putText(frame,f"Left IN:{total_in['left']} OUT:{total_out['left']}",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(frame,f"Center IN:{total_in['center']} OUT:{total_out['center']}",(30,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        cv2.putText(frame,f"Right IN:{total_in['right']} OUT:{total_out['right']}",(30,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        cv2.imshow("Vehicle Count & Speed", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Finished. Output saved to",OUTPUT_PATH)

if __name__=="__main__":
    main()

