from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
model = YOLO("yolov8n.pt")  
tracker = DeepSort(max_age=10)
cap = cv2.VideoCapture(0)  
target_id = None
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, classes=[0]) 
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))
    tracks = tracker.update_tracks(detections, frame=frame)
    h, w, _ = frame.shape
    center_frame = w // 2
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)
        if target_id is None:
            target_id = track_id
            print(f" Mục tiêu: {target_id}")
        color = (0, 255, 0) if track_id == target_id else (0, 0, 255)
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
        cv2.putText(frame, f"ID:{track_id}", (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if track_id == target_id:
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            if cx < center_frame - 80:
                direction = " Rẽ phải"
            elif cx > center_frame + 80:
                direction = " Rẽ trái"
            else:
                direction = "  Đi thẳng"

            print(f"ID {track_id}: {direction}")

    cv2.line(frame, (center_frame, 0), (center_frame, h), (255, 255, 0), 2)
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
