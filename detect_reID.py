from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
model = YOLO("yolov8n.pt")  
tracker = DeepSort(max_age=20)
cap = cv2.VideoCapture(0)  
id_muc_tieu = None
vi_tri_click = None
def xu_ly_click_chuot(event, x, y, flags, param):
    global id_muc_tieu, vi_tri_click
    if event == cv2.EVENT_LBUTTONDOWN:
        vi_tri_click = (x, y)
        print(f"Đã click tại: ({x}, {y})")
cv2.namedWindow("YOLOv8 + DeepSORT Tracking")
cv2.setMouseCallback("YOLOv8 + DeepSORT Tracking", xu_ly_click_chuot)
print("Click chuột vào người muốn theo dõi!")
print("ID sẽ được chọn tự động")
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
    frame_area = w * h
    far_threshold = 0.05 * frame_area  # Ngưỡng xa - tiến lên
    near_threshold = 0.3 * frame_area   # Ngưỡng gần - lùi lại
    if vi_tri_click is not None:
        closest_track = None
        min_distance = float('inf')
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cx = int((l + r) / 2)
            cy = int((t + b) / 2)
            distance = ((vi_tri_click[0] - cx) ** 2 + (vi_tri_click[1] - cy) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_track = track_id
        if closest_track is not None and min_distance < 100:
            id_muc_tieu = closest_track
            print(f"Đã chọn mục tiêu: ID {id_muc_tieu}")
        else:
            print("Không tìm thấy người gần vị trí click")
        vi_tri_click = None
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)
        if id_muc_tieu is None:
            id_muc_tieu = track_id
            print(f"Mục tiêu: {id_muc_tieu}")
        if track_id == id_muc_tieu:
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
            cv2.putText(frame, f"TARGET:{track_id}", (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            area = (r - l) * (b - t)
            if area < far_threshold:
                direction = "Tien len"
            elif area > near_threshold:
                direction = "LLui lai"
            else:
                if cx < center_frame - 80:
                    direction = "Re phai"
                elif cx > center_frame + 80:
                    direction = "Re trai"
                else:
                    direction = "Di thang"
            print(f"ID {track_id}: {direction}")
            info_text = [
                f"TARGET: {track_id}",
                f"LENH: {direction}",
                f"Vi tri: ({cx}, {cy})"
            ]
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            color = (0, 0, 255)
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 1)
            cv2.putText(frame, f"ID:{track_id}", (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.line(frame, (center_frame, 0), (center_frame, h), (255, 255, 0), 2)
    if id_muc_tieu is None:
        huong_dan = "Click chuot de chon nguoi theo doi"
    else:
        huong_dan = f"Dang theo doi ID: {id_muc_tieu} - Nhan 'c' de chon lai"
    cv2.putText(frame, huong_dan, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        id_muc_tieu = None
        print("Đã xóa lựa chọn, click để chọn mục tiêu mới")
cap.release()
cv2.destroyAllWindows()
