from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
# Biến toàn cục
id_muc_tieu = None
vi_tri_click = None
track_history = defaultdict(lambda: deque(maxlen=30))
# Khởi tạo model và tracker
model = YOLO("yolov8m-pose.pt")
tracker = DeepSort(
    max_age=150,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.3,
    nn_budget=100,
    nms_max_overlap=0.7,
    embedder="mobilenet"
)
def xu_ly_click_chuot(event, x, y, flags, param):
    """Xử lý sự kiện click chuột để chọn target"""
    global id_muc_tieu, vi_tri_click
    if event == cv2.EVENT_LBUTTONDOWN:
        vi_tri_click = (x, y)
        print(f"Đã click tại: ({x}, {y})")
# Mở video và setup
cap = cv2.VideoCapture("D:/2025/DoAn_TTNT/person.mp4")
cv2.namedWindow("YOLOv8 + DeepSORT Tracking")
cv2.setMouseCallback("YOLOv8 + DeepSORT Tracking", xu_ly_click_chuot)
print("Click chuột vào người muốn theo dõi!")
# Biến điều khiển
frame_count = 0
target_lost_frames = 0
MAX_LOST_FRAMES = 50
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    scale = 0.3
    frame = cv2.resize(frame, None, fx=scale, fy=scale)
    # Adaptive confidence threshold
    conf_high = 0.55
    conf_low = 0.4
    use_low_conf = (id_muc_tieu is not None and target_lost_frames > 5)
    conf_threshold = conf_low if use_low_conf else conf_high
    # YOLO Detection
    results = model(frame, classes=[0], conf=conf_threshold, iou=0.5)
    # Chuẩn bị detections cho DeepSORT
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            if conf > conf_threshold:
                x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
                w, h = x2_int - x1_int, y2_int - y1_int
                if w > 20 and h > 40:
                    person_region = frame[y1_int:y2_int, x1_int:x2_int]
                    if person_region.size > 0:
                        detections.append(([x1, y1, w, h], conf, person_region))
    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)
    # Tính toán thông số frame
    h_frame, w_frame, _ = frame.shape
    center_frame = w_frame // 2
    frame_area = w_frame * h_frame
    far_threshold = 0.05 * frame_area
    near_threshold = 0.3 * frame_area
    # Xử lý click chuột - chọn target
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
        if closest_track is not None and min_distance < 150:
            id_muc_tieu = closest_track
            target_lost_frames = 0
            print(f"Đã chọn mục tiêu: ID {id_muc_tieu}")
        else:
            print("Không tìm thấy người gần vị trí click")
        vi_tri_click = None
    # Xử lý từng track
    target_found = False
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)
        track_history[track_id].append((cx, cy))
        # Tự động chọn target đầu tiên nếu chưa có
        if id_muc_tieu is None:
            id_muc_tieu = track_id
            target_lost_frames = 0
            print(f"Mục tiêu tự động: {id_muc_tieu}")
        # Xử lý target
        if track_id == id_muc_tieu:
            target_found = True
            target_lost_frames = 0
            color = (0, 255, 0)
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
            cv2.putText(frame, f"TARGET:{track_id}", (int(l), int(t) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            # Hệ thống điều hướng
            area = (r - l) * (b - t)
            if area < far_threshold:
                direction = "Tien len"
            elif area > near_threshold:
                direction = "Lui lai"
            else:
                if cx < center_frame - 80:
                    direction = "Re phai"
                elif cx > center_frame + 80:
                    direction = "Re trai"
                else:
                    direction = "Di thang"
            # Hiển thị thông tin
            info_text = [f"TARGET: {track_id}", f"LENH: {direction}", f"Vi tri: ({cx}, {cy})"]
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            # Vẽ tracks khác
            color = (0, 0, 255)
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 1)
            cv2.putText(frame, f"ID:{track_id}", (int(l), int(t) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Xử lý mất target
    if not target_found and id_muc_tieu is not None:
        target_lost_frames += 1
        warning_text = f"MAT TARGET! ({target_lost_frames}/{MAX_LOST_FRAMES})"
        cv2.putText(frame, warning_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)    
        if target_lost_frames > MAX_LOST_FRAMES:
            print(f"Đã mất target ID {id_muc_tieu} quá lâu. Chọn lại mục tiêu!")
            id_muc_tieu = None
            target_lost_frames = 0
    # Vẽ UI elements
    cv2.line(frame, (center_frame, 0), (center_frame, h_frame), (255, 255, 0), 2)
    if id_muc_tieu is None:
        huong_dan = "Click chuot de chon nguoi theo doi"
        color_guide = (255, 255, 255)
    else:
        status = "OK" if target_found else f"MAT ({target_lost_frames})"
        huong_dan = f"ID: {id_muc_tieu} - Status: {status} - 'c': chon lai"
        color_guide = (0, 255, 0) if target_found else (0, 0, 255)
    
    cv2.putText(frame, huong_dan, (10, h_frame - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_guide, 2)
    cv2.putText(frame, f"Detected: {len(tracks)} persons", (w_frame - 200, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # Hiển thị và xử lý phím
    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        id_muc_tieu = None
        target_lost_frames = 0
        print("Đã xóa lựa chọn, click để chọn mục tiêu mới")
# Cleanup
cap.release()
cv2.destroyAllWindows()
