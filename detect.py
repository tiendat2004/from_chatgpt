import cv2
from ultralytics import YOLO
import time
model = YOLO("yolov8n.pt")  
cap = cv2.VideoCapture(0)
time_start = None
time_end = 30
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    h, w, _ = frame.shape
    frame_center = w // 2
    threshold = 60  
    gan = 0.3*(w*h)
    xa=0.05*(w*h)
    command = "Dừng"
    persons = [b for b, l in zip(boxes, labels) if l == 0]
    if len(persons) > 0:
        largest = max(persons, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        x1, y1, x2, y2 = map(int, largest)
        cx = (x1 + x2) // 2
      area = (x2 - x1) * (y2 - y1)
        if cx < frame_center - threshold:
            command = "Rẽ trái"
        elif cx > frame_center + threshold:
            command = "Rẽ phải"
        else:
            command = "Đi thẳng"
        if area > gan:
            command = "LUI"
        elif area < xa:
            command = "Tiến lên"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(frame, (cx, 0), (cx, h), (0, 255, 255), 2)
    else:
        if time_start is None:
            time_start = time.time()
        else:
            set = time.time() - time_start
            if set < time_end:
                command = "Tìm kiếm"
            else:
                command = "Dừng"
    cv2.putText(frame, f"CMD: {command}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.line(frame, (frame_center, 0), (frame_center, h), (0, 0, 255), 2)
    cv2.imshow("Person Tracking", frame)
    print(command)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
