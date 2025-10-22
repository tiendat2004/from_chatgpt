import cv2
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
    h, w, _ = frame.shape
    frame_center = w // 2
    threshold = 60  
    command = "STOP"
    persons = [b for b, l in zip(boxes, labels) if l == 0]
    if len(persons) > 0:
        largest = max(persons, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        x1, y1, x2, y2 = map(int, largest)
        cx = (x1 + x2) // 2
        if cx < frame_center - threshold:
            command = "LEFT"
        elif cx > frame_center + threshold:
            command = "RIGHT"
        else:
            command = "FORWARD"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(frame, (cx, 0), (cx, h), (0, 255, 255), 2)
    else:
        command = "STOP"
    cv2.putText(frame, f"CMD: {command}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.line(frame, (frame_center, 0), (frame_center, h), (0, 0, 255), 2)
    cv2.imshow("Person Tracking", frame)
    print(command)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
