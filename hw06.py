import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

model = YOLO("yolo11n.pt")

frame = picam2.capture_array()
results = model(frame, imgsz=320, conf=0.2)

annotated_frame = results[0].plot()
cv2.imwrite("yolo_camera_result.jpg", annotated_frame)

print("Saved: yolo_camera_result.jpg")