import cv2 
import torch
from PIL import Image
import playsound
import serial
import supervision as sv
from ultralytics import YOLO
import time

model = YOLO(f'best.pt')


stream_url = "" # Your custom IP URL
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


try:
    ser = serial.Serial('COM6', 9600, timeout=1)
    print("Serial port opened successfully.")
except serial.SerialException as e:
    print(f"Error: {e}")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    results = model(frame_pil)
    
    detections = results[0]

    if not detections:
        print("No Fire Detected")
        continue  

    for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  
            confidence = detection.conf[0]  
            label = detection.cls[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

           
            playsound.playsound('fire_detected.mp3')
         
    


    cv2.imshow("Stream View", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
