from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, imgsz=320, conf=0.4)
    
    # Extract detected class IDs
    # 0=Ambulance, 3=Fire_Engine
    detections = results[0].boxes.cls.tolist()
    vehicle_count = len(detections)
    
    # Logic: Priority for Emergency Vehicles
    if 0.0 in detections or 3.0 in detections:
        status = "EMERGENCY: PRIORITY GREEN"
        color = (0, 0, 255) # Red text for emergency
    elif vehicle_count > 5:
        status = "HEAVY TRAFFIC: EXTEND GREEN"
        color = (0, 255, 0)
    else:
        status = "NORMAL TRAFFIC"
        color = (255, 255, 255)

    # Display results
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"Count: {vehicle_count}", (20, 40), 1, 2, color, 2)
    cv2.putText(annotated_frame, status, (20, 80), 1, 1.5, color, 2)
    
    cv2.imshow("Smart Intersection Control", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()