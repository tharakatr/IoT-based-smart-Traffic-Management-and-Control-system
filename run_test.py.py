from ultralytics import YOLO
import cv2

# Load the model - ensure 'best.pt' is in this same folder
model = YOLO('best.pt') 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run detection
        results = model(frame, imgsz=320, conf=0.4)
        
        # Draw boxes
        annotated_frame = results[0].plot()
        
        cv2.imshow("Smart Intersection Test", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()