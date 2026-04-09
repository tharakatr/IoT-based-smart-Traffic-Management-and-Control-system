import cv2
import numpy as np
import onnxruntime as ort
import lgpio
import sys
import time

# --- GPIO SETUP ---
# Green = Traffic moving, Red = Emergency detected
GREEN_PIN = 17 
RED_PIN = 4

h = lgpio.gpiochip_open(0)

try:
    lgpio.gpio_claim_output(h, GREEN_PIN)
    lgpio.gpio_claim_output(h, RED_PIN)
except:
    pass # Already claimed

def set_status(emergency):
    if emergency:
        lgpio.gpio_write(h, GREEN_PIN, 0)
        lgpio.gpio_write(h, RED_PIN, 1) # STOP for Emergency
    else:
        lgpio.gpio_write(h, RED_PIN, 0)
        lgpio.gpio_write(h, GREEN_PIN, 1) # GO for Normal

# --- AI SETUP ---
session = ort.InferenceSession('best_compatible.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2]
EMERGENCY_CLASSES = [0, 3] # Ambulance, Fire Engine

# --- CAMERA SETUP ---
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("FATAL: Camera failed.")
    sys.exit()

try:
    print("System Live: Monitoring Lane for Emergency Vehicles...")
    while True:
        ret, frame = cap.read()
        if not ret: continue

        h_orig, w_orig = frame.shape[:2]
        
        # Preprocessing
        img = cv2.resize(frame, (size, size)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
        
        # Inference
        preds = np.squeeze(session.run(None, {input_name: img})[0]).T
        
        boxes, confs, class_ids = [], [], []
        for row in preds:
            conf = row[4:].max()
            if conf > 0.40: # High confidence for accuracy
                class_id = np.argmax(row[4:])
                x, y, w, h_box = row[:4]
                
                # Coordinate mapping
                x1 = int((x - w/2) * w_orig / size)
                y1 = int((y - h_box/2) * h_orig / size)
                boxes.append([x1, y1, int(w * w_orig / size), int(h_box * h_orig / size)])
                confs.append(float(conf))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.4, 0.45)
        
        veh_count = 0
        emergency_detected = False
        
        if len(indices) > 0:
            for i in indices.flatten():
                veh_count += 1
                label = "VEHICLE"
                color = (0, 255, 0)
                
                if class_ids[i] in EMERGENCY_CLASSES:
                    emergency_detected = True
                    label = "!! EMERGENCY !!"
                    color = (0, 0, 255)
                
                x, y, w, hb = boxes[i]
                cv2.rectangle(frame, (x, y), (x+w, y+hb), color, 2)
                cv2.putText(frame, label, (x, y-10), 0, 0.6, color, 2)

        # Update Hardware Status
        set_status(emergency_detected)

        # UI Overlay
        status_text = "EMERGENCY DETECTED" if emergency_detected else "NORMAL TRAFFIC"
        cv2.putText(frame, f"Density: {veh_count} | {status_text}", (20, 40), 0, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Single Lane Monitor", frame)
        if cv2.waitKey(1) == ord('q'): break

finally:
    lgpio.gpio_write(h, GREEN_PIN, 0)
    lgpio.gpio_write(h, RED_PIN, 0)
    cap.release()
    cv2.destroyAllWindows()
    lgpio.gpiochip_close(h)
