import cv2
import numpy as np
import onnxruntime as ort
import sys

print("Loading YOLOv8 Dual-Lane...")
session = ort.InferenceSession('best_compatible.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2]

# Ensure this index matches your ls -l /dev/video* output
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ADJUST THIS VARIABLE: Move the dividing line left or right (range: 0 to 320)
LANE_DIVIDER_X = 160 

if not cap.isOpened():
    print("FATAL: Camera dead.")
    sys.exit()

while True:
    ret, frame = cap.read()
    if not ret: continue

    img = cv2.resize(frame, (size, size)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    preds = np.squeeze(session.run(None, {input_name: img})[0]).T
    
    boxes, confs = [], []
    for row in preds:
        conf = row[4:].max()
        if conf > 0.40: 
            x, y, w, h = row[:4]
            x1 = int((x - w/2) * 320 / size)
            y1 = int((y - h/2) * 240 / size)
            boxes.append([x1, y1, int(w * 320 / size), int(h * 240 / size)])
            confs.append(float(conf))
            
    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.4, 0.4)
    lane1_count, lane2_count = 0, 0
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cx = x + (w // 2)
            
            # Sort into lanes based on your custom divider
            if cx < LANE_DIVIDER_X:
                lane1_count += 1
                color = (0, 255, 0) # Green
            else:
                lane2_count += 1
                color = (255, 0, 0) # Blue
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
    print(f"Density -> LANE 1: {lane1_count} | LANE 2: {lane2_count}")
    
    # Draw the custom dividing line
    cv2.line(frame, (LANE_DIVIDER_X, 0), (LANE_DIVIDER_X, 240), (0, 0, 255), 2)
    
    # The live window you require
    cv2.imshow("Smart Intersection Vision", frame)
    
    # Must use cv2.waitKey to keep the window open
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
