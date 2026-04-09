import cv2
import numpy as np
import onnxruntime as ort
import sys

print("Loading YOLOv8 Dual-Camera Engine...")
session = ort.InferenceSession('best_compatible.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2]

# YOU MUST VERIFY THESE INDICES WITH ls -l /dev/video*
# Usually they are 0 and 2, or 0 and 4.
cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2) 

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap1.isOpened() or not cap2.isOpened():
    print("FATAL: Cannot power both cameras. USB bus collapsed.")
    sys.exit()

LANE_DIVIDER = 160

def process_frame(frame, cam_name):
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
    left_count, right_count = 0, 0
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cx = x + (w // 2)
            
            if cx < LANE_DIVIDER:
                left_count += 1
                color = (0, 255, 0)
            else:
                right_count += 1
                color = (255, 0, 0)
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
    cv2.line(frame, (LANE_DIVIDER, 0), (LANE_DIVIDER, 240), (0, 0, 255), 2)
    return frame, left_count, right_count

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Hardware failure: Frame dropped.")
        continue

    # Process Cam 1 (Lanes 1 & 2)
    out1, l1, l2 = process_frame(frame1, "Cam1")
    # Process Cam 2 (Lanes 3 & 4)
    out2, l3, l4 = process_frame(frame2, "Cam2")
    
    print(f"Cam 1 (L1:{l1} | L2:{l2})  ---  Cam 2 (L3:{l3} | L4:{l4})")
    
    # Combine images side-by-side to bypass multiple window Wayland crashes
    combined_view = np.hstack((out1, out2))
    cv2.imshow("Dual Camera 4-Lane Vision", combined_view)
    
    if cv2.waitKey(1) == ord('q'): break

cap1.release()
cap2.release()
cv2.destroyAllWindows()