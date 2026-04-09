import cv2
import numpy as np
import onnxruntime as ort
import lgpio
import sys

# --- GPIO SETUP ---
RF_PIN = 27
GREEN_PINS = {"L1": 17, "L3": 23}
RED_PINS = {"L1": 4, "L3": 6}

h = lgpio.gpiochip_open(0)

try:
    lgpio.gpio_claim_input(h, RF_PIN)
    for pin in GREEN_PINS.values(): lgpio.gpio_claim_output(h, pin)
    for pin in RED_PINS.values(): lgpio.gpio_claim_output(h, pin)
except lgpio.error as e:
    print(f"GPIO Error. Kill previous scripts: {e}")
    sys.exit()

emergency_override = False

def rf_trigger(chip, gpio, level, timestamp):
    global emergency_override
    if level == 1:
        emergency_override = not emergency_override
        state = "ACTIVE: ALL RED" if emergency_override else "DISABLED: AI CONTROL"
        print(f"\n[!!!] EMERGENCY OVERRIDE {state} [!!!]\n")

lgpio.callback(h, RF_PIN, lgpio.RISING_EDGE, rf_trigger)

def set_all_red():
    for pin in GREEN_PINS.values(): lgpio.gpio_write(h, pin, 0)
    for pin in RED_PINS.values(): lgpio.gpio_write(h, pin, 1)

def set_green(lane):
    set_all_red()
    lgpio.gpio_write(h, RED_PINS[lane], 0)
    lgpio.gpio_write(h, GREEN_PINS[lane], 1)

# --- VISION SETUP ---
print("Loading 2-Camera AI...")
session = ort.InferenceSession('best_compatible.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2]

# Ensure indices match your ls -l /dev/video* output
cap1 = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap1.isOpened() or not cap2.isOpened():
    print("FATAL: Cannot open both cameras. USB bus failed.")
    lgpio.gpiochip_close(h)
    sys.exit()

def process_frame(frame, lane_name):
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
    count = 0
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, lane_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
    return count, frame

# --- MAIN LOOP ---
try:
    set_all_red()
    print("System Live: Cam1 -> Lane 1 | Cam2 -> Lane 3")
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2: continue

        if emergency_override:
            set_all_red()
            cv2.imshow("Dual Cam Vision", np.hstack((frame1, frame2)))
            if cv2.waitKey(100) == ord('q'): break
            continue

        l1_count, out1 = process_frame(frame1, "L1")
        l3_count, out2 = process_frame(frame2, "L3")
        
        densities = {"L1": l1_count, "L3": l3_count}
        
        winner = max(densities, key=densities.get)
        if densities[winner] == 0:
            set_green("L1")
        else:
            set_green(winner)

        cv2.imshow("Dual Cam Vision", np.hstack((out1, out2)))
        if cv2.waitKey(1) == ord('q'): break

except KeyboardInterrupt:
    print("\nShutting down safely.")
finally:
    set_all_red()
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    lgpio.gpiochip_close(h)
