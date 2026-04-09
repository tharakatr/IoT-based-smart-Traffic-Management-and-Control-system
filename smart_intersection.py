import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import lgpio

# --- 1. HARDWARE CONFIG (SAFE PINS ONLY) ---
print("Initializing Hardware...")
h = lgpio.gpiochip_open(0)

PINS = {
    "A": [17, 27, 22],
    "B": [23, 24, 25], # SAFE PINS (Moved from SPI 10,9,11)
    "C": [5, 6, 13],
    "D": [16, 20, 12]  # SAFE PINS (Moved from SPI 19,26,21)
}

# Force release, then claim to prevent "Busy" errors
for lane in PINS.values():
    for pin in lane:
        try: lgpio.gpio_free(h, pin)
        except: pass
        lgpio.gpio_claim_output(h, pin)

def set_led(pin, state):
    lgpio.gpio_write(h, pin, state)

def set_all_red():
    for lane in PINS.values():
        set_led(lane[0], 1) # Red ON
        set_led(lane[1], 0)
        set_led(lane[2], 0)

def traffic_cycle(name, duration):
    print(f"\n>>> LANE {name} | GREEN | {duration}s")
    lane = PINS[name]
    
    set_led(lane[0], 0)
    set_led(lane[2], 1)
    
    for i in range(duration, 0, -1):
        print(f"Timer: {i:02d}", end="\r")
        time.sleep(1)
        
    set_led(lane[2], 0)
    set_led(lane[1], 1)
    time.sleep(2)
    
    set_led(lane[1], 0)
    set_led(lane[0], 1)

# --- 2. AI & CAMERA CONFIG ---
print("Loading YOLOv8 AI...")
try:
    session = ort.InferenceSession('best_compatible.onnx', providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    size = session.get_inputs()[0].shape[2]
except Exception as e:
    print(f"Model Error: {e}")
    sys.exit()

print("Hunting for active USB Camera...")
cap = None
for i in [0, 2, 4, 14, 16]:
    temp_cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if temp_cap.isOpened():
        ret, _ = temp_cap.read()
        if ret:
            print(f">>> Camera locked on /dev/video{i}")
            cap = temp_cap
            break
        temp_cap.release()

if cap is None:
    print("FATAL: Camera dead. Check power supply.")
    sys.exit()

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# --- 3. MAIN LOGIC ENGINE ---
try:
    set_all_red()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame dropped. Power limit likely exceeded.")
            break

        # Preprocess
        img = cv2.resize(frame, (size, size)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
        preds = np.squeeze(session.run(None, {input_name: img})[0]).T
        
        boxes, confidences = [], []
        
        # Filter weak predictions
        for row in preds:
            conf = row[4:].max()
            if conf > 0.50: # Threshold for toys
                x, y, w, h_box = row[:4]
                x1 = int((x - w/2) * 320 / size)
                y1 = int((y - h_box/2) * 240 / size)
                boxes.append([x1, y1, int(w * 320 / size), int(h_box * 240 / size)])
                confidences.append(float(conf))
                
        # NMS to remove duplicates
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Quadrant Mapping (1 Camera -> 4 Lanes)
        densities = {"A": 0, "B": 0, "C": 0, "D": 0}
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h_box = boxes[i]
                cx, cy = x + (w//2), y + (h_box//2)
                
                # Assign to lane based on center point
                if cy < 120:
                    if cx < 160: densities["A"] += 1
                    else: densities["B"] += 1
                else:
                    if cx < 160: densities["C"] += 1
                    else: densities["D"] += 1
                
                cv2.rectangle(frame, (x, y), (x+w, y+h_box), (0, 255, 0), 2)
                
        # Draw Quadrants
        cv2.line(frame, (160, 0), (160, 240), (255, 0, 0), 1)
        cv2.line(frame, (0, 120), (320, 120), (255, 0, 0), 1)
        cv2.imshow("Intersection AI", frame)

        # Decide Winner
        winner = max(densities, key=densities.get)
        t_green = 5 + (densities[winner] * 2) # 5s base + 2s per toy
        
        # Execute Lights
        traffic_cycle(winner, t_green)
        
        if cv2.waitKey(1) == ord('q'): 
            break

except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    set_all_red()
    for lane in PINS.values():
        for pin in lane:
            lgpio.gpio_free(h, pin)
    lgpio.gpiochip_close(h)
    cap.release()
    cv2.destroyAllWindows()
