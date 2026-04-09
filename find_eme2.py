import cv2
import numpy as np
import onnxruntime as ort
import lgpio
import time
import sys

# --- GPIO SETUP ---
L1_G, L1_R, L2_G, L2_R, L3_G, L3_R = 22, 5, 24, 12, 17, 4
h = lgpio.gpiochip_open(0)

ALL_PROJECT_PINS = [17, 4, 22, 5, 23, 6, 24, 12] 
try:
    for p in ALL_PROJECT_PINS: 
        lgpio.gpio_claim_output(h, p)
        lgpio.gpio_write(h, p, 0)
except: pass

# --- AI SETUP ---
session = ort.InferenceSession('best_compatible.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2]
EM_IDS = [0, 3]

def find_and_open_cameras():
    found = []
    print("Searching for 3 cameras. Forcing low bandwidth to bypass USB limits...")
    for i in [0, 2, 4]: # Locked to your physical hardware indices
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                cap.set(cv2.CAP_PROP_FPS, 10)
                found.append(cap)
                print(f"Camera locked at index {i}")
            else:
                cap.release()
        if len(found) == 3: break
    return found

def get_smart_data(frame):
    if frame is None or frame.size == 0:
        return 0, 0, np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Full 640x480 frame. No cropping.
    frame = cv2.resize(frame, (640, 480))
    ch, cw = frame.shape[:2]
    
    img = cv2.resize(frame, (size, size)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    preds = np.squeeze(session.run(None, {input_name: img})[0]).T
    
    boxes, confs, ids = [], [], []
    for row in preds:
        score = row[4:].max()
        if score > 0.28:
            x, y, w, hb = row[:4]
            bx, by = int((x-w/2)*cw/size), int((y-hb/2)*ch/size)
            boxes.append([bx, by, int(w*cw/size), int(hb*ch/size)])
            confs.append(float(score))
            ids.append(np.argmax(row[4:]))

    indices = cv2.dnn.NMSBoxes(boxes, confs, 0.28, 0.6)
    cars, ems = 0, 0
    if len(indices) > 0:
        for i in indices.flatten():
            is_em = ids[i] in EM_IDS
            if is_em: ems += 1
            else: cars += 1
            color = (0, 0, 255) if is_em else (0, 255, 0)
            b = boxes[i]
            cv2.rectangle(frame, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), color, 2)
            cv2.putText(frame, f"{'EM' if is_em else 'CAR'}", (b[0], b[1]-5), 0, 0.5, color, 1)
            
    return cars, ems, frame

# --- INITIALIZATION ---
caps = find_and_open_cameras()
if len(caps) < 3: sys.exit("Error: Could not lock 3 cameras. USB data pipe collapsed.")
cap1, cap2, cap3 = caps

win_name = "3-Lane Smart Traffic System"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 

try:
    while True:
        # --- PHASE 1: SCANNING ---
        for g, r in [(L1_G, L1_R), (L2_G, L2_R), (L3_G, L3_R)]:
            lgpio.gpio_write(h, g, 0); lgpio.gpio_write(h, r, 1)

        l1_c, l1_e, l2_c, l2_e, l3_c, l3_e = [], [], [], [], [], []
        start_scan = time.time()
        
        while time.time() - start_scan < 5.0:
            _, f1 = cap1.read(); _, f2 = cap2.read(); _, f3 = cap3.read()
            c1, e1, o1 = get_smart_data(f1)
            c2, e2, o2 = get_smart_data(f2)
            c3, e3, o3 = get_smart_data(f3)
            
            l1_c.append(c1); l1_e.append(e1)
            l2_c.append(c2); l2_e.append(e2)
            l3_c.append(c3); l3_e.append(e3)
            
            display = np.hstack((o1, o2, o3))
            rem_scan = 5.0 - (time.time() - start_scan)
            
            # Centered for 1920px width
            cv2.putText(display, f"SCANNING: {rem_scan:.1f}s", (800, 50), 0, 1.2, (0, 255, 255), 3)
            cv2.imshow(win_name, display)
            if cv2.waitKey(1) == ord('q'): sys.exit()

        f_e1, f_e2, f_e3 = max(l1_e), max(l2_e), max(l3_e)
        f_c1, f_c2, f_c3 = max(l1_c), max(l2_c), max(l3_c)

        # --- PHASE 2: CALCULATE PRIORITY & TIMING ---
        d1 = 25 if f_e1 > 0 else min(10 + (f_c1 * 2), 20)
        d2 = 25 if f_e2 > 0 else min(10 + (f_c2 * 2), 20)
        d3 = 25 if f_e3 > 0 else min(10 + (f_c3 * 2), 20)
        
        p1 = (f_e1 * 100) + f_c1
        p2 = (f_e2 * 100) + f_c2
        p3 = (f_e3 * 100) + f_c3

        lane_data = [
            ("L1", p1, d1, L1_G, L1_R, f_c1, f_e1),
            ("L2", p2, d2, L2_G, L2_R, f_c2, f_e2),
            ("L3", p3, d3, L3_G, L3_R, f_c3, f_e3)
        ]
        sequence = sorted(lane_data, key=lambda x: x[1], reverse=True)

        # --- PHASE 3: EXECUTION ---
        for lane, p_score, dur, g_pin, r_pin, cars, ems in sequence:
            st = time.time()
            
            for tg, tr in [(L1_G, L1_R), (L2_G, L2_R), (L3_G, L3_R)]:
                if tg == g_pin:
                    lgpio.gpio_write(h, tg, 1); lgpio.gpio_write(h, tr, 0)
                else:
                    lgpio.gpio_write(h, tg, 0); lgpio.gpio_write(h, tr, 1)
            
            while time.time() - st < dur:
                _, lf1 = cap1.read(); _, lf2 = cap2.read(); _, lf3 = cap3.read()
                _, _, o1 = get_smart_data(lf1)
                _, _, o2 = get_smart_data(lf2)
                _, _, o3 = get_smart_data(lf3)
                
                view = np.hstack((o1, o2, o3))
                rem = dur - (time.time() - st)
                color = (0, 0, 255) if ems > 0 else (0, 255, 0)
                
                # Centered UI Overlays for 1920px width
                cv2.rectangle(view, (840, 400), (1080, 470), (0,0,0), -1)
                cv2.putText(view, f"{rem:.1f}s", (870, 455), 0, 1.5, color, 4)
                cv2.putText(view, f"ACTIVE: {lane}", (20, 40), 0, 1.0, color, 3)
                cv2.putText(view, f"L1:C{f_c1} E{f_e1} | L2:C{f_c2} E{f_e2} | L3:C{f_c3} E{f_e3}", (1300, 40), 0, 0.8, (255, 255, 255), 2)
                
                cv2.imshow(win_name, view)
                if cv2.waitKey(1) == ord('q'): sys.exit()

finally:
    for cap in caps: cap.release()
    for p in ALL_PROJECT_PINS: 
        try: lgpio.gpio_write(h, p, 0)
        except: pass
    lgpio.gpiochip_close(h)
