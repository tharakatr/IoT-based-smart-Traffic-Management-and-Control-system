import cv2
import numpy as np
import onnxruntime as ort
import lgpio
import time
import sys

# --- GPIO SETUP ---
L1_G, L1_R = 22, 5
L2_G, L2_R = 24, 12
L3_G, L3_R = 17, 4
L4_G, L4_R = 23, 6

h = lgpio.gpiochip_open(0)

ALL_PROJECT_PINS = [17, 4, 22, 5, 23, 6, 24, 12]
try:
    for p in ALL_PROJECT_PINS:
        lgpio.gpio_claim_output(h, p)
        lgpio.gpio_write(h, p, 0)
except: pass

session = ort.InferenceSession('best_compatible.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2]
EM_IDS = [0, 3]

def find_and_open_cameras():
    found = []
    print("Scanning for USB cameras...")
    PATHS = ['/dev/video2', '/dev/video4', '/dev/video6', '/dev/video0']
    for path in PATHS:
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"  X {path} -> Could not open")
            cap.release()
            continue
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  ! {path} -> No frame (USB power issue?)")
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        found.append(cap)
        print(f"  OK {path} -> WORKING")
    print(f"Total cameras: {len(found)}")
    return found

def read_frame(cap):
    if cap is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    ret, frame = cap.read()
    return frame if (ret and frame is not None) else np.zeros((480, 640, 3), dtype=np.uint8)

def get_smart_data(frame):
    if frame is None or frame.size == 0:
        return 0, 0, np.zeros((480, 640, 3), dtype=np.uint8)
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
            bx = int((x - w/2) * cw / size)
            by = int((y - hb/2) * ch / size)
            boxes.append([bx, by, int(w * cw / size), int(hb * ch / size)])
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
            cv2.putText(frame, "EM" if is_em else "CAR", (b[0], b[1]-5), 0, 0.5, color, 1)
    return cars, ems, frame

def add_label(frame, label, active=False, offline=False):
    if offline:
        cv2.rectangle(frame, (0, 0), (639, 479), (60, 60, 60), -1)
        cv2.putText(frame, f"{label}: NO CAMERA", (160, 250), 0, 1.0, (100, 100, 100), 2)
        return
    color = (0, 255, 0) if active else (255, 255, 0)
    cv2.putText(frame, label, (10, 30), 0, 1.0, color, 2)
    if active:
        cv2.rectangle(frame, (0, 0), (639, 479), color, 4)

def make_2x2_grid(frames):
    top = np.hstack((frames[0], frames[1]))
    bottom = np.hstack((frames[2], frames[3]))
    return np.vstack((top, bottom))

def get_duration(em, cars):
    return 25 if em > 0 else min(10 + (cars * 2), 20)

def get_priority(em, cars, offline):
    if offline: return -1
    return (em * 100) + cars

caps = find_and_open_cameras()

if len(caps) < 3:
    print(f"Only {len(caps)} camera(s) found! Need at least 3.")
    for cap in caps: cap.release()
    lgpio.gpiochip_close(h)
    sys.exit()

while len(caps) < 4:
    caps.append(None)
    print("  Camera slot padded with blank offline feed")

cap1, cap2, cap3, cap4 = caps
OFFLINE = [cap is None for cap in caps]
ALL_LANES = [(L1_G, L1_R), (L2_G, L2_R), (L3_G, L3_R), (L4_G, L4_R)]

win_name = "4-Lane Smart Traffic System"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1280, 960)
print(f"System ready! {4 - sum(OFFLINE)}/4 cameras active. Press q to quit.")

try:
    while True:
        for g, r in ALL_LANES:
            lgpio.gpio_write(h, g, 0)
            lgpio.gpio_write(h, r, 1)

        l1_c, l1_e = [], []
        l2_c, l2_e = [], []
        l3_c, l3_e = [], []
        l4_c, l4_e = [], []
        start_scan = time.time()

        while time.time() - start_scan < 5.0:
            f1 = read_frame(cap1)
            f2 = read_frame(cap2)
            f3 = read_frame(cap3)
            f4 = read_frame(cap4)
            c1, e1, o1 = get_smart_data(f1)
            c2, e2, o2 = get_smart_data(f2)
            c3, e3, o3 = get_smart_data(f3)
            c4, e4, o4 = get_smart_data(f4)
            l1_c.append(c1); l1_e.append(e1)
            l2_c.append(c2); l2_e.append(e2)
            l3_c.append(c3); l3_e.append(e3)
            l4_c.append(c4); l4_e.append(e4)
            add_label(o1, "L1", offline=OFFLINE[0])
            add_label(o2, "L2", offline=OFFLINE[1])
            add_label(o3, "L3", offline=OFFLINE[2])
            add_label(o4, "L4", offline=OFFLINE[3])
            grid = make_2x2_grid([o1, o2, o3, o4])
            rem = 5.0 - (time.time() - start_scan)
            cv2.putText(grid, f"SCANNING: {rem:.1f}s", (500, 50), 0, 1.5, (0, 255, 255), 3)
            cv2.imshow(win_name, grid)
            if cv2.waitKey(1) == ord('q'): sys.exit()

        f_c1, f_e1 = max(l1_c), max(l1_e)
        f_c2, f_e2 = max(l2_c), max(l2_e)
        f_c3, f_e3 = max(l3_c), max(l3_e)
        f_c4, f_e4 = max(l4_c), max(l4_e)

        lane_data = [
            ("L1", get_priority(f_e1,f_c1,OFFLINE[0]), get_duration(f_e1,f_c1), L1_G, L1_R, f_c1, f_e1),
            ("L2", get_priority(f_e2,f_c2,OFFLINE[1]), get_duration(f_e2,f_c2), L2_G, L2_R, f_c2, f_e2),
            ("L3", get_priority(f_e3,f_c3,OFFLINE[2]), get_duration(f_e3,f_c3), L3_G, L3_R, f_c3, f_e3),
            ("L4", get_priority(f_e4,f_c4,OFFLINE[3]), get_duration(f_e4,f_c4), L4_G, L4_R, f_c4, f_e4),
        ]
        sequence = sorted(lane_data, key=lambda x: x[1], reverse=True)
        print(f"Priority order: {[x[0] for x in sequence]}")

        for lane, p_score, dur, g_pin, r_pin, cars, ems in sequence:
            if p_score == -1:
                print(f"  Skipping {lane} - camera offline")
                continue
            st = time.time()
            for tg, tr in ALL_LANES:
                if tg == g_pin:
                    lgpio.gpio_write(h, tg, 1); lgpio.gpio_write(h, tr, 0)
                else:
                    lgpio.gpio_write(h, tg, 0); lgpio.gpio_write(h, tr, 1)
            while time.time() - st < dur:
                f1 = read_frame(cap1)
                f2 = read_frame(cap2)
                f3 = read_frame(cap3)
                f4 = read_frame(cap4)
                _, _, o1 = get_smart_data(f1)
                _, _, o2 = get_smart_data(f2)
                _, _, o3 = get_smart_data(f3)
                _, _, o4 = get_smart_data(f4)
                add_label(o1, "L1", active=(lane=="L1"), offline=OFFLINE[0])
                add_label(o2, "L2", active=(lane=="L2"), offline=OFFLINE[1])
                add_label(o3, "L3", active=(lane=="L3"), offline=OFFLINE[2])
                add_label(o4, "L4", active=(lane=="L4"), offline=OFFLINE[3])
                grid = make_2x2_grid([o1, o2, o3, o4])
                rem = dur - (time.time() - st)
                color = (0, 0, 255) if ems > 0 else (0, 255, 0)
                cv2.rectangle(grid, (540, 430), (740, 500), (0, 0, 0), -1)
                cv2.putText(grid, f"{rem:.1f}s", (560, 490), 0, 1.8, color, 4)
                tag = " [EMERGENCY]" if ems > 0 else ""
                cv2.putText(grid, f"ACTIVE: {lane}{tag}", (20, 50), 0, 1.2, color, 3)
                stats = f"L1:C{f_c1}E{f_e1}  L2:C{f_c2}E{f_e2}  L3:C{f_c3}E{f_e3}  L4:C{f_c4}E{f_e4}"
                cv2.putText(grid, stats, (20, 940), 0, 0.8, (255, 255, 255), 2)
                cam_st = f"CAM: {'ON' if not OFFLINE[0] else 'OFF'} {'ON' if not OFFLINE[1] else 'OFF'} {'ON' if not OFFLINE[2] else 'OFF'} {'ON' if not OFFLINE[3] else 'OFF'}"
                cv2.putText(grid, cam_st, (900, 40), 0, 0.8, (200, 200, 200), 2)
                cv2.imshow(win_name, grid)
                if cv2.waitKey(1) == ord('q'): sys.exit()

finally:
    print("Shutting down...")
    for cap in caps:
        if cap is not None: cap.release()
    for p in ALL_PROJECT_PINS:
        try: lgpio.gpio_write(h, p, 0)
        except: pass
    lgpio.gpiochip_close(h)
    cv2.destroyAllWindows()
    print("Done.")
