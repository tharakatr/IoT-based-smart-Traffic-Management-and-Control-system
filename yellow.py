import cv2
import numpy as np
import onnxruntime as ort
import lgpio
import time
import sys

# ─── GPIO PINS ───────────────────────────────
# Each lane: Green, Yellow, Red
L1_G, L1_Y, L1_R = 22, 27, 5
L2_G, L2_Y, L2_R = 24, 25, 12
L3_G, L3_Y, L3_R = 17, 16, 4
L4_G, L4_Y, L4_R = 23, 20, 6

ALL_PINS = [22,27,5, 24,25,12, 17,16,4, 23,20,6]

ALL_LANES = [
    ("L1", L1_G, L1_Y, L1_R),
    ("L2", L2_G, L2_Y, L2_R),
    ("L3", L3_G, L3_Y, L3_R),
    ("L4", L4_G, L4_Y, L4_R),
]

h = lgpio.gpiochip_open(0)
try:
    for p in ALL_PINS:
        lgpio.gpio_claim_output(h, p)
        lgpio.gpio_write(h, p, 0)
except: pass

# ─── AI SETUP ────────────────────────────────
session = ort.InferenceSession('best_compatible.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2]
EM_IDS = [0, 3]

YELLOW_DURATION = 10   # seconds warning before emergency override
EM_GREEN_MIN    = 15   # minimum green time for emergency lane

# ─── SIGNAL HELPERS ──────────────────────────
def all_red():
    for _, g, y, r in ALL_LANES:
        lgpio.gpio_write(h, g, 0)
        lgpio.gpio_write(h, y, 0)
        lgpio.gpio_write(h, r, 1)

def all_off():
    for p in ALL_PINS:
        lgpio.gpio_write(h, p, 0)

def set_signal(lane_name, state):
    """state: 'green', 'yellow', 'red'"""
    for name, g, y, r in ALL_LANES:
        if name == lane_name:
            lgpio.gpio_write(h, g, 1 if state=='green'  else 0)
            lgpio.gpio_write(h, y, 1 if state=='yellow' else 0)
            lgpio.gpio_write(h, r, 1 if state=='red'    else 0)
        else:
            lgpio.gpio_write(h, g, 0)
            lgpio.gpio_write(h, y, 0)
            lgpio.gpio_write(h, r, 1)

# ─── CAMERA SETUP ────────────────────────────
def find_cameras(need=4):
    found = []
    print(f"\n{'='*50}")
    print(f"Scanning for {need} cameras...")
    for i in range(10):
        cap = cv2.VideoCapture(f'/dev/video{i}', cv2.CAP_V4L2)
        if not cap.isOpened(): cap.release(); continue
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  /dev/video{i} -> no frame, skip")
            cap.release(); continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        print(f"  /dev/video{i} -> OK (slot {len(found)+1})")
        found.append(cap)
        if len(found) == need: break
    print(f"Found {len(found)}/{need} cameras\n{'='*50}\n")
    return found

def read_frame(cap):
    if cap is None: return np.zeros((480,640,3), dtype=np.uint8)
    ret, f = cap.read()
    return f if (ret and f is not None) else np.zeros((480,640,3), dtype=np.uint8)

# ─── AI DETECTION ────────────────────────────
def detect(frame):
    """Returns (cars, emergencies, annotated_frame)"""
    if frame is None or frame.size == 0:
        return 0, 0, np.zeros((480,640,3), dtype=np.uint8)
    frame = cv2.resize(frame, (640,480))
    ch, cw = frame.shape[:2]
    img = cv2.resize(frame,(size,size)).astype(np.float32)/255.0
    img = np.transpose(img,(2,0,1))[np.newaxis,:]
    preds = np.squeeze(session.run(None,{input_name:img})[0]).T
    boxes, confs, ids = [], [], []
    for row in preds:
        score = row[4:].max()
        if score > 0.28:
            x,y,w,hb = row[:4]
            bx = int((x-w/2)*cw/size); by = int((y-hb/2)*ch/size)
            boxes.append([bx,by,int(w*cw/size),int(hb*ch/size)])
            confs.append(float(score))
            ids.append(np.argmax(row[4:]))
    indices = cv2.dnn.NMSBoxes(boxes,confs,0.28,0.6)
    cars = ems = 0
    if len(indices) > 0:
        for i in indices.flatten():
            is_em = ids[i] in EM_IDS
            if is_em: ems += 1
            else: cars += 1
            color = (0,0,255) if is_em else (0,255,0)
            b = boxes[i]
            cv2.rectangle(frame,(b[0],b[1]),(b[0]+b[2],b[1]+b[3]),color,2)
            cv2.putText(frame,"EM" if is_em else "CAR",(b[0],b[1]-5),0,0.5,color,1)
    return cars, ems, frame

# ─── DISPLAY HELPERS ─────────────────────────
SIGNAL_COLORS = {
    'green':  (0,   255, 0),
    'yellow': (0,   200, 255),
    'red':    (0,   0,   255),
    'off':    (60,  60,  60),
}

def draw_signal_dot(frame, state):
    """Draw a small signal indicator dot top-right of frame"""
    color = SIGNAL_COLORS.get(state, SIGNAL_COLORS['off'])
    cv2.circle(frame, (615, 20), 12, color, -1)
    cv2.circle(frame, (615, 20), 12, (255,255,255), 1)

def add_label(frame, lane, signal_state, cars, ems, offline=False):
    if offline:
        cv2.rectangle(frame,(0,0),(639,479),(40,40,40),-1)
        cv2.putText(frame,f"{lane}: OFFLINE",(150,250),0,1.0,(80,80,80),2)
        return
    draw_signal_dot(frame, signal_state)
    color = SIGNAL_COLORS[signal_state]
    cv2.putText(frame, lane, (10,30), 0, 1.0, color, 2)
    cv2.putText(frame, f"C:{cars} E:{ems}", (10,60), 0, 0.6, (200,200,200), 1)
    if signal_state in ('green','yellow'):
        border_color = color
        cv2.rectangle(frame,(0,0),(639,479),border_color,4)

def make_grid(frames):
    top    = np.hstack((frames[0], frames[1]))
    bottom = np.hstack((frames[2], frames[3]))
    return np.vstack((top, bottom))

def overlay_status(grid, mode, active_lane, countdown, em_lane=None):
    h_g, w_g = grid.shape[:2]

    if mode == 'EMERGENCY':
        banner_color = (0, 0, 180)
        text = f"EMERGENCY: {em_lane} — clearing in {countdown:.1f}s" if countdown > 0 else f"EMERGENCY: {em_lane} GREEN"
    elif mode == 'YELLOW':
        banner_color = (0, 140, 220)
        text = f"WARNING: {active_lane} — yellow {countdown:.1f}s"
    elif mode == 'NORMAL':
        banner_color = (30, 100, 30)
        text = f"NORMAL: {active_lane} GREEN — {countdown:.1f}s"
    else:
        banner_color = (60,60,60)
        text = "SCANNING..."

    cv2.rectangle(grid, (0, h_g-50), (w_g, h_g), banner_color, -1)
    cv2.putText(grid, text, (20, h_g-15), 0, 0.9, (255,255,255), 2)

# ─── STARTUP ─────────────────────────────────
print("Smart Traffic System — 20 second startup delay")
for i in range(20, 0, -1):
    print(f"  Starting in {i}s...", end='\r')
    time.sleep(1)
print("\nLaunching!                    ")

caps = find_cameras(need=4)
if len(caps) < 2:
    print("FATAL: Need at least 2 cameras.")
    for c in caps: c.release()
    lgpio.gpiochip_close(h)
    sys.exit(1)
while len(caps) < 4:
    caps.append(None)

OFFLINE = [c is None for c in caps]
LANE_NAMES = ["L1","L2","L3","L4"]

win_name = "4-Lane Smart Traffic — AI + Emergency Override"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1280, 980)

print(f"Running — {4-sum(OFFLINE)}/4 cameras. Press Q to quit.\n")

# ─── CONTINUOUS STATE ────────────────────────
# Rolling detection window per lane (last N readings)
WINDOW = 8
lane_car_buf  = [[0]*WINDOW for _ in range(4)]
lane_em_buf   = [[0]*WINDOW for _ in range(4)]
buf_idx       = 0

# Normal rotation state
normal_order     = []   # sorted lane names
normal_idx       = 0    # which lane is currently green
normal_green_end = 0    # time when current green expires

# ─── MAIN LOOP ───────────────────────────────
all_red()
mode = 'SCAN'   # SCAN → NORMAL → YELLOW → EMERGENCY

try:
    while True:
        # ── Read all 4 cameras ──
        frames = [read_frame(c) for c in caps]
        results = [detect(f) for f in frames]   # (cars, ems, annotated)

        cars = [r[0] for r in results]
        ems  = [r[1] for r in results]
        imgs = [r[2] for r in results]

        # Update rolling buffers
        for i in range(4):
            lane_car_buf[i][buf_idx % WINDOW] = cars[i]
            lane_em_buf[i][buf_idx % WINDOW]  = ems[i]
        buf_idx += 1

        # Smoothed counts (max over window = most conservative)
        smooth_cars = [max(lane_car_buf[i]) for i in range(4)]
        smooth_ems  = [max(lane_em_buf[i])  for i in range(4)]

        # ── EMERGENCY CHECK (overrides everything) ──
        em_detected = any(e > 0 and not OFFLINE[i] for i, e in enumerate(smooth_ems))
        em_lane_idx = next((i for i,e in enumerate(smooth_ems) if e > 0 and not OFFLINE[i]), None)
        em_lane     = LANE_NAMES[em_lane_idx] if em_lane_idx is not None else None

        now = time.time()

        # ── STATE MACHINE ──────────────────────
        if mode in ('SCAN', 'NORMAL'):
            if em_detected:
                # Transition: find which lane is currently green, give it yellow
                mode = 'YELLOW'
                yellow_start = now
                # current active lane
                active_lane = normal_order[normal_idx] if normal_order else LANE_NAMES[0]
                set_signal(active_lane, 'yellow')
                print(f"[EMERGENCY DETECTED] {em_lane} — yellow on {active_lane} for {YELLOW_DURATION}s")

            else:
                # Normal rotation logic
                if mode == 'SCAN' or now >= normal_green_end or not normal_order:
                    # Recalculate priority
                    lane_scores = []
                    for i, name in enumerate(LANE_NAMES):
                        if OFFLINE[i]: continue
                        score = (smooth_ems[i]*100) + smooth_cars[i]
                        dur   = min(10 + smooth_cars[i]*2, 20)
                        lane_scores.append((name, score, dur, i))
                    lane_scores.sort(key=lambda x: x[1], reverse=True)
                    normal_order = [x[0] for x in lane_scores]
                    normal_idx   = 0

                    if normal_order:
                        active = normal_order[0]
                        dur    = next(x[2] for x in lane_scores if x[0]==active)
                        # Yellow transition from previous
                        if mode == 'NORMAL':
                            prev = normal_order[-1]
                            set_signal(prev, 'yellow')
                            time.sleep(3)   # brief yellow before switching
                        set_signal(active, 'green')
                        normal_green_end = now + dur
                        mode = 'NORMAL'
                        print(f"[NORMAL] Green: {active}  order={normal_order}  dur={dur}s")

        elif mode == 'YELLOW':
            elapsed = now - yellow_start
            remaining = YELLOW_DURATION - elapsed

            if remaining <= 0:
                # Yellow done — go all red, then give emergency lane green
                all_red()
                time.sleep(1)
                mode = 'EMERGENCY'
                em_green_start = now
                set_signal(em_lane, 'green')
                print(f"[EMERGENCY GREEN] {em_lane} is now GREEN")

        elif mode == 'EMERGENCY':
            elapsed_em = now - em_green_start

            # Check if emergency vehicle is still present
            still_em = smooth_ems[LANE_NAMES.index(em_lane)] > 0

            if not still_em and elapsed_em >= EM_GREEN_MIN:
                # Emergency cleared — transition back to normal
                set_signal(em_lane, 'yellow')
                time.sleep(3)
                all_red()
                time.sleep(1)
                mode = 'SCAN'
                normal_order = []
                print(f"[EMERGENCY CLEARED] Returning to normal rotation")

        # ── BUILD DISPLAY ──────────────────────
        signal_states = ['red'] * 4

        if mode == 'NORMAL' and normal_order:
            active = normal_order[normal_idx] if normal_idx < len(normal_order) else normal_order[0]
            active_idx = LANE_NAMES.index(active)
            signal_states[active_idx] = 'green'

        elif mode == 'YELLOW':
            active = normal_order[normal_idx] if normal_order else LANE_NAMES[0]
            active_idx = LANE_NAMES.index(active)
            signal_states[active_idx] = 'yellow'

        elif mode == 'EMERGENCY':
            em_idx = LANE_NAMES.index(em_lane)
            signal_states[em_idx] = 'green'

        for i, img in enumerate(imgs):
            add_label(img, LANE_NAMES[i], signal_states[i],
                      smooth_cars[i], smooth_ems[i], OFFLINE[i])

        grid = make_grid(imgs)

        # Countdown calculation for banner
        if mode == 'NORMAL' and normal_order:
            countdown = max(0, normal_green_end - now)
            overlay_status(grid, 'NORMAL', normal_order[0], countdown)
        elif mode == 'YELLOW':
            countdown = max(0, YELLOW_DURATION - (now - yellow_start))
            overlay_status(grid, 'YELLOW', active_lane, countdown)
        elif mode == 'EMERGENCY':
            countdown = max(0, EM_GREEN_MIN - (now - em_green_start))
            overlay_status(grid, 'EMERGENCY', em_lane, countdown, em_lane)
        else:
            overlay_status(grid, 'SCAN', '', 0)

        cv2.imshow(win_name, grid)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pass
