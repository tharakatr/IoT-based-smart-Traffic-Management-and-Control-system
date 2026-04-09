import cv2
import numpy as np
import onnxruntime as ort
import lgpio
import time

# ─── GPIO PINS ───────────────────────────────
L1_G, L1_Y, L1_R = 22, 27, 5
L2_G, L2_Y, L2_R = 24, 25, 12
L3_G, L3_Y, L3_R = 17, 16, 4
L4_G, L4_Y, L4_R = 23, 20, 6
RF_PIN            = 26  # BCM 26 = Physical pin 37

ALL_PINS  = [22,27,5, 24,25,12, 17,16,4, 23,20,6]
ALL_LANES = [
    ("L1", L1_G, L1_Y, L1_R),
    ("L2", L2_G, L2_Y, L2_R),
    ("L3", L3_G, L3_Y, L3_R),
    ("L4", L4_G, L4_Y, L4_R),
]
LANE_NAMES    = ["L1","L2","L3","L4"]
RF_LANE       = "L3"
RF_WARN_SECS  = 5
EM_WARN_SECS  = 10
EM_GREEN_MIN  = 15
WINDOW        = 8

# ─── GPIO ────────────────────────────────────
def init_gpio():
    global h
    while True:
        try:
            h = lgpio.gpiochip_open(0)
            for p in ALL_PINS:
                lgpio.gpio_claim_output(h, p)
                lgpio.gpio_write(h, p, 0)
            lgpio.gpio_claim_input(h, RF_PIN, lgpio.SET_PULL_DOWN)
            print("[GPIO] Ready")
            return
        except Exception as e:
            print(f"[GPIO] Init failed: {e} — retrying in 2s")
            time.sleep(2)

h = None
init_gpio()

def safe_write(pin, val):
    global h
    while True:
        try:
            lgpio.gpio_write(h, pin, val); return
        except Exception as e:
            print(f"[GPIO] Write error pin {pin}: {e} — reinit")
            time.sleep(1); init_gpio()

def read_rf():
    try:    return lgpio.gpio_read(h, RF_PIN) == 1
    except: return False

def all_red():
    for _,g,y,r in ALL_LANES:
        safe_write(g,0); safe_write(y,0); safe_write(r,1)

def all_yellow():
    for _,g,y,r in ALL_LANES:
        safe_write(g,0); safe_write(y,1); safe_write(r,0)

def set_signal(lane_name, state):
    for name,g,y,r in ALL_LANES:
        if name == lane_name:
            safe_write(g, 1 if state=='green'  else 0)
            safe_write(y, 1 if state=='yellow' else 0)
            safe_write(r, 1 if state=='red'    else 0)
        else:
            safe_write(g,0); safe_write(y,0); safe_write(r,1)

# ─── AI ──────────────────────────────────────
def init_ai():
    global session, input_name, size
    while True:
        try:
            session    = ort.InferenceSession('best_compatible.onnx',
                             providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            size       = session.get_inputs()[0].shape[2]
            print("[AI] Model loaded"); return
        except Exception as e:
            print(f"[AI] Load failed: {e} — retrying in 3s")
            time.sleep(3)

session = input_name = size = None
EM_IDS  = [0, 3]
init_ai()

# ─── CAMERAS ─────────────────────────────────
def open_camera(index):
    try:
        cap = cv2.VideoCapture(f'/dev/video{index}', cv2.CAP_V4L2)
        if not cap.isOpened(): cap.release(); return None
        ret, f = cap.read()
        if not ret or f is None: cap.release(); return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        return cap
    except: return None

def find_cameras(need=4):
    found = []
    print("Scanning cameras...")
    for i in range(10):
        cap = open_camera(i)
        if cap:
            print(f"  /dev/video{i} -> OK (slot {len(found)+1})")
            found.append(cap)
        if len(found) == need: break
    while len(found) < need: found.append(None)
    print(f"  {sum(c is not None for c in found)}/{need} active")
    return found

def recover_camera(caps, slot):
    try:
        if caps[slot]: caps[slot].release()
        for i in range(10):
            cap = open_camera(i)
            if cap:
                caps[slot] = cap
                print(f"[CAM] Slot {slot+1} recovered"); return
    except Exception as e:
        print(f"[CAM] Recovery failed slot {slot+1}: {e}")
    caps[slot] = None

def read_frame(cap):
    try:
        if cap is None: return False, np.zeros((480,640,3),dtype=np.uint8)
        ret, f = cap.read()
        if ret and f is not None: return True, f
        return False, np.zeros((480,640,3),dtype=np.uint8)
    except:
        return False, np.zeros((480,640,3),dtype=np.uint8)

# ─── DETECTION ───────────────────────────────
def detect(frame):
    try:
        if frame is None or frame.size == 0:
            return 0, 0, np.zeros((480,640,3),dtype=np.uint8)
        frame = cv2.resize(frame,(640,480))
        ch,cw = frame.shape[:2]
        img = cv2.resize(frame,(size,size)).astype(np.float32)/255.0
        img = np.transpose(img,(2,0,1))[np.newaxis,:]
        preds = np.squeeze(session.run(None,{input_name:img})[0]).T
        boxes,confs,ids = [],[],[]
        for row in preds:
            score = row[4:].max()
            if score > 0.28:
                x,y,w,hb = row[:4]
                bx=int((x-w/2)*cw/size); by=int((y-hb/2)*ch/size)
                boxes.append([bx,by,int(w*cw/size),int(hb*ch/size)])
                confs.append(float(score)); ids.append(np.argmax(row[4:]))
        indices = cv2.dnn.NMSBoxes(boxes,confs,0.28,0.6)
        cars=ems=0
        if len(indices) > 0:
            for i in indices.flatten():
                is_em = ids[i] in EM_IDS
                if is_em: ems+=1
                else:     cars+=1
                color=(0,0,255) if is_em else (0,255,0)
                b=boxes[i]
                cv2.rectangle(frame,(b[0],b[1]),(b[0]+b[2],b[1]+b[3]),color,2)
                cv2.putText(frame,"EM" if is_em else "CAR",(b[0],b[1]-5),0,0.5,color,1)
        return cars,ems,frame
    except Exception as e:
        print(f"[AI] Error: {e}")
        return 0,0,np.zeros((480,640,3),dtype=np.uint8)

# ─── DENSITY DURATION ────────────────────────
def density_duration(car_count):
    """Green time based purely on car density"""
    if car_count == 0:   return 8
    elif car_count <= 3: return 10 + (car_count * 2)
    elif car_count <= 7: return 16 + (car_count * 1)
    else:                return min(30, 23 + car_count)

# ─── DISPLAY ─────────────────────────────────
SIG_COLOR = {
    'green':  (0, 255,   0),
    'yellow': (0, 220, 255),
    'red':    (0,   0, 255),
}

def add_label(frame, lane, state, cars, ems, offline=False):
    try:
        if offline:
            cv2.rectangle(frame,(0,0),(639,479),(40,40,40),-1)
            cv2.putText(frame,f"{lane}: OFFLINE",(150,250),0,1.0,(80,80,80),2)
            return
        color = SIG_COLOR.get(state,(180,180,180))
        cv2.circle(frame,(615,20),14,color,-1)
        cv2.putText(frame, lane,             (10,32), 0, 1.1, color,       2)
        cv2.putText(frame, f"Cars:{cars}",   (10,62), 0, 0.6, (200,200,200),1)
        cv2.putText(frame, f"EM:{ems}",      (10,86), 0, 0.6,
                    (0,80,255) if ems>0 else (200,200,200), 1)
        if state in ('green','yellow'):
            cv2.rectangle(frame,(0,0),(639,479),color,5)
    except: pass

def make_grid(frames):
    try:
        return np.vstack((np.hstack((frames[0],frames[1])),
                          np.hstack((frames[2],frames[3]))))
    except:
        return np.zeros((960,1280,3),dtype=np.uint8)

def draw_banner(grid, mode, active, countdown):
    try:
        hg,wg = grid.shape[:2]
        cfg = {
            'NORMAL':    ((20,100,20),  f"NORMAL — {active} GREEN  |  {countdown:.1f}s"),
            'NORMAL_Y':  ((80,100,0),   f"SWITCHING — yellow on {active}  |  {countdown:.1f}s"),
            'EM_WARN':   ((0,120,180),  f"EMERGENCY DETECTED — warning {active}  |  {countdown:.1f}s"),
            'EMERGENCY': ((0,0,180),    f"EMERGENCY — {active} GREEN  |  {countdown:.1f}s"),
            'RF_WARN':   ((140,80,0),   f"RF SIGNAL — Lane 3 override  |  {countdown:.1f}s"),
            'RF_GREEN':  ((0,140,0),    f"RF OVERRIDE — Lane 3 GREEN  |  {countdown:.1f}s"),
            'SCAN':      ((60,60,60),   "SCANNING all lanes..."),
        }
        color, text = cfg.get(mode, ((60,60,60),"..."))
        cv2.rectangle(grid,(0,hg-55),(wg,hg),color,-1)
        cv2.putText(grid, text, (20,hg-18), 0, 0.85, (255,255,255), 2)
    except: pass

# ─── STARTUP ─────────────────────────────────
print("Smart Traffic System — starting in 20s")
for i in range(20,0,-1):
    print(f"  {i}s...", end='\r'); time.sleep(1)
print("\nLaunching!          ")

caps    = find_cameras(need=4)
OFFLINE = [c is None for c in caps]

cv2.namedWindow("Traffic", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic", 1280, 980)

# ─── STATE VARIABLES ─────────────────────────
buf_idx          = 0
lane_car_buf     = [[0]*WINDOW for _ in range(4)]
lane_em_buf      = [[0]*WINDOW for _ in range(4)]

mode             = 'SCAN'
active_lane      = LANE_NAMES[0]
normal_order     = []
normal_green_end = 0

em_lane          = None
em_green_start   = 0
warn_start       = 0

rf_warn_start    = 0
rf_green_start   = 0

all_red()
print("Running forever.\n")

# ─── MAIN LOOP ───────────────────────────────
while True:
    try:
        # ── Read cameras ──────────────────────
        frames = []
        for i,cap in enumerate(caps):
            ok,f = read_frame(cap)
            if not ok and cap is not None:
                recover_camera(caps,i)
                OFFLINE[i] = caps[i] is None
            frames.append(f)

        results     = [detect(f) for f in frames]
        cars        = [r[0] for r in results]
        ems         = [r[1] for r in results]
        imgs        = [r[2] for r in results]

        for i in range(4):
            lane_car_buf[i][buf_idx%WINDOW] = cars[i]
            lane_em_buf[i][buf_idx%WINDOW]  = ems[i]
        buf_idx += 1

        smooth_cars = [max(lane_car_buf[i]) for i in range(4)]
        smooth_ems  = [max(lane_em_buf[i])  for i in range(4)]

        rf_triggered = read_rf()
        any_em       = any(e>0 and not OFFLINE[i]
                           for i,e in enumerate(smooth_ems))
        now          = time.time()

        # ══════════════════════════════════════
        #  PRIORITY 1 — RF SIGNAL
        # ══════════════════════════════════════
        if rf_triggered and mode not in ('RF_WARN','RF_GREEN'):
            print(f"[RF] Triggered — Lane 3 in {RF_WARN_SECS}s")
            mode          = 'RF_WARN'
            rf_warn_start = now
            all_yellow()

        if mode == 'RF_WARN':
            elapsed = now - rf_warn_start
            if elapsed >= RF_WARN_SECS:
                all_red(); time.sleep(0.5)
                set_signal(RF_LANE,'green')
                mode           = 'RF_GREEN'
                rf_green_start = now
                print("[RF] Lane 3 GREEN")
            else:
                all_yellow()   # keep all yellow during countdown

        elif mode == 'RF_GREEN':
            if not rf_triggered and (now - rf_green_start) >= 10:
                set_signal(RF_LANE,'yellow'); time.sleep(3)
                all_red();                    time.sleep(0.5)
                mode         = 'SCAN'
                normal_order = []
                print("[RF] Ended — back to normal")

        # ══════════════════════════════════════
        #  PRIORITY 2 — EMERGENCY VEHICLES
        # ══════════════════════════════════════
        elif mode in ('SCAN','NORMAL'):
            if any_em:
                # Rank: EM count first → then car density
                scores = []
                for i,name in enumerate(LANE_NAMES):
                    if OFFLINE[i]: continue
                    scores.append((name, smooth_ems[i], smooth_cars[i]))
                scores.sort(key=lambda x:(x[1],x[2]), reverse=True)
                em_lane    = scores[0][0]
                warn_start = now
                mode       = 'EM_WARN'
                set_signal(active_lane,'yellow')
                print(f"[EM] Detected — top lane={em_lane}  warning {EM_WARN_SECS}s")

            else:
                # ══════════════════════════════
                #  PRIORITY 3 — NORMAL DENSITY
                # ══════════════════════════════
                time_left = normal_green_end - now

                # Last 3 seconds of green → yellow warning on current lane
                if mode == 'NORMAL' and 0 < time_left <= 3:
                    set_signal(active_lane, 'yellow')
                    print(f"[NORMAL] {active_lane} yellow warning — {time_left:.1f}s left")

                # Time expired → pick next lane by density
                if mode == 'SCAN' or now >= normal_green_end or not normal_order:
                    scores = []
                    for i,name in enumerate(LANE_NAMES):
                        if OFFLINE[i]: continue
                        c   = smooth_cars[i]
                        dur = density_duration(c)
                        scores.append((name, c, dur))
                    scores.sort(key=lambda x:x[1], reverse=True)
                    normal_order = [x[0] for x in scores]

                    if normal_order:
                        next_lane = normal_order[0]
                        dur       = next(x[2] for x in scores if x[0]==next_lane)

                        # Yellow on current lane 3s before switching
                        if mode == 'NORMAL' and next_lane != active_lane:
                            set_signal(active_lane,'yellow')
                            time.sleep(3)

                        active_lane      = next_lane
                        normal_green_end = now + dur
                        set_signal(active_lane,'green')
                        mode = 'NORMAL'
                        print(f"[NORMAL] Green:{active_lane}  "
                              f"Cars:{[(s[0],s[1]) for s in scores]}  "
                              f"Dur:{dur}s")

        elif mode == 'EM_WARN':
            if now - warn_start >= EM_WARN_SECS:
                all_red(); time.sleep(0.5)
                set_signal(em_lane,'green')
                mode           = 'EMERGENCY'
                em_green_start = now
                print(f"[EMERGENCY] {em_lane} GREEN")

        elif mode == 'EMERGENCY':
            # Re-rank continuously in case more EMs arrive
            scores = [(name, smooth_ems[i], smooth_cars[i])
                      for i,name in enumerate(LANE_NAMES) if not OFFLINE[i]]
            scores.sort(key=lambda x:(x[1],x[2]), reverse=True)
            top_em = scores[0][1] if scores else 0

            if top_em > 0:
                best = scores[0][0]
                if best != em_lane:
                    print(f"[EMERGENCY] Higher priority → switching to {best}")
                    set_signal(best,'green')
                    em_lane        = best
                    em_green_start = now
            else:
                # No more emergency vehicles → back to normal
                if (now - em_green_start) >= EM_GREEN_MIN:
                    set_signal(em_lane,'yellow'); time.sleep(3)
                    all_red();                    time.sleep(0.5)
                    mode         = 'SCAN'
                    normal_order = []
                    em_lane      = None
                    print("[EMERGENCY] Cleared — resuming density rotation")

        # ── BUILD DISPLAY ─────────────────────
        signal_states = ['red']*4
        if mode == 'NORMAL':
            signal_states[LANE_NAMES.index(active_lane)] = 'green'
        elif mode in ('NORMAL_Y','EM_WARN'):
            signal_states[LANE_NAMES.index(active_lane)] = 'yellow'
        elif mode == 'EMERGENCY' and em_lane:
            signal_states[LANE_NAMES.index(em_lane)]     = 'green'
        elif mode == 'RF_WARN':
            signal_states = ['yellow','yellow','yellow','yellow']
        elif mode == 'RF_GREEN':
            signal_states[LANE_NAMES.index(RF_LANE)]     = 'green'

        for i,img in enumerate(imgs):
            add_label(img,LANE_NAMES[i],signal_states[i],
                      smooth_cars[i],smooth_ems[i],OFFLINE[i])

        grid = make_grid(imgs)

        # Banner
        if mode == 'NORMAL':
            draw_banner(grid,'NORMAL', active_lane, max(0,normal_green_end-now))
        elif mode == 'EM_WARN':
            draw_banner(grid,'EM_WARN', active_lane, max(0,EM_WARN_SECS-(now-warn_start)))
        elif mode == 'EMERGENCY' and em_lane:
            draw_banner(grid,'EMERGENCY', em_lane,  max(0,EM_GREEN_MIN-(now-em_green_start)))
        elif mode == 'RF_WARN':
            draw_banner(grid,'RF_WARN',  '',         max(0,RF_WARN_SECS-(now-rf_warn_start)))
        elif mode == 'RF_GREEN':
            draw_banner(grid,'RF_GREEN', '',         max(0,10-(now-rf_green_start)))
        else:
            draw_banner(grid,'SCAN','',0)

        cv2.imshow("Traffic", grid)
        cv2.waitKey(1)

    except Exception as e:
        print(f"[LOOP ERROR] {e} — recovering...")
        time.sleep(0.5)
        try: all_red()
        except: pass
        continue