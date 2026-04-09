import cv2
import numpy as np
import onnxruntime as ort
import time

# 1. PATHS
MODEL_PATH = '/home/bandham/Downloads/Traffic_Model_Final/weights/best_compatible.onnx'
# REPLACE THIS with the actual path to a traffic image on your Pi
IMAGE_PATH = '/home/bandham/Downloads/Traffic_Model_Final/test_traffic2.jpeg' 
# UPDATE THIS DICTIONARY HERE:
CLASSES = {
    0: "Ambulance",
    1: "Fire Engine",
    2: "Truck",
    3: "Car",
    4: "Bike"
}

# 2. LOAD AI ENGINE
print("Loading AI Model for Static Test...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2] 

def detect(frame):
    img = cv2.resize(frame, (size, size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return session.run(None, {input_name: img})[0]

def post_process(prediction, threshold=0.4):
    prediction = np.squeeze(prediction).T
    max_conf, best_class = 0, -1
    for row in prediction:
        scores = row[4:]
        class_id = np.argmax(scores)
        if scores[class_id] > threshold and scores[class_id] > max_conf:
            max_conf, best_class = scores[class_id], class_id
    return best_class, max_conf

# 3. LOAD THE IMAGE
test_frame = cv2.imread(IMAGE_PATH)
if test_frame is None:
    print(f"ERROR: Could not find image at {IMAGE_PATH}")
    exit()

cv2.namedWindow("STATIC AI TEST", cv2.WINDOW_NORMAL)

print("--- STARTING STATIC TEST ---")
print("Press 'q' to stop.")

try:
    while True:
        # We use a copy so we don't draw over the original image every loop
        display_frame = test_frame.copy()

        # Run AI on the static image
        class_id, conf = post_process(detect(display_frame))

        # UI Overlay
        label = CLASSES.get(class_id, "No Vehicle") if class_id != -1 else "Scanning..."
        color = (0, 255, 0) if class_id in [0, 3] else (255, 120, 0)
        
        cv2.rectangle(display_frame, (0,0), (display_frame.shape[1], 60), (0,0,0), -1)
        cv2.putText(display_frame, f"TEST MODE: {label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("STATIC AI TEST", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)

except Exception as e:
    print(f"Error: {e}")
finally:
    cv2.destroyAllWindows()
