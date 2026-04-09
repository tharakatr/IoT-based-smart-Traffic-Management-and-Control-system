import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# 1. CONFIGURATION
MODEL_PATH = '/home/bandham/Downloads/Traffic_Model_Final/weights/best_compatible.onnx'
# Put all your test images in this folder
IMAGE_FOLDER = '/home/bandham/Downloads/Traffic_Model_Final/test_images/' 

CLASSES = {
    0: "Ambulance",
    1: "Fire Engine",
    2: "Truck",
    3: "Car",
    4: "Bike"
}

# 2. LOAD AI ENGINE
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
size = session.get_inputs()[0].shape[2] 

def detect(frame):
    img = cv2.resize(frame, (size, size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return session.run(None, {input_name: img})[0]

def post_process(prediction, threshold=0.5):
    prediction = np.squeeze(prediction).T
    max_conf, best_class = 0, -1
    for row in prediction:
        scores = row[4:]
        class_id = np.argmax(scores)
        if scores[class_id] > threshold and scores[class_id] > max_conf:
            max_conf, best_class = scores[class_id], class_id
    return best_class, max_conf

# 3. PROCESS THE FOLDER
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
    print(f"Created folder: {IMAGE_FOLDER}. Put your images there and run again.")
    exit()

cv2.namedWindow("MULTI-IMAGE AI DEMO", cv2.WINDOW_NORMAL)

print("--- STARTING BATCH PROCESSING ---")
print("Press any key to see the NEXT image, or 'q' to quit.")

images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]

if not images:
    print("No images found in the folder!")
    exit()

for img_name in images:
    path = os.path.join(IMAGE_FOLDER, img_name)
    frame = cv2.imread(path)
    
    if frame is None: continue

    # Run AI
    class_id, conf = post_process(detect(frame))

    # UI Overlay
    label = CLASSES.get(class_id, "Scanning...")
    color = (0, 0, 255) if class_id in [0, 1] else (255, 120, 0) # RED for Emergency
    
    cv2.rectangle(frame, (0,0), (frame.shape[1], 70), (0,0,0), -1)
    cv2.putText(frame, f"FILE: {img_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"DETECTED: {label} ({int(conf*100)}%)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("MULTI-IMAGE AI DEMO", frame)
    
    # Wait for a key press to show the next image
    key = cv2.waitKey(0) 
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("Batch Demo Finished.")