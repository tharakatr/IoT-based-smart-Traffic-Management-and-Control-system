import cv2
import numpy as np

def get_lane_densities(cap0, cap1, session, input_name, size):
    """
    Returns a dictionary with the vehicle count for all 4 lanes.
    """
    densities = {"Lane_A": 0, "Lane_B": 0, "Lane_C": 0, "Lane_D": 0}
    
    # Process both cameras
    for cam_id, cap in enumerate([cap0, cap1]):
        ret, frame = cap.read()
        if not ret: continue
        
        h, w, _ = frame.shape
        # Preprocess for AI
        img = cv2.resize(frame, (size, size)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
        preds = np.squeeze(session.run(None, {input_name: img})[0]).T

        for row in preds:
            conf = row[4:].max()
            if conf > 0.45: # Threshold
                # Get center coordinates of the vehicle
                cx = int(row[0] * w / size)
                cy = int(row[1] * h / size)

                # Split Logic
                if cam_id == 0: # Camera 1 (Lanes A & B)
                    if cx < w // 2: densities["Lane_A"] += 1
                    else: densities["Lane_B"] += 1
                else: # Camera 2 (Lanes C & D)
                    if cx < w // 2: densities["Lane_C"] += 1
                    else: densities["Lane_D"] += 1
                    
    return densities