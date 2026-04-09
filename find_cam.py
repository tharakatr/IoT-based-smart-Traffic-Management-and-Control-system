import cv2

def find_cameras():
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # CAP_DSHOW is faster for Windows USB
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Index {i}: WORKING")
                cv2.imshow(f"Camera {i}", frame)
                cv2.waitKey(2000) # Show for 2 seconds
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"Index {i}: Not found")

find_cameras()