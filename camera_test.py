import cv2

def find_working_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            return cap
        cap.release()
    return None

cap = find_working_camera()
if cap is None:
    print("❌ No working camera found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    cv2.imshow("Camera Test", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
