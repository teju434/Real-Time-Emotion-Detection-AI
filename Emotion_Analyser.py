import cv2
from deepface import DeepFace
import time

# -------- CONFIG --------
EMOTIONS = ["angry", "happy", "sad", "surprise", "neutral"]
WHITE = (255, 255, 255)
NEON_PINK = (255, 0, 255)
ANALYZE_EVERY = 0.6  # seconds between analyses
MAX_CAMERAS = 5      # max indices to check for camera

# -------- CAMERA SETUP --------
def find_camera(max_index=MAX_CAMERAS):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows backend
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            return cap
        cap.release()
    return None

cap = find_camera()
if cap is None:
    print("❌ No working camera found")
    exit()

# -------- INITIALIZATION --------
last_analysis_time = 0
emotion_scores = {e: 0 for e in EMOTIONS}
top_emotion = "neutral"
top_conf = 0

# -------- EMOTION ANALYSIS FUNCTION --------
def analyze_emotion(frame_bgr):
    global emotion_scores, top_emotion, top_conf
    try:
        result = DeepFace.analyze(
            frame_bgr,
            actions=["emotion"],
            enforce_detection=False
        )
        emotions = result[0]["emotion"]
        for e in EMOTIONS:
            emotion_scores[e] = float(emotions.get(e, 0))
        top_emotion = max(EMOTIONS, key=lambda e: emotion_scores[e])
        top_conf = int(emotion_scores[top_emotion])
    except Exception as e:
        print("⚠️ DeepFace analyze error:", e)

# -------- MAIN LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame")
        break

    frame = cv2.flip(frame, 1)  # mirror view
    h, w, _ = frame.shape

    # -------- HUD HEADER --------
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, "AI FACE EMOTION HUD",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                NEON_PINK,
                2)

    # -------- EMOTION ANALYSIS --------
    if time.time() - last_analysis_time > ANALYZE_EVERY:
        analyze_emotion(frame)
        last_analysis_time = time.time()

    # -------- FACE BOX --------
    box_w, box_h = 260, 300
    x = w // 2 - box_w // 2
    y = h // 2 - box_h // 2
    cv2.rectangle(frame, (x - 3, y - 3), (x + box_w + 3, y + box_h + 3), (200, 200, 200), 2)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), WHITE, 2)

    # Top emotion label
    label = f"{top_emotion} ({top_conf}%)"
    cv2.rectangle(frame, (x, y - 30), (x + box_w, y), (0, 0, 0), -1)
    cv2.putText(frame, label, (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # -------- EMOTION BARS PANEL --------
    panel_x, panel_y = 20, 60
    line_h = 28
    max_bar_w = 160
    cv2.putText(frame, "Tracked emotions:", (panel_x, panel_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    for i, emo in enumerate(EMOTIONS):
        y_off = panel_y + i * line_h
        score = emotion_scores[emo]
        bar_w = int((score / 100) * max_bar_w)

        # emotion name
        cv2.putText(frame, f"{emo:8s}", (panel_x, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        # bar background
        cv2.rectangle(frame, (panel_x + 90, y_off - 12),
                      (panel_x + 90 + max_bar_w, y_off + 4), (50, 50, 50), -1)

        # bar fill
        color = NEON_PINK if emo == top_emotion else (180, 180, 180)
        cv2.rectangle(frame, (panel_x + 90, y_off - 12),
                      (panel_x + 90 + bar_w, y_off + 4), color, -1)

    # quit instruction
    cv2.putText(frame, "Press 'q' to quit", (w - 210, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

    cv2.imshow("AI Face Emotion HUD", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
