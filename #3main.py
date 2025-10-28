import cv2, os, time
import numpy as np
import pandas as pd
from datetime import datetime, date






# ---------- CONFIG ----------
CASCADE_FACE = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
CASCADE_EYE  = cv2.data.haarcascades + 'haarcascade_eye.xml'
TRAINER_DIR = 'trainer'
ATTENDANCE_DIR = 'attendance_logs'  #a folder for daily attendance logs
SNAPSHOT_DIR = 'snapshots'
CONFIDENCE_THRESHOLD = 70
MASK_SCORE_THRESHOLD = 0.25
LOG_ONCE_PER_DAY = True
SAVE_SNAPSHOT_ON_LOG = True
# ----------------------------




os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)


today_str = date.today().strftime("%Y-%m-%d")
CSV_FILE = os.path.join(ATTENDANCE_DIR, f"attendance_{today_str}.csv")


face_cascade = cv2.CascadeClassifier(CASCADE_FACE)
eye_cascade  = cv2.CascadeClassifier(CASCADE_EYE)

#recognizer & names
recognizer = None
names = {}
if os.path.exists(os.path.join(TRAINER_DIR, 'trainer.yml')):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(TRAINER_DIR, 'trainer.yml'))
    names = np.load(os.path.join(TRAINER_DIR, 'names.npy'), allow_pickle=True).item()
    print("Loaded recognizer and names:", names)
else:
    print("No trained recognizer found. Run train_recognizer.py after creating dataset.")
    recognizer = None

# create attendance CSV
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=['ID', 'Name', 'Time', 'Mask', 'Confidence'])
    df.to_csv(CSV_FILE, index=False)

# helper: mask detection (from AI model idea)
def is_mask(roi_color):
    if roi_color.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total = hsv.shape[0]*hsv.shape[1]

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask_white = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,40,255]))
    mask_black = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,40]))

    mask_combined = cv2.bitwise_or(mask_blue, mask_white)
    mask_combined = cv2.bitwise_or(mask_combined, mask_black)

    mask_pixels = cv2.countNonZero(mask_combined)
    mask_fraction = mask_pixels / float(total)

    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    var = np.var(gray) / 255.0
    score = mask_fraction * 0.8 + (1 - var) * 0.2
    return (score >= MASK_SCORE_THRESHOLD), score




# --- keep in-memory set to stop duplicate attendance in same run --- #
logged_today = set()

# helper: log attendance avoiding duplicates
def log_attendance(person_id, name, mask_label, confidence):
    global logged_today
    df = pd.read_csv(CSV_FILE)

    # Skip if already logged (in file or memory)
    if person_id in logged_today or (
        LOG_ONCE_PER_DAY and not df[df['ID'] == person_id].empty
    ):
        return False

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = {'ID': person_id, 'Name': name, 'Time': ts, 'Mask': mask_label, 'Confidence': confidence}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    logged_today.add(person_id)  # Remember this ID to avoid re-logging

    if SAVE_SNAPSHOT_ON_LOG:
        snap_path = os.path.join(SNAPSHOT_DIR, f"{person_id}_{int(time.time())}.jpg")
        cv2.imwrite(snap_path, current_frame)
    print("Logged:", new_row)
    return True


cap = cv2.VideoCapture(0)


if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press 'q' to quit. One attendance per person per day will be logged.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        person_name = "Unknown"
        person_id = f"unk_{x}_{y}"
        conf = None

        if recognizer is not None:
            try:
                id_pred, confidence = recognizer.predict(roi_gray)
                conf = confidence


                if confidence < CONFIDENCE_THRESHOLD:
                    person_name = names.get(id_pred, "Unknown")
                    person_id = str(id_pred)
                else:
                    person_name = "Unknown"
                    person_id = f"unk_{int(confidence)}"
            except Exception as e:
                person_name = "Unknown"

        lh = int(h * 0.45)
        lower_face = roi_color[lh:, :]
        mask_bool, mask_score = is_mask(lower_face)
        mask_label = "Mask" if mask_bool else "No Mask"

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (255,255,0), 2)

        color = (0,255,0) if mask_bool else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        text = f"{person_name} | {mask_label}"
        if conf is not None:
            text += f" | {int(conf)}"


        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if person_name != "Unknown":
            log_attendance(person_id, person_name, mask_label, conf if conf is not None else 0)

    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("Smart Attendance (Press q to quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





