import cv2, os, numpy as np





def create_dataset(name, max_images=75, save_dir='dataset'):
    name = name.strip().lower()
    person_dir = os.path.join(save_dir, name) 
    os.makedirs(person_dir, exist_ok=True)

    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam. Check your camera.")




    count = 0
    print(f"📸 Capturing dataset for '{name}'.")
    print("⚙️  Press 'q' to quit early. Need clear face, eyes visible, and not blurred.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Only proceed if exactly ONE face is detected
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]

            # bluur detection check for validity
            blur_score = cv2.Laplacian(face_img, cv2.CV_64F).var()
            if blur_score < 60:  # adjust threshold as needed
                cv2.putText(frame, "❌ Too blurry!", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Dataset Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue





            #eye  detection check for validity
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) < 2:
                cv2.putText(frame, "❌ Eyes not visible!", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Dataset Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            #if all done then save the faceimage
            count += 1
            file_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Saved {count}/{max_images}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        elif len(faces) > 1:
            cv2.putText(frame, "❌ Multiple faces detected!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "❌ No clear face!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Dataset Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or count >= max_images:
            break

  
  
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Captured {count} clean images for '{name}' in {person_dir}")






if __name__ == "__main__":
    n = input("Enter your name (no spaces): ").strip()
    create_dataset(n, max_images=50)







