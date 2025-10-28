import cv2, os, numpy as np
from PIL import Image






DATASET_DIR = 'dataset'
TRAINER_DIR = 'trainer'
os.makedirs(TRAINER_DIR, exist_ok=True)





def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_samples = []
    ids = []
    names = {}
    current_id = 0

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        names[current_id] = person_name
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = Image.open(img_path).convert('L')  # grayscale
            img_np = np.array(img, 'uint8')
            face_samples.append(img_np)
            ids.append(current_id)
        current_id += 1

    if len(face_samples) == 0:
        print("No images found in dataset. Run dataset_creator.py first.")
        return

    recognizer.train(face_samples, np.array(ids))
    recognizer.save(os.path.join(TRAINER_DIR, 'trainer.yml'))
    np.save(os.path.join(TRAINER_DIR, 'names.npy'), names)
    print(f"Trained on {current_id} persons. Model saved to {TRAINER_DIR}/trainer.yml")




if __name__ == "__main__":
    train()