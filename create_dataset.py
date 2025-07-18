import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue
    for img_path in os.listdir(dir_path):
        data_aux = []

        x_ = []
        y_ = []

        try:
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Standardize to exactly 21 landmarks (full hand)
                if len(hand_landmarks.landmark) != 21:
                    print(f"Skipping image {img_path} - detected {len(hand_landmarks.landmark)} landmarks (expected 21)")
                    continue
                    
                # Collect normalized coordinates
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                
                # Verify we have 42 features (x,y for 21 landmarks)
                if len(data_aux) != 42:
                    print(f"Skipping image {img_path} - invalid feature count: {len(data_aux)}")
                    continue
                    
                data.append(data_aux)
                labels.append(dir_)
                print(f"Processed {img_path} successfully")

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
