import os
import cv2
import string


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

letters = list(string.ascii_uppercase)  # A-Z
number_of_classes = len(letters)
dataset_size = 50

cap = cv2.VideoCapture(0)
for letter in letters:
    class_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for letter {letter}')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready for {letter}? Press "Q"!', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.putText(frame, f'Capturing {letter}: {counter+1}/{dataset_size}', 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, letter, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
